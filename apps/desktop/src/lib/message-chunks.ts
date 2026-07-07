export interface MessageChunk {
  index: number
  text: string
}

const MIN_CHUNKABLE_LENGTH = 700
const MAX_CHUNK_LENGTH = 900
const MIN_CHUNK_LENGTH = 180

function hasFence(text: string): boolean {
  return /```/.test(text)
}

function normalizeChunk(text: string): string {
  return text.trim().replace(/\n{3,}/g, '\n\n')
}

function pushChunk(chunks: string[], value: string): void {
  const chunk = normalizeChunk(value)

  if (chunk) {
    chunks.push(chunk)
  }
}

function splitNumberedSections(text: string): string[] {
  const lines = text.split('\n')
  const chunks: string[] = []
  let current: string[] = []

  for (const line of lines) {
    if (/^\s*(?:\d+|[-*])\.?:?\s+\S/.test(line) && current.join('\n').trim().length >= MIN_CHUNK_LENGTH) {
      pushChunk(chunks, current.join('\n'))
      current = [line]
    } else {
      current.push(line)
    }
  }

  pushChunk(chunks, current.join('\n'))

  return chunks
}

function splitMarkdownHeadings(text: string): string[] {
  const lines = text.split('\n')
  const chunks: string[] = []
  let current: string[] = []

  for (const line of lines) {
    if (/^#{2,4}\s+\S/.test(line) && current.join('\n').trim().length >= MIN_CHUNK_LENGTH) {
      pushChunk(chunks, current.join('\n'))
      current = [line]
    } else {
      current.push(line)
    }
  }

  pushChunk(chunks, current.join('\n'))

  return chunks
}

function splitParagraphs(text: string): string[] {
  const paragraphs = text.split(/\n\s*\n/g).map(normalizeChunk).filter(Boolean)

  if (paragraphs.length <= 1) {
    return [normalizeChunk(text)]
  }

  const chunks: string[] = []
  let current = ''

  for (const paragraph of paragraphs) {
    const next = current ? `${current}\n\n${paragraph}` : paragraph

    if (current && next.length > MAX_CHUNK_LENGTH && current.length >= MIN_CHUNK_LENGTH) {
      pushChunk(chunks, current)
      current = paragraph
    } else {
      current = next
    }
  }

  pushChunk(chunks, current)

  return chunks
}

function rebalanceTinyChunks(chunks: string[]): string[] {
  const out: string[] = []

  for (const chunk of chunks) {
    const prev = out[out.length - 1]

    if (prev && chunk.length < MIN_CHUNK_LENGTH && prev.length + chunk.length < MAX_CHUNK_LENGTH) {
      out[out.length - 1] = `${prev}\n\n${chunk}`
    } else {
      out.push(chunk)
    }
  }

  return out
}

export function splitAssistantMessageIntoReplyChunks(text: string): MessageChunk[] {
  const normalized = normalizeChunk(text)

  if (normalized.length < MIN_CHUNKABLE_LENGTH || hasFence(normalized)) {
    return [{ index: 0, text: normalized }]
  }

  const candidates = /^#{2,4}\s+\S/m.test(normalized)
    ? splitMarkdownHeadings(normalized)
    : /^\s*(?:\d+|[-*])\.?:?\s+\S/m.test(normalized)
      ? splitNumberedSections(normalized)
      : splitParagraphs(normalized)

  const chunks = rebalanceTinyChunks(candidates).filter(chunk => chunk.length >= 1)

  return (chunks.length > 1 ? chunks : [normalized]).map((chunk, index) => ({ index, text: chunk }))
}
