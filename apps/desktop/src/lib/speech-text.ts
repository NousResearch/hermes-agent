const EMOJI_RE = /(?:[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]|[\u{FE0F}\u{200D}]|[\u{E0020}-\u{E007F}])+/gu

const FENCED_CODE_RE = /```[\s\S]*?(?:```|$)/g
const INLINE_CODE_RE = /`([^`]+)`/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g
const PARAGRAPH_BREAK_RE = /[ \t]*\n{2,}[ \t]*/g
const PUNCTUATED_PARAGRAPH_BREAK_RE = /([.!?])([*_~`>"'’”)}\]]*)[ \t]*\n{2,}[ \t]*/g
const SOFT_BREAK_RE = /[ \t]*\n[ \t]*/g

// A file-link token ("MEDIA:/path/to/report.xlsx") renders as a chip on
// screen; spoken, its hyphenated slug makes voices loop. It is silence, but a
// sentence-final period/comma after it is kept ("see MEDIA:/x.py. Then").
const MEDIA_PATH_RE = /[ \t]*MEDIA:\S+?(?=[.,;:!?)\]]*(?:\s|$))/g
const LINE_FINAL_COLON_RE = /:\s*$/gm

const THINKING_PREFIX_RE =
  /^\s*(?:\([^)\n]{1,48}\)\s*)?(?:processing|thinking|reasoning|analyzing|pondering|contemplating|musing|cogitating|ruminating|deliberating|mulling|reflecting|computing|synthesizing|formulating|brainstorming)\.\.\.\s*/i

const URL_RE = /\bhttps?:\/\/\S+/gi
const CLOSED_THINK_BLOCK_RE = /<think[\s>][\s\S]*?<\/think>/g

const MARKDOWN_TABLE_DELIMITER_CELL_RE = /^:?-{3,}:?$/

interface MarkdownTableRow {
  blockquoteDepth: number
  cells: string[]
}

function isUnescapedPipe(row: string, index: number): boolean {
  let backslashes = 0

  for (let cursor = index - 1; cursor >= 0 && row[cursor] === '\\'; cursor -= 1) {
    backslashes += 1
  }

  return backslashes % 2 === 0
}

function splitMarkdownTableCells(row: string): string[] {
  const cells: string[] = []
  let cellStart = 0

  for (let index = 0; index < row.length; index += 1) {
    if (row[index] === '|' && isUnescapedPipe(row, index)) {
      cells.push(row.slice(cellStart, index).trim())
      cellStart = index + 1
    }
  }

  cells.push(row.slice(cellStart).trim())

  return cells
}

function parseMarkdownTableRow(line: string): MarkdownTableRow | null {
  let row = line
  let blockquoteDepth = 0

  while (true) {
    const indentation = row.match(/^[ \t]*/)?.[0] ?? ''

    if (indentation.includes('\t') || indentation.length > 3) {
      return null
    }

    row = row.slice(indentation.length)

    if (!row.startsWith('>')) {
      break
    }

    blockquoteDepth += 1
    row = row.slice(1)

    if (row.startsWith(' ')) {
      row = row.slice(1)
    }
  }

  row = row.trimEnd()

  const pipeIndexes = [...row.matchAll(/\|/g)].map(match => match.index).filter(index => isUnescapedPipe(row, index))

  if (pipeIndexes.length === 0) {
    return null
  }

  const hasLeadingPipe = pipeIndexes[0] === 0
  const hasTrailingPipe = pipeIndexes.at(-1) === row.length - 1

  if (hasLeadingPipe) {
    row = row.slice(1)
  }

  if (hasTrailingPipe) {
    row = row.slice(0, -1)
  }

  const cells = splitMarkdownTableCells(row)

  if (cells.length < 2 && !(hasLeadingPipe && hasTrailingPipe && cells.length === 1)) {
    return null
  }

  return { blockquoteDepth, cells }
}

// The header row is spoken in place of the table ("Model, Price, Context."):
// the listener learns a table is on screen and what it compares, in the reply's
// own language, without the body data being read cell by cell (#86602). A
// table with an empty header stays silent — there is nothing to announce.
function speakableTableHeader(cells: string[]): string {
  const header = cells
    .map(cell => cell.replace(/\\\|/g, ' ').trim())
    .filter(Boolean)
    .join(', ')

  return header && !/[.!?:]$/.test(header) ? `${header}.` : header
}

function summarizeMarkdownTables(text: string): string {
  const lines = text.replace(/\r\n?/g, '\n').split('\n')
  const tableLines = new Set<number>()
  const headers = new Map<number, string>()

  let index = 1

  while (index < lines.length) {
    const delimiterRow = parseMarkdownTableRow(lines[index])
    const headerRow = parseMarkdownTableRow(lines[index - 1])

    if (
      !delimiterRow ||
      !headerRow ||
      !delimiterRow.cells.every(cell => MARKDOWN_TABLE_DELIMITER_CELL_RE.test(cell)) ||
      headerRow.cells.length !== delimiterRow.cells.length ||
      headerRow.blockquoteDepth !== delimiterRow.blockquoteDepth
    ) {
      index += 1

      continue
    }

    tableLines.add(index - 1)
    tableLines.add(index)
    headers.set(index - 1, speakableTableHeader(headerRow.cells))

    let rowIndex = index + 1

    for (; rowIndex < lines.length; rowIndex += 1) {
      const bodyRow = parseMarkdownTableRow(lines[rowIndex])

      if (!bodyRow || bodyRow.blockquoteDepth !== delimiterRow.blockquoteDepth) {
        break
      }

      tableLines.add(rowIndex)
    }

    index = rowIndex
  }

  return lines
    .flatMap((line, index) => {
      if (!tableLines.has(index)) {
        return [line]
      }

      const header = headers.get(index)

      return header ? [header] : []
    })
    .join('\n')
}

function normalizeLineBreaks(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/(\p{L})-\n(\p{L})/gu, '$1$2')
    .replace(PUNCTUATED_PARAGRAPH_BREAK_RE, '$1$2 ')
    .replace(PARAGRAPH_BREAK_RE, '. ')
    .replace(SOFT_BREAK_RE, ' ')
}

// ---------------------------------------------------------------------------
// Sentence cutter for the streaming TTS session — mirrors the server-side
// SentenceChunker's contract: emit complete sentences as they form, hold
// the incomplete tail, flush everything on finish.
// ---------------------------------------------------------------------------

const SENTENCE_CUT_RE = /[.!?…。！？]+["'”’)\]]*\s+/g
const MIN_SENTENCE_CHARS = 24

export function cutSentences(
  buffer: string,
  flush: boolean,
  minSentenceChars?: null | number
): { sentences: string[]; rest: string } {
  // tts.streaming.min_len when the backend sends it (a 5–7 char CJK opener is a
  // whole clause); the historical 24 for older backends without the key.
  const minChars = minSentenceChars ?? MIN_SENTENCE_CHARS
  const sentences: string[] = []
  let rest = buffer
  let start = 0

  SENTENCE_CUT_RE.lastIndex = 0

  let match = SENTENCE_CUT_RE.exec(buffer)

  while (match) {
    const end = match.index + match[0].length
    const candidate = buffer.slice(start, end).trim()

    // Too-short fragments ("e.g. ", "1. ") stay buffered so we don't fire a
    // provider call per abbreviation — unless a later boundary extends them.
    if (candidate.length >= minChars) {
      sentences.push(candidate)
      start = end
    }

    match = SENTENCE_CUT_RE.exec(buffer)
  }

  rest = buffer.slice(start)

  if (flush) {
    const tail = rest.trim()

    if (tail) {
      sentences.push(tail)
    }

    rest = ''
  }

  return { sentences, rest }
}

/** Incremental wrapper over cutSentences() that also hides <think> blocks
 *  split across deltas. Used by the sync (non-streaming provider) fallback. */
export class IncrementalSpeechSentenceBuffer {
  private buffer = ''

  constructor(private readonly minSentenceChars?: null | number) {}

  append(delta: string): string[] {
    this.buffer = (this.buffer + delta).replace(CLOSED_THINK_BLOCK_RE, '')

    if (this.buffer.includes('<think') && !this.buffer.includes('</think>')) {
      return []
    }

    const { sentences, rest } = cutSentences(this.buffer, false, this.minSentenceChars)
    this.buffer = rest

    return sentences
  }

  flush(): string[] {
    const { sentences } = cutSentences(this.buffer.replace(CLOSED_THINK_BLOCK_RE, ''), true, this.minSentenceChars)
    this.buffer = ''

    return sentences
  }
}

export function sanitizeTextForSpeech(text: string): string {
  // Tables first: their right-align marker is a trailing colon (":-"), and
  // closing colons before the table detector runs would mangle it.
  const withoutTables = summarizeMarkdownTables(String(text))

  // Close line-final colons BEFORE newlines are flattened: "the regex list:"
  // followed by a code block keeps its colon if this runs after the flatten,
  // and the voice hangs on it. Closing early turns it into "the regex list.".
  const pre = withoutTables.replace(LINE_FINAL_COLON_RE, '.')

  // Unspeakable tokens are silence, never a placeholder word: an English
  // "code block omitted" / "link" is wrong for every non-English voice.
  return normalizeLineBreaks(pre)
    .replace(FENCED_CODE_RE, '')
    .replace(THINKING_PREFIX_RE, ' ')
    .replace(MARKDOWN_LINK_RE, '$1')
    .replace(INLINE_CODE_RE, '$1')
    .replace(URL_RE, '')
    .replace(MEDIA_PATH_RE, '')
    .replace(EMOJI_RE, ' ')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_~>#]/g, '')
    .replace(/^\s*[-+*]\s+/gm, '')
    .replace(/:\s*$/, '.') // colon orphaned when its link/code was stripped
    .replace(/\s+/g, ' ')
    .trim()
}
