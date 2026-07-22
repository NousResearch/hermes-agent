const EMOJI_RE = /(?:[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}]|[\u{FE0F}\u{200D}]|[\u{E0020}-\u{E007F}])+/gu

const FENCED_CODE_RE = /```[\s\S]*?(?:```|$)/g
const INLINE_CODE_RE = /`([^`]+)`/g
const MARKDOWN_LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g
const PARAGRAPH_BREAK_RE = /[ \t]*\n{2,}[ \t]*/g
const SOFT_BREAK_RE = /[ \t]*\n[ \t]*/g

const THINKING_PREFIX_RE =
  /^\s*(?:\([^)\n]{1,48}\)\s*)?(?:processing|thinking|reasoning|analyzing|pondering|contemplating|musing|cogitating|ruminating|deliberating|mulling|reflecting|computing|synthesizing|formulating|brainstorming)\.\.\.\s*/i

const URL_RE = /\bhttps?:\/\/\S+/gi

function normalizeLineBreaks(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/(\p{L})-\n(\p{L})/gu, '$1$2')
    .replace(PARAGRAPH_BREAK_RE, '. ')
    .replace(SOFT_BREAK_RE, ' ')
}

export function sanitizeTextForSpeech(text: string): string {
  return normalizeLineBreaks(text)
    .replace(FENCED_CODE_RE, ' ')
    .replace(THINKING_PREFIX_RE, ' ')
    .replace(MARKDOWN_LINK_RE, '$1')
    .replace(INLINE_CODE_RE, '$1')
    .replace(URL_RE, ' link ')
    .replace(EMOJI_RE, ' ')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_~>#]/g, '')
    .replace(/^\s*[-+*]\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim()
}


export function splitSpeechText(text: string, maxChars = 4_500): string[] {
  let remaining = text.trim()

  if (!remaining) {
    return []
  }

  const chunks: string[] = []

  while (remaining.length > maxChars) {
    const prefix = remaining.slice(0, maxChars + 1)
    const sentenceBoundary = /[.!?](?:["')\]]*)\s+/g
    let cut = -1
    let match: RegExpExecArray | null

    while ((match = sentenceBoundary.exec(prefix)) !== null) {
      cut = match.index + match[0].trimEnd().length
    }

    if (cut < Math.floor(maxChars / 2)) {
      cut = remaining.lastIndexOf(' ', maxChars)
    }

    if (cut <= 0) {
      cut = maxChars
    }

    chunks.push(remaining.slice(0, cut).trimEnd())
    remaining = remaining.slice(cut).trimStart()
  }

  if (remaining) {
    chunks.push(remaining)
  }

  return chunks
}
