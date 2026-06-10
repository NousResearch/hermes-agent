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
    .replace(/(?:^|[.!?]\s+)[-+*]\s+/g, match => match.replace(/[-+*]\s+$/, ''))
    .replace(/\s[-+*]\s+/g, ' ')
    .replace(/[*_~>#]/g, '')
    .replace(/^\s*[-+*]\s+/gm, '')
    .replace(/([.!?])\s*\.\s+/g, '$1 ')
    .replace(/\s+/g, ' ')
    .trim()
}

const SPOKEN_DIGEST_SHORT_CHARS = 220
const SPOKEN_DIGEST_TARGET_CHARS = 300
const SPOKEN_DIGEST_MAX_CHARS = 380
const SPOKEN_DIGEST_VERDICT_MAX_CHARS = 140

function truncateAtSpeechBoundary(text: string, maxChars = SPOKEN_DIGEST_MAX_CHARS): string {
  if (text.length <= maxChars) {
    return text
  }

  const slice = text.slice(0, maxChars)
  const boundary = Math.max(
    slice.lastIndexOf('. '),
    slice.lastIndexOf('! '),
    slice.lastIndexOf('? '),
    slice.lastIndexOf('; '),
    slice.lastIndexOf(', '),
    slice.lastIndexOf(' ')
  )

  return `${slice.slice(0, boundary > 120 ? boundary : maxChars).trim()}…`
}

function splitSpeechSentences(text: string): string[] {
  return (text.match(/[^.!?。！？]+[.!?。！？]+(?:\s+|$)|[^.!?。！？]+$/g) ?? [text])
    .map(sentence => sentence.trim())
    .filter(Boolean)
}

function isVerdictSentence(sentence: string): boolean {
  return sentence.length <= SPOKEN_DIGEST_VERDICT_MAX_CHARS && /^(?:yes|no|partially|good|done|fixed|working|healthy|blocked|not yet|mostly)\b/i.test(sentence)
}

export function buildSpokenDigest(text: string): string {
  const speakableText = sanitizeTextForSpeech(text)

  if (!speakableText) {
    return ''
  }

  const sentences = splitSpeechSentences(speakableText)

  if (speakableText.length <= SPOKEN_DIGEST_SHORT_CHARS || sentences.length <= 2) {
    return truncateAtSpeechBoundary(speakableText)
  }

  const verdict = isVerdictSentence(sentences[0]) ? sentences[0] : null
  const tail: string[] = []
  let total = verdict ? verdict.length + 1 : 0

  for (let index = sentences.length - 1; index >= 0; index -= 1) {
    const sentence = sentences[index]

    if (!sentence || sentence === verdict) {
      continue
    }

    if (tail.length > 0 && total + sentence.length > SPOKEN_DIGEST_TARGET_CHARS) {
      break
    }

    tail.unshift(sentence)
    total += sentence.length + 1

    if (tail.length >= 2 || total >= SPOKEN_DIGEST_TARGET_CHARS) {
      break
    }
  }

  const digest = [verdict, ...tail].filter(Boolean).join(' ')
  return truncateAtSpeechBoundary(digest || speakableText)
}
