import { $textDirection, forcedTextDirection } from '@/store/text-direction'

export type TextDirection = 'ltr' | 'rtl'

const RTL_STRONG_RE =
  /[\p{Script=Adlam}\p{Script=Arabic}\p{Script=Hebrew}\p{Script=Nko}\p{Script=Syriac}\p{Script=Thaana}]/u

const LETTER_RE = /\p{Letter}/u

const LEADING_DIRECTIVE_RE = /^@[\w-]{1,64}:(?:`[^`]*`|"[^"]*"|'[^']*'|[^\s]+)/u
const LEADING_INLINE_CODE_RE = /^(`+)[\s\S]*?\1/u

const LEADING_LATIN_LABEL_TOKEN_RE =
  /^(?:[\p{Script=Latin}\p{Number}][\p{Script=Latin}\p{Number}\p{Punctuation}+]*|\([\p{Script=Latin}\p{Number}\p{Punctuation}\s+]+\))(?:\s+|$)/u
const LOWERCASE_LETTER_RE = /^\p{Lowercase_Letter}/u

const LEADING_PATH_TOKEN_RE = /^(?:\.{1,2}\/|\/|~\/|[A-Za-z]:[\\/])[^\s]+/u
const LEADING_SLASH_COMMAND_RE = /^\/[A-Za-z][\w-]*(?=\s|$)/u

function firstStrongDirection(text: string): TextDirection | null {
  for (const ch of text) {
    if (RTL_STRONG_RE.test(ch)) {
      return 'rtl'
    }

    if (LETTER_RE.test(ch)) {
      return 'ltr'
    }
  }

  return null
}

function dominantStrongDirection(text: string): TextDirection | null {
  let ltr = 0
  let rtl = 0

  for (const ch of text) {
    if (RTL_STRONG_RE.test(ch)) {
      rtl += 1
    } else if (LETTER_RE.test(ch)) {
      ltr += 1
    }
  }

  if (rtl > ltr) {
    return 'rtl'
  }

  if (ltr > 0) {
    return 'ltr'
  }

  return rtl > 0 ? 'rtl' : null
}

function stripLeadingNonStrong(text: string) {
  let index = 0

  for (const ch of text) {
    if (RTL_STRONG_RE.test(ch) || LETTER_RE.test(ch)) {
      break
    }

    index += ch.length
  }

  return text.slice(index)
}

function stripOneLeadingToken(text: string) {
  const trimmed = text.trimStart()

  const token =
    trimmed.match(LEADING_INLINE_CODE_RE)?.[0] ??
    trimmed.match(LEADING_DIRECTIVE_RE)?.[0] ??
    trimmed.match(LEADING_SLASH_COMMAND_RE)?.[0] ??
    trimmed.match(LEADING_PATH_TOKEN_RE)?.[0]

  return token ? trimmed.slice(token.length) : stripLeadingNonStrong(trimmed)
}

function stripLeadingDirectionalTokens(text: string) {
  let next = text

  for (let i = 0; i < 8; i += 1) {
    const stripped = stripOneLeadingToken(next)

    if (stripped === next) {
      return stripped
    }

    next = stripped
  }

  return next
}

function isLabelShapedToken(token: string) {
  const word = token.trim()

  // Technical tokens ("existing:", "Task16", "GPT-5.6") and acronyms ("NOOP")
  // read as labels; a plain capitalized word does not, because every English
  // sentence starts with one.
  return (
    /[\d\p{Punctuation}]/u.test(word) ||
    (word.length > 1 && /\p{Letter}/u.test(word) && word === word.toUpperCase())
  )
}

function lowercaseLatinContinuesAfterRtl(text: string) {
  const tokens = text.split(/\s+/u)
  const firstRtl = tokens.findIndex(token => RTL_STRONG_RE.test(token))

  for (const token of tokens.slice(firstRtl + 1)) {
    const first = token.charAt(0)

    if (LETTER_RE.test(first)) {
      return LOWERCASE_LETTER_RE.test(first)
    }
  }

  return false
}

// Does Latin content (a letter or number) follow the line's first RTL word?
// A trailing Arabic word is a quotation, never a brand tail.
function latinFollowsRtlWord(text: string) {
  const tokens = text.split(/\s+/u)
  const firstRtl = tokens.findIndex(token => RTL_STRONG_RE.test(token))

  return tokens
    .slice(firstRtl + 1)
    .some(token => /[\p{Letter}\p{Number}]/u.test(token.charAt(0)))
}

// Direction concluded from a leading Latin label/prefix, or null when the
// input has no RTL content reachable that way (caller falls through to the
// dominant script).
function leadingLabelDirection(text: string): TextDirection | null {
  let remainder = text.trimStart()

  for (let i = 0; i < 8; i += 1) {
    const token = remainder.match(LEADING_LATIN_LABEL_TOKEN_RE)?.[0]

    if (!token) {
      return null
    }

    remainder = remainder.slice(token.length)

    if (firstStrongDirection(remainder) !== 'rtl') {
      continue
    }

    const rtlWordCount = remainder.split(/\s+/u).filter(token => RTL_STRONG_RE.test(token)).length

    if (rtlWordCount > 1) {
      return 'rtl'
    }

    // One RTL word. An English sentence that merely quotes one Arabic word
    // stays LTR — lowercase prose continuing past the word ("explain what
    // مرحبا means?") or a trailing word after a bare lead ("Try مرحبا").
    // Only a label-shaped token ("Learning terminal existing: NOOP
    // مايتحولش.") or brand position in a mixed line ("Google عندها Gemini
    // 3.5", Latin on both sides) flips the line.
    if (lowercaseLatinContinuesAfterRtl(remainder)) {
      return 'ltr'
    }

    if (isLabelShapedToken(token)) {
      return 'rtl'
    }

    return i === 0 && latinFollowsRtlWord(remainder) ? 'rtl' : 'ltr'
  }

  return null
}

export function resolveTextDirection(text: string, fallback: TextDirection = 'ltr'): TextDirection {
  const afterSpecialStart = stripLeadingDirectionalTokens(text)
  const afterSpecialDirection = firstStrongDirection(afterSpecialStart)

  if (afterSpecialDirection === 'rtl') {
    return 'rtl'
  }

  const leadingDirection = leadingLabelDirection(afterSpecialStart)

  if (leadingDirection !== null) {
    return leadingDirection
  }

  return dominantStrongDirection(text) ?? afterSpecialDirection ?? firstStrongDirection(text) ?? fallback
}

export function syncElementTextDirection(element: HTMLElement, text: string) {
  // Appearance → Text direction (forced RTL/LTR) outranks the resolver. Under
  // Auto, resolve from the draft so an Arabic sentence that opens with a Latin
  // brand does not stay LTR.
  const forced = forcedTextDirection($textDirection.get())

  element.dir = forced ?? (text.trim() ? resolveTextDirection(text) : 'auto')
}
