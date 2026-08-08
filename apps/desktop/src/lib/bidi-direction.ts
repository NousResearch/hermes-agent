export type TextDirection = 'ltr' | 'rtl'

const RTL_STRONG_RE =
  /[\p{Script=Adlam}\p{Script=Arabic}\p{Script=Hebrew}\p{Script=Nko}\p{Script=Syriac}\p{Script=Thaana}]/u

const LETTER_RE = /\p{Letter}/u

const LEADING_DIRECTIVE_RE = /^@[\w-]{1,64}:(?:`[^`]*`|"[^"]*"|'[^']*'|[^\s]+)/u
const LEADING_INLINE_CODE_RE = /^(`+)[\s\S]*?\1/u

const LEADING_LATIN_LABEL_RE =
  /^(?:[\p{Script=Latin}\p{Number}][\p{Script=Latin}\p{Number}._+:/#@-]*)(?:\s+\([\p{Script=Latin}\p{Number}\s._+:/#@-]+\))?/u

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

function startsWithLatinLabelThenRtl(text: string) {
  const label = text.match(LEADING_LATIN_LABEL_RE)?.[0]

  return label ? firstStrongDirection(text.slice(label.length)) === 'rtl' : false
}

export function resolveTextDirection(text: string, fallback: TextDirection = 'ltr'): TextDirection {
  const afterSpecialStart = stripLeadingDirectionalTokens(text)
  const afterSpecialDirection = firstStrongDirection(afterSpecialStart)

  if (afterSpecialDirection === 'rtl' || startsWithLatinLabelThenRtl(afterSpecialStart)) {
    return 'rtl'
  }

  return dominantStrongDirection(text) ?? afterSpecialDirection ?? firstStrongDirection(text) ?? fallback
}

export function syncElementTextDirection(element: HTMLElement, text: string) {
  element.dir = text.trim() ? resolveTextDirection(text) : 'auto'
}
