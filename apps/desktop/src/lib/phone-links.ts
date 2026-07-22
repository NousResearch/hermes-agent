const PHONE_CHARACTER_RE = /^\+?[\d().\s-]+$/u
const PHONE_GROUP_RE = /\d+/g
const PHONE_CANDIDATE_RE = /(?:\+\d(?:[\d() ]|[.-](?!\s)){5,}\d|\(\d{2,4}\)(?:[\d() ]|[.-](?!\s)){4,}\d|\d{2,4}[- ]\d{2,4}(?:[- ]\d{2,4}){1,4})/gu
const PROTECTED_MARKDOWN_RE = /(<(?:https?|mailto|tel):[^>\n]+>|!?\[(?:[^\u005b\u005d\n]|\[[^\u005b\u005d\n]*\])*\](?:\((?:[^()\n]|\([^()\n]*\))*\)|\[[^\]\n]*\])?|(?:https?:\/\/|www\.|(?:[\p{L}\p{N}-]+\.)+[\p{L}]{2,})[^\s<]*|mailto:[^\s<]+|[\p{L}\p{N}._%+-]+@(?:[\p{L}\p{N}-]+\.)+[\p{L}]{2,}|tel:\s+\+?\d[\d(). -]{5,}\d|tel:\+?\d[\d().-]*\d)/giu
const PHONE_CONTEXT_BEFORE_RE = /(?:call|contact|fax|mobile|phone|sms|tel(?:ephone)?|whatsapp)[^\p{L}\p{N}]{0,16}$/iu
const PHONE_CONTEXT_AFTER_RE = /^[^\p{L}\p{N}]{0,16}(?:call|contact|fax|mobile|phone|sms|tel(?:ephone)?|whatsapp)\b/iu
const PERSON_LABEL_BEFORE_RE = /(?:^|\s)\p{Lu}[\p{L}'’-]+(?:\s+\p{Lu}[\p{L}'’-]+){1,3}\s*:[^\p{L}\p{N}]{0,16}$/u
const IDENTIFIER_BOUNDARY_RE = /[\p{L}\p{N}_@]/u
const IDENTIFIER_PREFIX_RE = /[\p{L}\p{N}_]+-$/u
const DATE_RE = /^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})$/u
const IPV4_RE = /^(?:\d{1,3}\.){3}\d{1,3}$/u
const SSN_RE = /^\+?\d{3}-\d{2}-\d{4}$/u
const DIMENSION_RE = /^\+?\d{4}\s+\d{3,4}\s+\d{2,4}$/u
const ORDER_NUMBER_RE = /^\+?\d{4}-\d{4}-\d{2}$/u

function hasPhoneNumberGrouping(groups: string[]): boolean {
  if (groups.length === 3) {
    const [area, exchange, subscriber] = groups

    return area.length >= 2 && area.length <= 3 && exchange.length >= 3 && exchange.length <= 4 && subscriber.length === 4
  }

  if (groups.length === 4) {
    const [area, first, second, last] = groups

    return (
      area.length >= 2 &&
      area.length <= 3 &&
      first.length >= 2 &&
      first.length <= 3 &&
      second.length >= 2 &&
      second.length <= 3 &&
      last.length === 2
    )
  }

  return groups.length === 5 && groups.every(group => group.length === 2)
}

function hasInternationalPhoneNumberGrouping(groups: string[]): boolean {
  return (
    hasPhoneNumberGrouping(groups) ||
    (groups.length === 5 && groups[0].length === 1 && groups.slice(1).every(group => group.length === 2))
  )
}

function trimInternationalPhoneCandidate(value: string): string {
  if (!value.startsWith('+')) {
    return value
  }

  const groups = Array.from(value.matchAll(PHONE_GROUP_RE), match => ({
    index: match.index,
    value: match[0]
  }))

  const hasOptionalTrunkPrefix = groups[1]?.value === '0' && /^\+\d{1,3}\s*\(0\)/u.test(value)
  const nationalStart = hasOptionalTrunkPrefix ? 2 : 1

  for (let nextGroup = nationalStart + 3; nextGroup < groups.length; nextGroup += 1) {
    const nationalGroups = groups.slice(nationalStart, nextGroup).map(group => group.value)

    if (hasInternationalPhoneNumberGrouping(nationalGroups)) {
      return value.slice(0, groups[nextGroup].index).trimEnd()
    }
  }

  return value
}

export function phoneLinkTarget(rawValue: string): string | null {
  const value = rawValue.trim()
  const comparableValue = value.startsWith('+') ? value.slice(1) : value

  if (
    !PHONE_CHARACTER_RE.test(value) ||
    DATE_RE.test(comparableValue) ||
    IPV4_RE.test(comparableValue) ||
    SSN_RE.test(value) ||
    DIMENSION_RE.test(value) ||
    ORDER_NUMBER_RE.test(value)
  ) {
    return null
  }

  const groups = value.match(PHONE_GROUP_RE) ?? []
  const hasInternationalPrefix = value.startsWith('+')
  const normalizedValue = hasInternationalPrefix ? value.replace(/\(0\)/u, '') : value
  const digits = normalizedValue.replace(/\D/gu, '')
  const digitCount = digits.length

  if (digitCount < 7 || digitCount > 15 || (!hasInternationalPrefix && !hasPhoneNumberGrouping(groups))) {
    return null
  }

  return `tel:${hasInternationalPrefix ? '+' : ''}${digits}`
}

export function hasPhoneNumberContext(text: string, start: number, end: number): boolean {
  const value = text.slice(start, end)
  const before = text.slice(Math.max(0, start - 64), start)

  if (value.startsWith('+') || value.startsWith('(')) {
    return true
  }

  return (
    PHONE_CONTEXT_BEFORE_RE.test(before) ||
    PERSON_LABEL_BEFORE_RE.test(before) ||
    PHONE_CONTEXT_AFTER_RE.test(text.slice(end, end + 64))
  )
}

function linkifyPhoneNumbersInText(text: string): string {
  return text.replace(PHONE_CANDIDATE_RE, (value, offset: number) => {
    const previousCharacter = text[offset - 1] || ''
    const nextCharacter = text[offset + value.length] || ''
    const before = text.slice(Math.max(0, offset - 32), offset)

    if (
      IDENTIFIER_BOUNDARY_RE.test(previousCharacter) ||
      IDENTIFIER_BOUNDARY_RE.test(nextCharacter) ||
      IDENTIFIER_PREFIX_RE.test(before)
    ) {
      return value
    }

    if (!hasPhoneNumberContext(text, offset, offset + value.length)) {
      return value
    }

    const candidate = trimInternationalPhoneCandidate(value)
    const target = phoneLinkTarget(candidate)

    return target ? `[${candidate}](${target})${value.slice(candidate.length)}` : value
  })
}

export function linkifyPhoneNumbersInMarkdown(text: string): string {
  return text
    .split(PROTECTED_MARKDOWN_RE)
    .map((part, index) => (index % 2 === 1 ? part : linkifyPhoneNumbersInText(part)))
    .join('')
}

export function isProtectedMarkdownRange(text: string, start: number, end: number): boolean {
  for (const match of text.matchAll(PROTECTED_MARKDOWN_RE)) {
    const matchStart = match.index

    if (start >= matchStart && end <= matchStart + match[0].length) {
      return true
    }
  }

  return false
}
