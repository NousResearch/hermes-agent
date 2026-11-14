// Controls, format characters (bidi overrides, zero-width joiners) and line
// separators: invisible text a plugin string could use to spoof or reflow the
// surface it lands on.
const UNSAFE_CHARS = /[\p{Cc}\p{Cf}\u2028\u2029]/gu

/** A plugin-supplied string as one line of plain text: unsafe characters
 *  removed, whitespace collapsed, shortened with an ellipsis past `maxChars`.
 *  `null` for anything that is not a non-empty string. */
export function plainText(value: unknown, maxChars: number): null | string {
  if (typeof value !== 'string') {
    return null
  }

  const text = value.replace(UNSAFE_CHARS, '').replace(/\s+/g, ' ').trim()
  const chars = [...text]

  if (chars.length === 0) {
    return null
  }

  return chars.length > maxChars
    ? `${chars
        .slice(0, maxChars - 1)
        .join('')
        .trimEnd()}…`
    : text
}
