/** Return a readable, grapheme-safe label for a compact profile control. */
export function profileShortLabel(label: string, maxGraphemes = 1): string {
  const trimmed = label.trim()

  if (!trimmed) {
    return '?'
  }

  const graphemes =
    typeof Intl.Segmenter === 'function'
      ? Array.from(new Intl.Segmenter(undefined, { granularity: 'grapheme' }).segment(trimmed), part => part.segment)
      : Array.from(trimmed)

  return graphemes.slice(0, maxGraphemes).join('').trimEnd() || '?'
}
