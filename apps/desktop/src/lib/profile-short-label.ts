const profileGraphemes = new Intl.Segmenter(undefined, { granularity: 'grapheme' })

/** Keep combining marks and emoji sequences together without changing the rail's size. */
export function profileShortLabel(label: string): string {
  for (const { segment } of profileGraphemes.segment(label.trim())) {
    if (/[\p{L}\p{N}\p{Extended_Pictographic}\p{Regional_Indicator}]/u.test(segment)) {
      return segment
    }
  }

  return '?'
}
