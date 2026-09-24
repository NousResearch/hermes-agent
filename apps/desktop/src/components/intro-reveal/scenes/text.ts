const SCRAMBLE_CHARS = '/\\|-_=+<>~:*'

export const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

export const scrambleGlyph = (i: number, tick: number) => {
  const n = (i * 2654435761 + tick * 40503) >>> 0

  return SCRAMBLE_CHARS[n % SCRAMBLE_CHARS.length]
}

/** Spaces stay fixed so word boundaries survive the scramble. */
export function decoded(text: string, bornAt: number, tick: number, spanMs = 520): string {
  const age = tick * 45 - bornAt
  const resolved = Math.max(0, Math.min(text.length, Math.ceil((age / spanMs) * text.length)))

  return Array.from(text, (ch, i) => (ch === ' ' || i < resolved ? ch : scrambleGlyph(i, tick))).join('')
}
