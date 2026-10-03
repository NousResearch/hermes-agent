/** Span/trace duration (seconds) → "120ms" / "1.50s" / "2m 3s". */
export function fmtDuration(s: number): string {
  if (s < 1) {
    return `${Math.round(s * 1000)}ms`
  }

  if (s < 60) {
    return `${s.toFixed(s < 10 ? 2 : 1)}s`
  }

  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}
