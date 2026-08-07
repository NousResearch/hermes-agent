// THE compact-number formatter — every user-facing count/token figure goes
// through here. 999 → "999", 1000 → "1k", 1230 → "1.2k", 10000 → "10k",
// 1_500_000 → "1.5M". Do not hand-roll `/ 1000` display math elsewhere.
export function compactNumber(value: null | number | undefined): string {
  const num = Number(value ?? 0)

  if (!Number.isFinite(num) || num <= 0) {
    return '0'
  }

  const scaled = (v: number, suffix: string) => `${v.toFixed(1).replace(/\.0$/, '')}${suffix}`

  // Thresholds sit just under the unit boundary so rounding can't produce
  // "1000k" or "1000" — those promote to the next unit instead.
  if (num >= 999_950) {
    return scaled(num / 1_000_000, 'M')
  }

  if (num >= 999.5) {
    return scaled(num / 1_000, 'k')
  }

  return `${Math.round(num)}`
}
