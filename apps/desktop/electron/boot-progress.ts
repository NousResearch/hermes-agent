/** Normalize backend boot progress into the renderer's 0–100 display range. */
export function clampBootProgress(value: unknown): number {
  const numeric = Number(value)

  return Number.isFinite(numeric) ? Math.max(0, Math.min(100, Math.round(numeric))) : 0
}
