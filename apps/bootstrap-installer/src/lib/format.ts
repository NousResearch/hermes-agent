/*
 * Duration formatters for the stage list. Pure functions, no React — kept out
 * of progress.tsx so tests-js can exercise them without dragging in the Tauri
 * renderer.
 */

// Duration of a completed stage: ms, then s, then "Xm Ys", then "Xh Ym".
export function formatDuration(ms: number): string {
  if (ms < 1000) {return `${ms}ms`}

  if (ms < 60000) {return `${(ms / 1000).toFixed(1)}s`}
  const m = Math.floor(ms / 60000)
  const s = Math.round((ms % 60000) / 1000)

  if (m < 60) {return `${m}m ${s}s`}
  const h = Math.floor(m / 60)

  return `${h}h ${m - h * 60}m`
}

// Live elapsed for a running stage: bare seconds under a minute, then m:ss,
// then h:mm:ss past an hour. Without the hour rollover a stalled overnight
// stage read as "744:38" — minutes rendered unbounded — which one user
// understandably reported as "744 hours".
export function formatElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))

  if (s < 60) {return `${s}s`}
  const m = Math.floor(s / 60)

  if (m < 60) {return `${m}:${String(s - m * 60).padStart(2, '0')}`}
  const h = Math.floor(m / 60)

  return `${h}:${String(m - h * 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
