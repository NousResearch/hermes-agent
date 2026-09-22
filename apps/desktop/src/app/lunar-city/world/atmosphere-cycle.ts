/** Presentation time only: a paused tab never catches up to a backend or wall clock. */
export function createAtmosphereClock(durationMs = 20 * 60 * 1000) {
  let phase = .27

  return {
    setPhase(value: number) {
      if (!Number.isFinite(value)) {throw new Error('Atmosphere phase must be finite')}
      phase = ((value % 1) + 1) % 1
    },
    advance(elapsedMs: number, paused: boolean) {
      if (!paused && Number.isFinite(elapsedMs)) {phase = (phase + Math.max(0, Math.min(elapsedMs, 1000)) / durationMs) % 1}
      const elevation = Math.sin(phase * Math.PI * 2)
      const daylight = Math.max(0, elevation)

      return { phase, elevation, daylight, keyIntensity: .28 + 1.15 * daylight, fillIntensity: .42 + .25 * daylight }
    }
  }
}
