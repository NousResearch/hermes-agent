/** Shared ambient-floor estimator for capture endpointing and barge-in. */
export class AdaptiveAcousticThreshold {
  private floor = 0.01

  constructor(private readonly maximumFloor = 0.08) {}

  get noiseFloor(): number {
    return this.floor
  }

  get startThreshold(): number {
    return Math.min(1, this.floor + 0.045)
  }

  get endThreshold(): number {
    return Math.min(1, this.floor + 0.02)
  }

  /** Call only for known-quiet samples; callers freeze this during speech/TTS. */
  observeQuiet(rawLevel: number): void {
    const level = Math.max(0, Math.min(this.maximumFloor, Number.isFinite(rawLevel) ? rawLevel : 0))

    if (level >= this.startThreshold) {
      return
    }

    this.floor = Math.min(this.maximumFloor, this.floor * 0.92 + level * 0.08)
  }
}
