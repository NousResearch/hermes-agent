const EPOCH_SECONDS_CUTOFF = 100_000_000_000

export function artifactTimestampToDate(timestamp: number): Date {
  const timestampMs = timestamp > 0 && timestamp < EPOCH_SECONDS_CUTOFF ? timestamp * 1000 : timestamp

  return new Date(timestampMs)
}
