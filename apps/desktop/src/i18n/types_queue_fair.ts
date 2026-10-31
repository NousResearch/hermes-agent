/** Server-queued rows: why each one waits, and moving it in the owner's FIFO. */
export interface QueueFairCopy {
  /** A hold reason the owner names that this build has no label for. */
  holdUnknown: (reason: string) => string
  /** Known hold reasons, keyed by the owner's `hold_reason`. */
  hold: Record<string, string>
  moveUp: string
  moveDown: string
  /** Toast title: the owner could not hold a row open for editing (it may start before Save). */
  holdFailed: string
}
