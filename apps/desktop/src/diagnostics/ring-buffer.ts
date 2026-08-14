// Bounded in-memory event store for a diagnostics capture.
//
// Three caps, all enforced on push so the buffer can never grow between
// captures: a count cap, a byte cap (estimated at record time — a capture that
// fires thousands of long frames must not turn into a heap problem of its own),
// and an age cap so a capture left armed for an hour still exports only the
// last ~300s around the hitch the user reported.
//
// Eviction is oldest-first. Nothing here formats or serialises: the export step
// (U4) reads `entries()` and writes JSONL.

export interface RingBufferLimits {
  /** Hard cap on retained events. */
  maxEvents: number
  /** Hard cap on the estimated retained bytes. */
  maxBytes: number
  /** Retain only events within this window of the newest event. */
  maxAgeMs: number
}

/** Every diagnostics event carries a monotonic `performance.now()` stamp. */
interface TimestampedEvent {
  t: number
}

// Rough sizing, not exact: numbers are 8 bytes, strings 2 bytes per code unit,
// plus a flat per-object tax for keys/headers. Exact accounting would mean
// JSON.stringify on the hot path, which is the sort of cost this module exists
// to avoid attributing to the app.
const OBJECT_OVERHEAD_BYTES = 48

export function estimateEventBytes(value: unknown): number {
  if (typeof value === 'number' || typeof value === 'boolean') {
    return 8
  }

  if (typeof value === 'string') {
    return value.length * 2
  }

  if (Array.isArray(value)) {
    return value.reduce<number>((total, item) => total + estimateEventBytes(item), OBJECT_OVERHEAD_BYTES)
  }

  if (value && typeof value === 'object') {
    let total = OBJECT_OVERHEAD_BYTES

    for (const [key, entry] of Object.entries(value)) {
      total += key.length * 2 + estimateEventBytes(entry)
    }

    return total
  }

  return 0
}

export class DiagnosticsRingBuffer<T extends TimestampedEvent> {
  private readonly limits: RingBufferLimits
  // Deque backed by an array plus a read cursor: dropping the oldest event is
  // an index bump, not an O(n) shift, and the dead prefix is compacted only
  // when it grows past the live window.
  private items: T[] = []
  private head = 0
  private bytes = 0
  private byteSizes: number[] = []
  private dropped = 0

  constructor(limits: RingBufferLimits) {
    this.limits = limits
  }

  push(event: T): T {
    const size = estimateEventBytes(event)
    this.items.push(event)
    this.byteSizes.push(size)
    this.bytes += size
    this.evict(event.t)

    return event
  }

  entries(): T[] {
    return this.items.slice(this.head)
  }

  get size(): number {
    return this.items.length - this.head
  }

  get byteSize(): number {
    return this.bytes
  }

  /** How many events eviction has discarded since the buffer was created. */
  get droppedCount(): number {
    return this.dropped
  }

  clear(): void {
    this.items = []
    this.byteSizes = []
    this.head = 0
    this.bytes = 0
    this.dropped = 0
  }

  private evict(newestAt: number): void {
    while (
      this.size > 0 &&
      (this.size > this.limits.maxEvents ||
        this.bytes > this.limits.maxBytes ||
        newestAt - this.items[this.head].t > this.limits.maxAgeMs)
    ) {
      this.bytes -= this.byteSizes[this.head]
      this.head += 1
      this.dropped += 1
    }

    // Compact once the discarded prefix outweighs the live window, so the
    // backing arrays track the retained events rather than the whole capture.
    if (this.head > 0 && this.head >= this.items.length - this.head) {
      this.items = this.items.slice(this.head)
      this.byteSizes = this.byteSizes.slice(this.head)
      this.head = 0
    }
  }
}
