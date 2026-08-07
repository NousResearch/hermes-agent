/**
 * ManagedTimer: a named, lifecycle-managed wrapper around setTimeout.
 *
 * Eliminates repetitive clear→set→fire→clean patterns by providing:
 * - `schedule(delayMs, cb)` — cancel any pending timer, set a new one
 * - `scheduleAt(epochMs, cb)` — schedule by absolute time point
 * - `tryAdvanceTo(epochMs, cb)` — only reschedule if new time is *earlier*
 * - `cancel()` — cancel without triggering
 * - `flush()` — trigger immediately (for graceful shutdown)
 * - `pending` — whether a timer is waiting
 *
 * The optional `isDestroyed` guard prevents firing after the owner is torn down.
 */

type TimerHandle = ReturnType<typeof setTimeout>;

export class ManagedTimer {
  private handle: TimerHandle | null = null;
  private callback: (() => void) | null = null;
  /** Absolute epoch-ms when the current timer is scheduled to fire. */
  private scheduledAt = 0;

  constructor(
    /** Human-readable name for logging. */
    public readonly name: string,
    /** If provided, checked before firing — skips callback when true. */
    private readonly isDestroyed?: () => boolean,
  ) {}

  // ── Core operations ──────────────────────────────────

  /**
   * Cancel any pending timer and schedule a new one after `delayMs`.
   * The callback fires once; the timer auto-clears after firing.
   */
  schedule(delayMs: number, callback: () => void): void {
    this.cancelInternal();
    this.callback = callback;
    this.scheduledAt = Date.now() + delayMs;
    this.handle = setTimeout(() => this.fire(), delayMs);
    // Don't let pipeline timers keep the process alive in CLI mode.
    // In gateway mode the server listener holds the event loop anyway.
    this.handle.unref();
  }

  /**
   * Cancel any pending timer and schedule to fire at an absolute epoch-ms.
   * If `epochMs` is in the past, fires on next tick (delay = 0).
   */
  scheduleAt(epochMs: number, callback: () => void): void {
    this.cancelInternal();
    this.callback = callback;
    this.scheduledAt = epochMs;
    const delay = Math.max(0, epochMs - Date.now());
    this.handle = setTimeout(() => this.fire(), delay);
    this.handle.unref();
  }

  /**
   * Only reschedule if `epochMs` is *earlier* than the current scheduled time.
   * This implements the "downward-only" timer pattern (L2 scheduling).
   * If no timer is pending, behaves like `scheduleAt()`.
   *
   * @returns true if the timer was actually advanced (or newly set).
   */
  tryAdvanceTo(epochMs: number, callback: () => void): boolean {
    if (this.handle === null) {
      // No pending timer → set it
      this.scheduleAt(epochMs, callback);
      return true;
    }

    if (epochMs < this.scheduledAt) {
      // New time is earlier → reschedule
      this.scheduleAt(epochMs, callback);
      return true;
    }

    // Current timer is already earlier or equal → keep it
    return false;
  }

  /**
   * Cancel the pending timer without triggering the callback.
   */
  cancel(): void {
    this.cancelInternal();
  }

  /**
   * Immediately trigger the callback (if pending) and clear the timer.
   * Used for graceful shutdown to flush pending work.
   *
   * Note: Unlike `fire()`, this method intentionally does NOT check `isDestroyed`.
   * This is by design — during shutdown, `destroy()` sets `destroyed = true` first,
   * then calls `flush()` to drain pending work. The `isDestroyed` guard only applies
   * to natural timer expiration via `fire()`, not to explicit shutdown flushes.
   */
  flush(): void {
    if (this.handle === null) return;
    const cb = this.callback;
    this.cancelInternal();
    if (cb) cb();
  }

  // ── Accessors ────────────────────────────────────────

  /** Whether a timer is currently pending. */
  get pending(): boolean {
    return this.handle !== null;
  }

  /** The epoch-ms when the current timer is scheduled to fire (0 if none). */
  get scheduledTime(): number {
    return this.handle !== null ? this.scheduledAt : 0;
  }

  // ── Internals ────────────────────────────────────────

  private fire(): void {
    const cb = this.callback;
    this.handle = null;
    this.callback = null;
    this.scheduledAt = 0;

    if (this.isDestroyed?.()) return;
    if (cb) cb();
  }

  private cancelInternal(): void {
    if (this.handle !== null) {
      clearTimeout(this.handle);
      this.handle = null;
    }
    this.callback = null;
    this.scheduledAt = 0;
  }
}
