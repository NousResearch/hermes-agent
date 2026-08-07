/**
 * QuitBarrier — coordinates all async work that must complete before Electron
 * actually exits (SSH teardown, backend process drain).
 *
 * The race this prevents: multiple exit paths each call `event.preventDefault()`
 * and each schedule their own `app.quit()` re-entry in the same `before-quit`
 * invocation. Whichever path's promise resolves first re-enters `before-quit`,
 * sees the other path's latch already set, and lets Electron exit while the
 * other wait is still pending — orphaning whatever that path was draining.
 *
 * QuitBarrier fixes it: every path registers its pending promise, then `arm()`
 * snapshots ALL of them, holds the quit once, and re-enters `app.quit()` only
 * after every registered promise has settled (or reached its own timeout
 * bound). A re-entrant quit cannot bypass an outstanding wait because the
 * barrier already snapped the full set and the second `app.quit()` runs with
 * the pending list empty.
 */
export class QuitBarrier {
  private pending: Promise<unknown>[] = []
  private armed = false

  /** Register async work that must finish before quit completes. */
  add(work: Promise<unknown>): void {
    if (this.armed) {
      // A re-entrant before-quit must not re-schedule work that was already
      // snapshotted by the armed barrier.
      return
    }

    this.pending.push(work)
  }

  get isEmpty(): boolean {
    return this.pending.length === 0
  }

  /**
   * Arm the barrier exactly once. Snaps the current pending set, resets the
   * accumulator, and returns a promise that resolves when every snapshotted
   * promise has settled. `null` if there is nothing to wait on.
   *
   * Repeated calls return `null` after the first arm — a re-entrant
   * `before-quit` can't double-arm.
   */
  arm(): Promise<void> | null {
    if (this.armed || this.pending.length === 0) {
      return null
    }

    this.armed = true
    const batch = this.pending
    this.pending = []

    return Promise.allSettled(batch).then(() => undefined)
  }
}
