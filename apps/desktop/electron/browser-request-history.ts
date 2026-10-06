export const BROWSER_REQUEST_TIMEOUT_MS = { act: 20_000, read: 8_000 } as const
// Spend 100ms of the existing budget on cross-process clock discrepancy, never
// extend admission or expiry. Larger discrepancies must still fail closed.
export const BROWSER_REQUEST_DEADLINE_HEADROOM_MS = 100
export const BROWSER_REQUEST_HISTORY_LIMIT = 1024

/** Shared by main and the responder. Keep only compact replay tombstones, not
 * captured owners/targets. At capacity refuse new work; never evict an ID that
 * could still execute. Expired packets are rejected before their IDs are pruned.
 * The deadline is part of the ID so a replay cannot renew its execution budget.
 * Lazy pruning also bounds idle windows without adding a timer per request. */
export class BrowserRequestHistory {
  private deadlines = new Map<string, number>()
  private latestNow = Number.NEGATIVE_INFINITY

  admit(id: string, kind: 'act' | 'read', deadline: unknown): boolean {
    const now = Date.now()
    // Once expired/pruned, an ID cannot become live again after clock rollback.
    // Keep the upper future bound tied to the raw clock, not this expiry floor.
    this.latestNow = Math.max(this.latestNow, now)

    if (typeof deadline !== 'number' || !Number.isFinite(deadline) ||
      deadline <= this.latestNow || deadline > now + BROWSER_REQUEST_TIMEOUT_MS[kind] ||
      !id.endsWith(`-${deadline}`)) {return false}

    for (const [previousId, expires] of this.deadlines) {
      if (expires <= this.latestNow) {this.deadlines.delete(previousId)}
    }

    if (this.deadlines.has(id) || this.deadlines.size >= BROWSER_REQUEST_HISTORY_LIMIT) {return false}
    this.deadlines.set(id, deadline)

    return true
  }
}
