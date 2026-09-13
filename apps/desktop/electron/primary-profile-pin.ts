/**
 * Which profile the PRIMARY (window) backend answers for.
 *
 * Two signals exist and they legitimately disagree mid-session:
 *
 *   - the stored preference (active-profile.json) — the profile Desktop should
 *     boot into on the NEXT launch. The rail's live workspace switch rewrites
 *     it via `hermes:profile:remember` without re-homing the primary.
 *   - the profile the live primary child was actually spawned with.
 *
 * Routing (`resolveProfileBackendRoute`, `ensureBackend`, the pool) must use
 * the second while a primary is up. Reading the preference instead made a
 * request for the booted profile (e.g. "default") stop matching
 * `primaryProfile`, fall through to the pool, and spawn a second backend for
 * the same HERMES_HOME. That duplicate held a pool slot and starved every other
 * profile into "timed out while waiting for a free slot".
 *
 * Pure: main.ts owns the file read and the start/teardown call sites.
 */
export class PrimaryProfilePin {
  #booted: null | string = null

  /** Called by startHermes() with the profile the primary is launching as. */
  pin(profile: null | string | undefined): string {
    const value = String(profile ?? '').trim() || 'default'
    this.#booted = value

    return value
  }

  /** Called when the primary is torn down; the next start re-reads the preference. */
  clear(): void {
    this.#booted = null
  }

  get booted(): null | string {
    return this.#booted
  }

  /**
   * The key routing should treat as "primary": the live primary's profile when
   * one is pinned, else the stored preference, else 'default'.
   */
  resolve(readPreference: () => null | string | undefined): string {
    if (this.#booted) {
      return this.#booted
    }

    return String(readPreference() ?? '').trim() || 'default'
  }
}
