// Profile-delete routing logic for the `hermes:api` IPC handler.
//
// When the renderer issues DELETE /api/profiles/<name>, the handler must
// tear down every local backend for that profile and route the DELETE itself
// away from the just-deleted profile. Concurrent and delayed starts must also
// be rejected: spawning a fresh backend would call ensure_hermes_home() and
// recreate the profile directory the delete just removed, leaving a zombie
// process behind (issue #52279).
//
// These helpers are pure so they can be unit-tested without Electron.

/** Parse a profile name from an `/api/profiles/<name>` request path. */
export function profileNameFromPath(path: unknown): string | null {
  const match = String(path || '').match(/^\/api\/profiles\/([^/?#]+)(?:[?#].*)?$/)

  if (!match) {
    return null
  }

  let raw = ''

  try {
    raw = decodeURIComponent(match[1])
  } catch {
    return null
  }

  const name = raw.trim()

  if (!name) {
    return null
  }

  if (name.toLowerCase() === 'default') {
    return 'default'
  }

  return name.toLowerCase()
}

/** Parse a `hermes:api` request into the profile name a DELETE targets. */
export function profileNameFromDeleteRequest(request) {
  if (!request || String(request.method || 'GET').toUpperCase() !== 'DELETE') {
    return null
  }

  return profileNameFromPath(request.path)
}

export type ProfileDeleteAction = 'noop' | 'teardown-primary' | 'teardown-pool'

export interface ProfileDeleteDecision {
  action: ProfileDeleteAction
  profile: string | null
}

export interface ProfileDeleteDecisionDeps {
  isDefaultProfile: (profile: string) => boolean
  isValidProfileName: (profile: string) => boolean
  primaryProfileKey: () => string
}

/**
 * Process-local barrier for profile deletion. Electron IPC handlers run
 * concurrently, so tearing down a pooled backend is not enough by itself: a
 * renderer reconnect can enter ensureBackend() while the DELETE request is
 * still removing the profile and recreate its HERMES_HOME.
 *
 * Counts instead of a Set keep overlapping requests for the same profile
 * blocked until the last request releases its lease.
 */
export class ProfileDeletionGate {
  readonly #active = new Map<string, number>()

  acquire(profile: unknown): () => void {
    const key = String(profile ?? '')
      .trim()
      .toLowerCase()

    if (!key) {
      return () => undefined
    }

    this.#active.set(key, (this.#active.get(key) ?? 0) + 1)
    let released = false

    return () => {
      if (released) {
        return
      }

      released = true
      const remaining = (this.#active.get(key) ?? 1) - 1

      if (remaining > 0) {
        this.#active.set(key, remaining)
      } else {
        this.#active.delete(key)
      }
    }
  }

  blocks(profile: unknown): boolean {
    const key = String(profile ?? '')
      .trim()
      .toLowerCase()

    return Boolean(key && this.#active.has(key))
  }

  assertCanStart(profile: unknown): void {
    const key = String(profile ?? '').trim()

    if (this.blocks(key)) {
      throw new Error(`Profile "${key}" is being deleted.`)
    }
  }
}

/**
 * Validate the final boundary before spawning a local profile backend. The
 * deletion gate closes the in-flight race; the directory check rejects a
 * delayed renderer retry after the DELETE request has already completed.
 */
export function assertLocalProfileCanStart(
  profile: unknown,
  gate: ProfileDeletionGate,
  profileDirectoryExists: (profile: string) => boolean
): void {
  const key = String(profile ?? '')
    .trim()
    .toLowerCase()

  gate.assertCanStart(key)

  if (key && key !== 'default' && !profileDirectoryExists(key)) {
    throw new Error(`Profile "${key}" no longer exists.`)
  }
}

/**
 * A local profile can occupy the legacy bare pool slot or the explicit-local
 * registry slot when the v1 route points elsewhere. Both processes own the
 * same on-disk profile and must be stopped before deleting it.
 */
export function localProfilePoolKeys(profile: unknown): string[] {
  const key = String(profile ?? '')
    .trim()
    .toLowerCase()

  return key ? [key, `conn:local::${key}`] : []
}

/**
 * Pure decision logic for prepareProfileDeleteRequest: given the parsed
 * profile name (or null), decide which side-effecting branch the caller
 * should take and what profile name it should ultimately report as
 * torn-down. No I/O, no async -- the caller performs the actual teardown
 * based on `action`.
 */
export function decideProfileDeleteAction(
  profile: string | null,
  deps: ProfileDeleteDecisionDeps
): ProfileDeleteDecision {
  if (!profile || deps.isDefaultProfile(profile) || !deps.isValidProfileName(profile)) {
    return { action: 'noop', profile: null }
  }

  if (profile === deps.primaryProfileKey()) {
    return { action: 'teardown-primary', profile }
  }

  return { action: 'teardown-pool', profile }
}

/**
 * Route the next `hermes:api` request away from the primary/window backend
 * whenever a profile was just torn down -- otherwise ensureBackend would
 * spawn a fresh pool backend for the deleted profile, whose
 * ensure_hermes_home() recreates the directory the delete just removed.
 */
export function resolveRouteProfile(
  tornDownProfile: string | null,
  profile: string | null | undefined
): string | null | undefined {
  return tornDownProfile ? null : profile
}
