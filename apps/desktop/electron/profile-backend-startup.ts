export type ProfileBackendStartReason = 'primary_boot' | 'profile_activate' | 'background_session' | 'unknown'

const START_REASONS = new Set<ProfileBackendStartReason>([
  'primary_boot',
  'profile_activate',
  'background_session',
  'unknown'
])

export function normalizeProfileBackendStartReason(value: unknown): ProfileBackendStartReason {
  return typeof value === 'string' && START_REASONS.has(value as ProfileBackendStartReason)
    ? (value as ProfileBackendStartReason)
    : 'unknown'
}

/**
 * Runs local profile backend startup work one at a time. This intentionally
 * does not own remote connections: they do not spawn a local child or compete
 * for the machine resources this queue protects.
 */
export function createProfileBackendStartupQueue() {
  let tail: Promise<void> = Promise.resolve()

  return {
    run<T>(task: () => Promise<T>): Promise<T> {
      const previous = tail
      let release: () => void

      tail = new Promise<void>(resolve => {
        release = resolve
      })

      return previous
        .catch(() => undefined)
        .then(task)
        .finally(release!)
    }
  }
}

export interface ReusablePoolConnection<T> {
  connectionPromise: Promise<T>
  lastActiveAt: number
}

/**
 * Keep an existing pool connection on its current path. In particular, do not
 * make it wait behind a different profile's cold start.
 */
export function reusePoolConnection<T>(
  entry: ReusablePoolConnection<T> | undefined,
  now = Date.now()
): Promise<T> | null {
  if (!entry) {
    return null
  }

  entry.lastActiveAt = now

  return entry.connectionPromise
}
