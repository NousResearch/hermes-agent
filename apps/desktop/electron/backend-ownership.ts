export interface BackendIdentity {
  nonce: string
  pid: number
  profile: string
  startMarker: string
}

export interface BackendOwnershipEntry extends BackendIdentity {
  command?: string
}

export interface BackendOwnershipStore {
  read: () => string | null
  write: (contents: string) => void
}

export interface BackendOwnershipDeps {
  matchesIdentity: (identity: BackendIdentity) => Promise<boolean | undefined>
  stop: (identity: BackendIdentity) => Promise<void> | void
  store: BackendOwnershipStore
}

export interface BackendClaim extends BackendIdentity {
  command?: string
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}

function isCompleteIdentity(value: unknown): value is BackendIdentity {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Partial<BackendIdentity>

  return (
    Number.isInteger(candidate.pid) &&
    Number(candidate.pid) > 0 &&
    isNonEmptyString(candidate.startMarker) &&
    isNonEmptyString(candidate.nonce) &&
    isNonEmptyString(candidate.profile)
  )
}

function identitiesMatch(left: BackendIdentity, right: BackendIdentity): boolean {
  return (
    left.pid === right.pid &&
    left.startMarker === right.startMarker &&
    left.nonce === right.nonce &&
    left.profile === right.profile
  )
}

export function parseBackendOwnership(contents: unknown): BackendOwnershipEntry[] {
  let parsed: unknown

  try {
    parsed = JSON.parse(String(contents ?? ''))
  } catch {
    return []
  }

  const values = Array.isArray(parsed)
    ? parsed
    : parsed && typeof parsed === 'object' && Array.isArray((parsed as { backends?: unknown }).backends)
      ? (parsed as { backends: unknown[] }).backends
      : []

  const entries: BackendOwnershipEntry[] = []

  for (const value of values) {
    if (!isCompleteIdentity(value)) {
      continue
    }

    const candidate = value as BackendOwnershipEntry

    const entry: BackendOwnershipEntry = {
      nonce: candidate.nonce,
      pid: candidate.pid,
      profile: candidate.profile,
      startMarker: candidate.startMarker
    }

    if (typeof candidate.command === 'string') {
      entry.command = candidate.command
    }

    if (!entries.some(existing => identitiesMatch(existing, entry))) {
      entries.push(entry)
    }
  }

  return entries
}

export function serializeBackendOwnership(entries: BackendOwnershipEntry[]): string {
  return `${JSON.stringify({ backends: entries }, null, 2)}\n`
}

/**
 * Persistent ownership for local backend roots.
 *
 * Claiming is asynchronous so a failed persistence transaction can await child
 * cleanup before reporting failure to the caller.
 */
export function createBackendOwnership(deps: BackendOwnershipDeps) {
  const read = () => parseBackendOwnership(deps.store.read())
  const write = (entries: BackendOwnershipEntry[]) => deps.store.write(serializeBackendOwnership(entries))

  return {
    async claim(claim: BackendClaim): Promise<BackendOwnershipEntry> {
      if (!isCompleteIdentity(claim)) {
        throw new Error('Cannot own a backend without a complete process identity.')
      }

      const entry: BackendOwnershipEntry = {
        nonce: claim.nonce,
        pid: claim.pid,
        profile: claim.profile,
        startMarker: claim.startMarker
      }

      if (typeof claim.command === 'string') {
        entry.command = claim.command
      }

      try {
        const entries = read().filter(candidate => candidate.pid !== entry.pid)
        write([...entries, entry])
      } catch (error) {
        try {
          await deps.stop(entry)
        } catch {
          // Persistence remains the claim failure even if cleanup also fails.
        }

        throw error
      }

      return entry
    },

    release(identity: BackendIdentity): void {
      if (!isCompleteIdentity(identity)) {
        throw new Error('Cannot release a backend without a complete process identity.')
      }

      const entries = read()
      const next = entries.filter(entry => !identitiesMatch(entry, identity))

      if (next.length !== entries.length) {
        write(next)
      }
    },

    async reapOrphans(): Promise<number[]> {
      const entries = read()
      const survivors: BackendOwnershipEntry[] = []
      const reaped: number[] = []

      for (const entry of entries) {
        let matches: boolean | undefined

        try {
          matches = await deps.matchesIdentity(entry)
        } catch {
          survivors.push(entry)

          continue
        }

        if (matches === false) {
          continue
        }

        if (matches !== true) {
          survivors.push(entry)

          continue
        }

        try {
          await deps.stop(entry)
          reaped.push(entry.pid)
        } catch {
          // Preserve failed ownership so a later startup can retry it.
          survivors.push(entry)
        }
      }

      write(survivors)

      return reaped
    },

    clear(): void {
      write([])
    }
  }
}

export function backendCommandMatches(command: unknown): boolean {
  return /(?:^|[\s/\\"])(?:hermes(?:\.exe)?|hermes_cli\.main|hermes_cli[/\\]main\.py)"?(?:\s+(?:--profile|-p)\s+\S+)?\s+(?:serve|dashboard)(?:\s|$)/i.test(
    String(command ?? '')
  )
}

/** Coordinates all quit paths so asynchronous backend teardown runs once. */
export function createBackendShutdownCoordinator(teardown: () => Promise<void> | void) {
  let completion: Promise<void> | undefined

  return {
    run(): Promise<void> {
      if (!completion) {
        completion = Promise.resolve().then(teardown)
      }

      return completion
    },
    hasStarted(): boolean {
      return completion !== undefined
    }
  }
}
