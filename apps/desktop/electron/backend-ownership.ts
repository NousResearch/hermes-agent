export interface BackendIdentity {
  nonce: string
  pid: number
  profile: string
  startMarker: string
}

export interface BackendOwnershipEntry extends BackendIdentity {
  command?: string
  /** PID of the Electron parent that spawned this backend, when known. */
  parentPid?: number
  /** Start marker of that parent, so a reused PID is not mistaken for it. */
  parentStartMarker?: string
}

export interface BackendOwnershipStore {
  read: () => string | null
  write: (contents: string) => void
  /** Move an unreadable ownership file aside (e.g. rename to `.corrupt`) so
   *  its contents survive for inspection instead of being rewritten away.
   *  Optional: stores that can't quarantine simply skip the sweep. */
  quarantine?: () => void
}

export interface BackendOwnershipDeps {
  matchesIdentity: (identity: BackendIdentity) => Promise<boolean | undefined>
  /** True when the recorded parent is still running; undefined when unknown. */
  matchesParent: (entry: BackendOwnershipEntry) => Promise<boolean | undefined>
  stop: (identity: BackendIdentity) => Promise<void> | void
  store: BackendOwnershipStore
  /**
   * Overall time budget for one reap sweep. The ownership file legitimately
   * accumulates one record per profile per launch, and each record can cost
   * up to two identity probes (parent + backend) plus a stop — on Windows
   * those shell out to PowerShell, whose 5.1 cold starts are slow (#87169).
   * Without a bound, a large roster could stall boot for minutes while the
   * renderer's 45s backend-boot budget expires and the user stares at the
   * connecting screen. When the budget is exhausted the sweep preserves the
   * unprocessed records for the next launch and returns what it reaped.
   */
  reapDeadlineMs?: number
}

/** Default budget for one reap sweep (see `reapDeadlineMs`). */
export const REAP_ORPHANS_DEADLINE_MS = 5_000

export interface BackendClaim extends BackendIdentity {
  command?: string
  parentPid?: number
  parentStartMarker?: string
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
  return parseBackendOwnershipDetailed(contents).entries
}

/** Parse result that distinguishes "empty/valid" from "unreadable". A corrupt
 *  ownership file must NOT read as an empty roster: `reapOrphans` rewrites the
 *  file with its survivors, so treating garbage as `[]` permanently erased the
 *  records of still-running backends — the exact shape of the #89298 report
 *  (ownership file gone, 28 leaked serve processes nothing will ever reap). */
export function parseBackendOwnershipDetailed(contents: unknown): {
  corrupt: boolean
  entries: BackendOwnershipEntry[]
} {
  const text = String(contents ?? '')

  if (!text.trim()) {
    return { corrupt: false, entries: [] }
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    return { corrupt: true, entries: [] }
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

    if (Number.isInteger(candidate.parentPid) && Number(candidate.parentPid) > 0) {
      entry.parentPid = candidate.parentPid
    }

    if (isNonEmptyString(candidate.parentStartMarker)) {
      entry.parentStartMarker = candidate.parentStartMarker
    }

    if (!entries.some(existing => identitiesMatch(existing, entry))) {
      entries.push(entry)
    }
  }

  return { corrupt: false, entries }
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
  const readDetailed = () => parseBackendOwnershipDetailed(deps.store.read())
  const read = () => readDetailed().entries
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

      if (Number.isInteger(claim.parentPid) && Number(claim.parentPid) > 0) {
        entry.parentPid = claim.parentPid
      }

      if (isNonEmptyString(claim.parentStartMarker)) {
        entry.parentStartMarker = claim.parentStartMarker
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
      const { corrupt, entries } = readDetailed()

      // An unreadable ownership file yields zero parsed entries — rewriting
      // survivors ([]) here would DESTROY the only record of any backends the
      // corrupt file described, guaranteeing they leak forever (#89298).
      // Preserve the evidence for inspection and skip the sweep.
      if (corrupt) {
        try {
          deps.store.quarantine?.()
        } catch {
          // Quarantine is best-effort; the important part is not rewriting.
        }

        return []
      }

      const survivors: BackendOwnershipEntry[] = []
      const reaped: number[] = []
      const deadline = Date.now() + (deps.reapDeadlineMs ?? REAP_ORPHANS_DEADLINE_MS)

      for (let i = 0; i < entries.length; i += 1) {
        // Budget exhausted: preserve the unprocessed records so a later launch
        // can retry them. A slow identity probe must never stall boot — the
        // renderer's backend-boot budget is 45s and the spawn itself needs
        // most of it.
        if (Date.now() >= deadline) {
          survivors.push(...entries.slice(i))

          break
        }

        const entry = entries[i]

        // A backend whose Electron parent is still running is NOT an orphan:
        // reaping it would kill a live instance's session. This is what stops
        // a second launch from SIGTERMing the running instance's backend even
        // if it reaches reapOrphans (see main.ts startHermes + #87295).
        let parentAlive: boolean | undefined

        try {
          parentAlive = await deps.matchesParent(entry)
        } catch {
          survivors.push(entry)

          continue
        }

        if (parentAlive === true) {
          survivors.push(entry)

          continue
        }

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
