export function createSshReuseProbe(
  fetchJson: (url: string, token: string) => Promise<unknown>,
  classify: (proof: unknown, nonce: string) => string
) {
  return async (baseUrl: string, token: string, spawnNonce: string) => {
    try {
      return classify(await fetchJson(`${baseUrl}/api/ssh/ownership`, token), spawnNonce)
    } catch (error: any) {
      if (/^(401|403|404):/.test(String(error?.message || ''))) {
        return 'authenticated-stale'
      }

      throw error
    }
  }
}

interface SshTransportState {
  ssh: {
    cancelForward: (localPort: number, remotePort: number) => Promise<unknown>
    close: () => Promise<unknown>
  }
  localPort?: number
  remotePort?: number
}

/** Never calls disconnect: the current remote lock may already name a new owner. */
export async function releaseSshTransport(
  state: SshTransportState,
  { preserveMaster = false }: { preserveMaster?: boolean } = {}
): Promise<void> {
  try {
    if (state.localPort && state.remotePort) {
      await state.ssh.cancelForward(state.localPort, state.remotePort)
    }
  } catch {
    // Closing the captured master also drops its forwards.
  }

  if (preserveMaster) {
    return
  }

  try {
    await state.ssh.close()
  } catch {
    // A failed transport is already unusable; do not turn this into a remote kill.
  }
}

interface PooledSshDispatchRecoveryOptions<TState> {
  state: TState | undefined
  currentState: () => TState | undefined
  isCurrentEntry: () => boolean
  proveStopped: (state: TState) => Promise<boolean>
  markRecovery: () => void
  retireEntry: () => Promise<void>
  releaseTransport: (state: TState) => Promise<void>
  drain: (cleanup: () => Promise<void>) => Promise<void>
}

/** Capture both identities before any await; an old probe cannot retire a new scope. */
export function createPooledSshDispatchRecovery<TState>({
  state,
  currentState,
  isCurrentEntry,
  proveStopped,
  markRecovery,
  retireEntry,
  releaseTransport,
  drain
}: PooledSshDispatchRecoveryOptions<TState>) {
  let permitted = false
  const isCurrent = () => Boolean(state) && currentState() === state && isCurrentEntry()

  return {
    canRetire: async () => {
      if (!isCurrent()) {
        return false
      }

      markRecovery()
      permitted = isCurrent() && (await proveStopped(state))

      return permitted && isCurrent()
    },
    retire: async () => {
      if (!permitted || !isCurrent()) {
        return
      }

      await drain(async () => {
        if (!isCurrent()) {
          return
        }

        await retireEntry()

        if (currentState() === state) {
          await releaseTransport(state)
        }
      })
    }
  }
}

interface PooledBackendRecoveryOptions<TEntry, TState> {
  pool: ReadonlyMap<string, TEntry>
  sshStates: ReadonlyMap<string, TState>
  scopeFor: (key: string) => string
  proveStopped: (state: TState) => Promise<boolean>
  markRecovery: (scope: string) => void
  retireEntry: (key: string) => Promise<void>
  releaseTransport: (scope: string, state: TState) => Promise<void>
  drain: (scope: string, cleanup: () => Promise<void>) => Promise<void>
}

/** Shared by dispatch and post-resume recovery; explicit disconnect never uses this policy. */
export function createPooledBackendRecovery<TEntry, TState>({
  pool,
  sshStates,
  scopeFor,
  proveStopped,
  markRecovery,
  retireEntry,
  releaseTransport,
  drain
}: PooledBackendRecoveryOptions<TEntry, TState>) {
  return (key: string, entry: TEntry, isSsh: boolean) => {
    const scope = scopeFor(key)

    if (!isSsh) {
      return {
        canRetire: async () => true,
        retire: async () => {
          if (pool.get(key) === entry) {
            await retireEntry(key)
          }
        }
      }
    }

    return createPooledSshDispatchRecovery({
      state: sshStates.get(scope),
      currentState: () => sshStates.get(scope),
      isCurrentEntry: () => pool.get(key) === entry,
      proveStopped,
      markRecovery: () => markRecovery(scope),
      retireEntry: () => retireEntry(key),
      releaseTransport: state => releaseTransport(scope, state),
      drain: cleanup => drain(scope, cleanup)
    })
  }
}

/** Read-only process observations for conservative pooled SSH dispatch recovery. */
export function stoppedOwnerProbeProgram(pid?: number): string {
  return `import os,sys
pid=${pid === undefined ? 'int(sys.argv[1])' : pid}
try:
 os.kill(pid,0)
except ProcessLookupError:
 print("DEAD")
except OSError:
 print("UNKNOWN")
else:
 print("ALIVE")`
}

/** Run only while holding the existing connect reservation and install mutex.
 * The shell may remove the dead record only after this exact snapshot still
 * matches. Missing, replaced, malformed and permission-denied owners all refuse.
 */
export function stoppedOwnerReservationProgram(expected: Record<string, unknown> | null): string {
  return `import json,os,sys
expected=json.loads(${JSON.stringify(JSON.stringify(expected))})
verdict="REFUSED"
try:
 def read():
  with open(sys.argv[1],encoding="utf-8") as record:return json.load(record)
 if expected is None:
  try:os.lstat(sys.argv[1])
  except FileNotFoundError:verdict="ABSENT"
 elif expected.get("creationTime") and read()==expected:
  pid=expected["pid"]
  try:
   os.kill(pid,0)
  except ProcessLookupError:
   if read()==expected:verdict="DEAD"
except (OSError,ValueError,KeyError,TypeError):pass
print(verdict)`
}
