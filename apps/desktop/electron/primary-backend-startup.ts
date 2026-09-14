import { runBackendStartStep } from './backend-start-cancellation'
import type { FirstRunSetupDecision } from './first-run-setup-gate'

export interface PrimaryBackendStartupOptions<Backend, RuntimeBackend, Remote, Connection> {
  assertCurrentAttempt: () => void
  signal?: AbortSignal
  connectRemote: (remote: Remote) => Promise<Connection>
  ensureLocalRuntime: (backend: Backend) => Promise<RuntimeBackend>
  prepareLocalBackend: () => Backend | Promise<Backend>
  resolveRemote: () => Promise<Remote | null>
  waitForDecision: (backend: Backend) => Promise<FirstRunSetupDecision>
  waitForLocalStart: () => Promise<unknown>
}

export type PrimaryBackendStartupResult<RuntimeBackend, Connection> =
  { kind: 'local'; backend: RuntimeBackend } | { kind: 'remote'; connection: Connection }

interface ResolvedPrimaryRemote {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  connectionId?: string
  remoteHermesVersion?: string
  remoteHost?: string
  remoteKind?: 'cloud' | 'ssh' | 'url'
  source?: string
  ssh?: {
    effectiveConfigFingerprint?: string
    host?: string
    keyPath?: string
    port?: number
    remoteHermesPath?: string
    remoteProfile?: string
    user?: string
  }
  token: unknown
  wsUrl: string
}

/**
 * Build the renderer-facing primary remote descriptor without dropping route
 * identity. Tests cross this same seam, so adding a field to the resolved
 * remote cannot silently disappear during primary startup.
 */
export function createPrimaryRemoteConnection<State extends object>(
  remote: ResolvedPrimaryRemote,
  logs: string[],
  windowState: State
) {
  return {
    baseUrl: remote.baseUrl,
    mode: 'remote' as const,
    source: remote.source,
    authMode: remote.authMode || 'token',
    remoteHost: remote.remoteHost,
    remoteKind: remote.remoteKind,
    remoteHermesVersion: remote.remoteHermesVersion,
    ...(remote.connectionId ? { connectionId: remote.connectionId } : {}),
    ...(remote.ssh ? { ssh: remote.ssh } : {}),
    token: remote.token,
    wsUrl: remote.wsUrl,
    logs,
    ...windowState
  }
}

export class FirstRunSetupResetError extends Error {
  readonly firstRunSetupReset = true

  constructor() {
    super('First-run setup was reset before a choice completed.')
    this.name = 'FirstRunSetupResetError'
  }
}

// Owns the production startHermes path up to canonical local gateway ensure. Keeping
// the full ordering here makes the first-run remote boundary executable in a
// test: an already-saved remote wins immediately; otherwise update exclusion
// and local backend resolution happen before the setup gate, and a remote Apply
// re-resolves persisted config without ever entering ensureRuntime/bootstrap.
export async function runPrimaryBackendStartup<Backend, RuntimeBackend, Remote, Connection>({
  assertCurrentAttempt,
  connectRemote,
  ensureLocalRuntime,
  prepareLocalBackend,
  resolveRemote,
  waitForDecision,
  waitForLocalStart,
  signal
}: PrimaryBackendStartupOptions<Backend, RuntimeBackend, Remote, Connection>): Promise<
  PrimaryBackendStartupResult<RuntimeBackend, Connection>
> {
  // Fence in this continuation, not another async wrapper, so no await separates
  // the ownership check from the next startup action.
  const step = <T>(run: () => T | Promise<T>) => runBackendStartStep(signal, run)
  assertCurrentAttempt()
  const savedRemote = await step(resolveRemote)
  assertCurrentAttempt()

  if (savedRemote) {
    const connection = await step(() => connectRemote(savedRemote))
    assertCurrentAttempt()

    return { kind: 'remote', connection }
  }

  await step(waitForLocalStart)
  assertCurrentAttempt()

  const backend = await step(prepareLocalBackend)
  assertCurrentAttempt()
  const decision = await step(() => waitForDecision(backend))
  assertCurrentAttempt()

  if (decision === 'remote-applied') {
    const appliedRemote = await step(resolveRemote)
    assertCurrentAttempt()

    if (!appliedRemote) {
      throw new Error('First-run remote setup completed without a saved remote backend.')
    }

    const connection = await step(() => connectRemote(appliedRemote))
    assertCurrentAttempt()

    return { kind: 'remote', connection }
  }

  if (decision === 'reset') {
    throw new FirstRunSetupResetError()
  }

  const runtimeBackend = await step(() => ensureLocalRuntime(backend))
  assertCurrentAttempt()

  return { kind: 'local', backend: runtimeBackend }
}
