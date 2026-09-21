import { runBackendStartStep } from './backend-start-cancellation'
import type { FirstRunSetupDecision } from './first-run-setup-gate'

export interface PrimaryBackendStartupOptions<Backend, RuntimeBackend, Remote, Connection, Attached> {
  assertCurrentAttempt: () => void
  signal?: AbortSignal
  /**
   * Multiplex-only: attach to the backend already running on this HOST.
   * Resolves null when the host has none, which is the only case that spawns.
   */
  attachHostBackend?: () => Promise<Attached | null>
  connectRemote: (remote: Remote) => Promise<Connection>
  ensureLocalRuntime: (backend: Backend) => Promise<RuntimeBackend>
  prepareLocalBackend: () => Backend | Promise<Backend>
  resolveRemote: () => Promise<Remote | null>
  waitForDecision: (backend: Backend) => Promise<FirstRunSetupDecision>
  waitForLocalStart: () => Promise<unknown>
}

export type PrimaryBackendStartupResult<RuntimeBackend, Connection, Attached = never> =
  | { kind: 'attached'; attached: Attached }
  | { kind: 'local'; backend: RuntimeBackend }
  | { kind: 'remote'; connection: Connection }

interface ResolvedPrimaryRemote {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  connectionId?: string
  headers?: Record<string, string>
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
    // fetchJsonForBackend reads descriptor.headers for every REST call; the
    // WebSocket header store is keyed by exact URL and cannot stand in for it.
    headers: remote.headers,
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

// Owns the production startHermes path up to the local process spawn. Keeping
// the full ordering here makes the first-run remote boundary executable in a
// test: an already-saved remote wins immediately; otherwise update exclusion
// and local backend resolution happen before the setup gate, and a remote Apply
// re-resolves persisted config without ever entering ensureRuntime/bootstrap.
export async function runPrimaryBackendStartup<Backend, RuntimeBackend, Remote, Connection, Attached = never>({
  assertCurrentAttempt,
  attachHostBackend,
  connectRemote,
  ensureLocalRuntime,
  prepareLocalBackend,
  resolveRemote,
  waitForDecision,
  waitForLocalStart,
  signal
}: PrimaryBackendStartupOptions<Backend, RuntimeBackend, Remote, Connection, Attached>): Promise<
  PrimaryBackendStartupResult<RuntimeBackend, Connection, Attached>
> {
  const step = async <T>(run: () => T | Promise<T>) => {
    const result = await runBackendStartStep(signal, run)
    assertCurrentAttempt()

    return result
  }

  const savedRemote = await step(resolveRemote)

  if (savedRemote) {
    return { kind: 'remote', connection: await step(() => connectRemote(savedRemote)) }
  }

  await step(waitForLocalStart)

  // Multiplex-only: one backend per HOST. Attach before resolving a runtime or
  // entering the first-run gate — a machine with a live backend is, by
  // definition, already set up, and the runtime resolve is only needed to spawn.
  const attached = attachHostBackend ? await step(attachHostBackend) : null

  if (attached) {
    return { kind: 'attached', attached }
  }

  const backend = await step(prepareLocalBackend)
  const decision = await step(() => waitForDecision(backend))

  if (decision === 'remote-applied') {
    const appliedRemote = await step(resolveRemote)

    if (!appliedRemote) {
      throw new Error('First-run remote setup completed without a saved remote backend.')
    }

    return { kind: 'remote', connection: await step(() => connectRemote(appliedRemote)) }
  }

  if (decision === 'reset') {
    throw new FirstRunSetupResetError()
  }

  return { kind: 'local', backend: await step(() => ensureLocalRuntime(backend)) }
}
