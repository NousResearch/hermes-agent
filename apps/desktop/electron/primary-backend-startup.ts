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
  /**
   * After update clearance and before any local attach/spawn. When launchMode
   * is primary and the registry primary is non-local, this returns that
   * remote so a stale pre-update resolve cannot boot an empty local backend.
   */
  selectRegistryPrimary?: () => Promise<Remote | null>
  waitForDecision: (backend: Backend) => Promise<FirstRunSetupDecision>
  waitForLocalStart: () => Promise<unknown>
}

export type PrimaryBackendStartupResult<RuntimeBackend, Connection, Attached = never> =
  | { kind: 'attached'; attached: Attached }
  | { kind: 'local'; backend: RuntimeBackend }
  | { kind: 'remote'; connection: Connection }

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
  selectRegistryPrimary,
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

  // Update clearance can land after the first resolve saw a stale local route
  // (legacy connection.json, a mid-update registry read). Re-select the
  // registry primary before attaching to or spawning a local backend.
  if (selectRegistryPrimary) {
    const registryPrimary = await step(selectRegistryPrimary)

    if (registryPrimary) {
      return { kind: 'remote', connection: await step(() => connectRemote(registryPrimary)) }
    }
  }

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
