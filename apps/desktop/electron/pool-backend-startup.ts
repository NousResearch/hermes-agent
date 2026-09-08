interface PoolBackendRuntimeOptions<Backend, RuntimeBackend> {
  backend: Backend
  ensureRuntime: (backend: Backend) => Promise<RuntimeBackend>
  profile: string
}

export async function ensurePoolBackendRuntime<Backend extends { kind?: string }, RuntimeBackend>({
  backend,
  ensureRuntime,
  profile
}: PoolBackendRuntimeOptions<Backend, RuntimeBackend>): Promise<RuntimeBackend> {
  // Background resolution can fail transiently. Only primary startup owns
  // the explicit local setup decision; a pool start must never install.
  if (backend.kind === 'bootstrap-needed') {
    throw new Error(
      `No usable local Hermes runtime was found for profile "${profile}". ` +
        'Use the desktop installer to explicitly install or repair Hermes, then retry this profile.'
    )
  }

  return ensureRuntime(backend)
}
