export type PoolBackendLaunchDeps<TBackend, TRemote> = {
  createToken: () => string
  ensureRuntime: (backend: TBackend) => Promise<TBackend>
  getBackendArgsForRuntime: (backend: TBackend) => string[]
  resolveHermesBackend: (args: string[]) => TBackend
  resolveRemoteBackend: (profile: string) => Promise<TRemote | null>
  waitForHermes: (baseUrl: string, token: string | null) => Promise<void>
  waitForUpdateToFinish: () => Promise<unknown>
}

export type RemotePoolBackendLaunch<TRemote> = {
  kind: 'remote'
  remote: TRemote
}

export type LocalPoolBackendLaunch<TBackend> = {
  kind: 'local'
  backend: TBackend
  token: string
}

export type PoolBackendLaunch<TBackend, TRemote> =
  | RemotePoolBackendLaunch<TRemote>
  | LocalPoolBackendLaunch<TBackend>

export async function preparePoolBackendLaunch<TBackend extends { args: string[] }, TRemote extends { baseUrl: string; token: string | null }>(
  profile: string,
  deps: PoolBackendLaunchDeps<TBackend, TRemote>
): Promise<PoolBackendLaunch<TBackend, TRemote>> {
  const remote = await deps.resolveRemoteBackend(profile)

  if (remote) {
    await deps.waitForHermes(remote.baseUrl, remote.token)

    return { kind: 'remote', remote }
  }

  await deps.waitForUpdateToFinish()

  const token = deps.createToken()
  const backendArgs = ['--profile', profile, 'serve', '--host', '127.0.0.1', '--port', '0']
  const backend = await deps.ensureRuntime(deps.resolveHermesBackend(backendArgs))
  backend.args = deps.getBackendArgsForRuntime(backend)

  return { kind: 'local', backend, token }
}
