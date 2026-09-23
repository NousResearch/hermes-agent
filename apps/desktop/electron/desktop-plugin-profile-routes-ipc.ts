import {
  buildRegistryProfileRoutes,
  isLocalEnumerationFailure,
  localRouteFallbackProfiles,
  undialedSshRouteSeeds
} from './plugin-profile-routes'

export interface DesktopPluginProfileRoutesIpcDeps {
  ipcMain: any
  readDesktopConnectionConfig: any
  sanitizeDesktopConnectionConfig: any
  readDesktopConnectionsRegistry: any
  enumerateRegistryAgentSources: any
  buildAgentRoster: any
}

export function registerDesktopPluginProfileRoutesIpc(deps: DesktopPluginProfileRoutesIpcDeps) {
  const {
    ipcMain,
    readDesktopConnectionConfig,
    sanitizeDesktopConnectionConfig,
    readDesktopConnectionsRegistry,
    enumerateRegistryAgentSources,
    buildAgentRoster
  } = deps

  ipcMain.handle('hermes:connection-config:get', async (_event, profile) =>
    sanitizeDesktopConnectionConfig(readDesktopConnectionConfig(), profile)
  )
  ipcMain.handle('hermes:plugin-profile-routes', async (_event, rawProfileNames) => {
    const fallbackProfileNames = Array.isArray(rawProfileNames)
      ? rawProfileNames
          .filter(name => typeof name === 'string')
          .map(name => name.trim())
          .filter(Boolean)
          .slice(0, 256)
      : []

    const registry = readDesktopConnectionsRegistry()
    const enumerations = await enumerateRegistryAgentSources(registry)
    let agents = buildAgentRoster(enumerations, { primaryConnectionId: registry.primary })

    // Roster enumeration deliberately does not dial connect-on-demand SSH
    // sources. Publish one credential-free seed route so a plugin can be the
    // first caller that opens the tunnel.
    const sshSeeds = undialedSshRouteSeeds(agents, registry.connections)

    if (sshSeeds.length > 0) {
      agents = [
        ...agents,
        ...sshSeeds.map(seed => {
          const source = registry.connections.find(connection => connection.id === seed.connectionId)!

          return {
            connectionId: source.id,
            connectionKind: source.kind,
            connectionLabel: source.label,
            handle: seed.profile,
            profile: seed.profile
          }
        })
      ]
    }

    // A local enumeration can fail while remote/cloud sources succeed. Preserve
    // cached v1 profile names as explicitly-local rows so those valid routes do
    // not disappear and duplicate names remain source-qualified.
    const localSource = registry.connections.find(source => source.kind === 'local')

    const localEnumeration = localSource
      ? enumerations.find(({ connection }) => connection.id === localSource.id)
      : undefined

    const localFallbackProfiles = localSource
      ? localRouteFallbackProfiles(
          agents,
          localSource.id,
          fallbackProfileNames,
          isLocalEnumerationFailure(localEnumeration?.error)
        )
      : []

    if (localSource && localFallbackProfiles.length > 0) {
      agents = [
        ...agents,
        ...localFallbackProfiles.map(profile => ({
          connectionId: localSource.id,
          connectionKind: localSource.kind,
          connectionLabel: localSource.label,
          handle: profile,
          profile
        }))
      ]
    }

    return buildRegistryProfileRoutes({ agents, sources: registry.connections })
  })
}
