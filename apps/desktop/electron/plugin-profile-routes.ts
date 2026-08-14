import crypto from 'node:crypto'

export interface ProfileRouteConfig {
  cloudOrg: string
  mode: 'cloud' | 'local' | 'remote' | 'ssh'
  remoteUrl: string
  sshHost: string
  sshPort: null | number
  sshRemoteHermesPath: string
  sshRemoteProfile: string
  sshUser: string
}

export interface EffectiveSshRoute {
  hostname: string
  port: null | number
  user: string
}

export interface OpaqueProfileRoute {
  connectionId: string
  mode: 'local' | 'remote'
  profile: string
  targetProfile: string
}

interface BuildOpaqueProfileRoutesOptions {
  getProfileConfig: (profile: string) => ProfileRouteConfig | Promise<ProfileRouteConfig>
  globalConfig: ProfileRouteConfig
  installationId: string
  primaryProfile: string
  profileNames: string[]
  resolveSsh: (config: ProfileRouteConfig) => Promise<EffectiveSshRoute>
}

function normalizeProfile(name: null | string | undefined): string {
  return String(name ?? '').trim() || 'default'
}

function normalizeRemoteUrl(raw: string): string {
  const value = String(raw || '').trim()

  try {
    const url = new URL(value)
    url.username = ''
    url.password = ''
    url.search = ''
    url.hash = ''
    url.pathname = url.pathname.replace(/\/+$/, '') || '/'

    return url.toString().replace(/\/$/, '')
  } catch {
    return value.toLowerCase().replace(/\/+$/, '')
  }
}

async function connectionScope(
  config: ProfileRouteConfig,
  resolveSsh: BuildOpaqueProfileRoutesOptions['resolveSsh']
): Promise<{ key: string; mode: 'local' | 'remote' }> {
  if (config.mode === 'ssh') {
    const effective = await resolveSsh(config)

    // Remote profile is intentionally excluded: profiles mapped into the same
    // remote Hermes home form one interaction scope. Key/identity-file paths are
    // credentials and likewise stay out of the scope material.
    return {
      key: [
        'ssh',
        effective.user.trim(),
        effective.hostname.trim().toLowerCase(),
        effective.port ?? 22,
        config.sshRemoteHermesPath.trim()
      ].join('\0'),
      mode: 'remote'
    }
  }

  if (config.mode === 'cloud') {
    return {
      key: `cloud\0${normalizeRemoteUrl(config.remoteUrl)}\0${config.cloudOrg.trim()}`,
      mode: 'remote'
    }
  }

  if (config.mode === 'remote') {
    return { key: `remote\0${normalizeRemoteUrl(config.remoteUrl)}`, mode: 'remote' }
  }

  return { key: 'local', mode: 'local' }
}

function opaqueConnectionId(scope: string, installationId: string): string {
  const digest = crypto.createHmac('sha256', installationId).update(scope).digest('hex')

  return `connection-${digest.slice(0, 24)}`
}

/**
 * Resolve Desktop routing profiles at the Electron boundary and return only
 * keyed, credential-free descriptors to the renderer/plugin runtime.
 */
export async function buildOpaqueProfileRoutes({
  getProfileConfig,
  globalConfig,
  installationId,
  primaryProfile,
  profileNames,
  resolveSsh
}: BuildOpaqueProfileRoutesOptions): Promise<OpaqueProfileRoute[]> {
  const primary = normalizeProfile(primaryProfile)
  const names: string[] = []
  const seen = new Set<string>()

  for (const raw of [primary, ...profileNames]) {
    const name = normalizeProfile(raw)

    if (!seen.has(name)) {
      seen.add(name)
      names.push(name)
    }
  }

  const globalScope = await connectionScope(globalConfig, resolveSsh)

  return Promise.all(
    names.map(async profile => {
      const scoped = await getProfileConfig(profile)
      const scope = scoped.mode === 'local' ? globalScope : await connectionScope(scoped, resolveSsh)

      return {
        connectionId: opaqueConnectionId(scope.key, installationId),
        mode: scope.mode,
        profile,
        targetProfile:
          scoped.mode === 'ssh' ? normalizeProfile(scoped.sshRemoteProfile || profile) : profile
      }
    })
  )
}
