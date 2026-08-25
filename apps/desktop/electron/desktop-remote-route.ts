import {
  connectionScopeKey,
  modeIsRemoteLike,
  normalizeSshConfig,
  normAuthMode,
  profileRemoteOverride,
  profileSshOverride
} from './connection-config'
import type { ConnectionRegistry } from './connection-registry'
import { matchingConnectionId, type StoredRoute } from './connection-route-identity'

type RouteSource = 'env' | 'profile' | 'settings'

interface SshRouteConfig {
  host: string
  keyPath?: string
  mode: 'ssh'
  port?: number
  remoteHermesPath?: string
  remoteProfile?: string
  user?: string
}

export type DesktopRemoteRoute =
  | {
      authMode: 'oauth' | 'token'
      connectionId?: string
      headers?: Record<string, unknown>
      kind: 'cloud' | 'remote'
      org?: string
      source: RouteSource
      token?: unknown
      url: string
    }
  | {
      connectionId?: string
      kind: 'ssh'
      source: Exclude<RouteSource, 'env'>
      ssh: SshRouteConfig
      token?: unknown
    }

export interface DesktopRemoteRouteInput {
  config: Record<string, any>
  env?: { token?: null | string; url?: null | string }
  profile?: null | string
  registry: ConnectionRegistry
}

function withConnectionId<T extends object>(route: T, connectionId?: string): T & { connectionId?: string } {
  return connectionId ? { ...route, connectionId } : route
}

/**
 * Select one remote route with the existing precedence and freeze any exact
 * registry identity before I/O. A null result means the profile resolves
 * locally. Invalid dial data remains the dialler's error, except the existing
 * env-pair validation which belongs to selection.
 */
export function resolveDesktopRemoteRoute({
  config,
  env = {},
  profile,
  registry
}: DesktopRemoteRouteInput): DesktopRemoteRoute | null {
  const profileKey = connectionScopeKey(profile)
  const profileConfig = profileKey ? config.profiles?.[profileKey] : null
  const sshOverride = profileSshOverride(config, profile)

  if (sshOverride) {
    const route = { ...sshOverride, kind: 'ssh' as const }

    return withConnectionId(
      {
        kind: 'ssh' as const,
        source: 'profile' as const,
        ssh: sshOverride,
        token: profileConfig?.token
      },
      matchingConnectionId(registry, route, 'unique')
    )
  }

  const override = profileRemoteOverride(config, profile)

  if (override) {
    const kind = profileConfig?.mode === 'cloud' ? 'cloud' : 'remote'
    const authMode = override.authMode === 'oauth' ? 'oauth' : 'token'
    const route = { ...profileConfig, kind } as StoredRoute

    return withConnectionId(
      {
        authMode,
        headers: override.headers,
        kind,
        org: kind === 'cloud' ? String(profileConfig?.org || '').trim() || undefined : undefined,
        source: 'profile' as const,
        token: override.token,
        url: override.url
      },
      matchingConnectionId(registry, route, 'unique')
    )
  }

  const envUrl = String(env.url || '').trim()

  if (envUrl) {
    const envToken = String(env.token || '').trim()

    if (!envToken) {
      throw new Error(
        'HERMES_DESKTOP_REMOTE_URL is set but HERMES_DESKTOP_REMOTE_TOKEN is not. ' +
          'Both must be provided to connect to a remote Hermes backend.'
      )
    }

    return { authMode: 'token', kind: 'remote', source: 'env', token: envToken, url: envUrl }
  }

  if (config.mode === 'ssh') {
    const ssh = normalizeSshConfig({ mode: 'ssh', ...(config.remote || {}) })

    if (!ssh) {
      throw new Error('SSH remote mode is selected but no host is configured.')
    }

    const route = { ...ssh, kind: 'ssh' as const }

    return withConnectionId(
      { kind: 'ssh' as const, source: 'settings' as const, ssh, token: config.remote?.token },
      matchingConnectionId(registry, route, 'primary')
    )
  }

  if (!modeIsRemoteLike(config.mode)) {
    return null
  }

  const kind = config.mode === 'cloud' ? 'cloud' : 'remote'
  const authMode = normAuthMode(config.remote?.authMode)
  const route = { ...config.remote, kind } as StoredRoute

  return withConnectionId(
    {
      authMode,
      headers: config.remote?.headers,
      kind,
      org: kind === 'cloud' ? String(config.remote?.org || '').trim() || undefined : undefined,
      source: 'settings' as const,
      token: config.remote?.token,
      url: String(config.remote?.url || '')
    },
    matchingConnectionId(registry, route, 'primary')
  )
}
