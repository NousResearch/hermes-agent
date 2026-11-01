import { describe, expect, it } from 'vitest'

import { registryGatewayWsUrl } from './plugin-profile-routes'
import { resolveSharedSshProfileBackend } from './ssh-shared-profile-backend'

const ssh = { id: 'box', kind: 'ssh' as const }

let nextPort = 4100

function defaultBackend(profiles: unknown) {
  const calls = { ensure: 0, list: 0 }
  const port = nextPort++
  const base = { baseUrl: `http://127.0.0.1:${port}`, wsUrl: `ws://127.0.0.1:${port}/api/ws?token=t`, profile: 'default' }

  return {
    calls,
    deps: {
      ensureDefault: async () => {
        calls.ensure += 1

        return base
      },
      listProfiles: async () => {
        calls.list += 1

        if (profiles instanceof Error) {
          throw profiles
        }

        return profiles
      }
    }
  }
}

describe('SSH named profiles share the connection backend', () => {
  it('serves a listed profile from the default backend, scoped per request and per socket', async () => {
    const { calls, deps } = defaultBackend({ profiles: [{ name: 'default' }, { name: 'gamma' }, { name: 'delta' }] })

    const gamma = await resolveSharedSshProfileBackend(ssh, 'gamma', deps)
    const delta = await resolveSharedSshProfileBackend(ssh, 'delta', deps)

    expect(gamma).toMatchObject({ baseUrl: 'http://127.0.0.1:4100', wsUrl: expect.any(String), profile: 'gamma', sharedRemote: true, connectionId: 'box' })
    expect(delta?.baseUrl).toBe(gamma?.baseUrl)
    expect(new URL(registryGatewayWsUrl(gamma!, gamma!.wsUrl)).searchParams.get('profile')).toBe('gamma')
    // Both profiles ride the one backend; its roster is read once.
    expect(calls).toEqual({ ensure: 2, list: 1 })
  })

  it('keeps the per-profile backend when the shared one cannot serve the profile', async () => {
    const listed = defaultBackend({ profiles: [{ name: 'default' }] })
    const old = defaultBackend(new Error('404 Not Found'))
    const pinned = defaultBackend({ profiles: [{ name: 'gamma' }] })

    expect(await resolveSharedSshProfileBackend(ssh, 'gamma', listed.deps)).toBeNull()
    expect(await resolveSharedSshProfileBackend(ssh, 'gamma', old.deps)).toBeNull()
    expect(await resolveSharedSshProfileBackend({ ...ssh, remoteProfile: 'gamma' }, 'gamma', pinned.deps)).toBeNull()
    expect(await resolveSharedSshProfileBackend(ssh, 'default', pinned.deps)).toBeNull()
    expect(await resolveSharedSshProfileBackend({ id: 'lan', kind: 'remote' }, 'gamma', pinned.deps)).toBeNull()
    expect(pinned.calls.ensure).toBe(0)
  })
})
