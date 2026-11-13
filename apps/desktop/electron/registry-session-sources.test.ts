import { describe, expect, it } from 'vitest'

import { hasPinnedRegistrySessionSource, mergeRegistrySessionRows } from './profile-session-routing'
import { collectRegistrySessionSources } from './registry-session-sources'
import { resolveSharedSshProfileBackend, sharedSshProfiles } from './ssh-shared-profile-backend'

// Box (SSH) is the window's primary; a URL remote is connected as a registry
// gateway. Box's registry routes reuse the primary descriptor, so the pool only
// holds the URL remote.
const primary = { baseUrl: 'http://127.0.0.1:4310', mode: 'remote', remoteKind: 'ssh', profile: 'default' }
const lan = { baseUrl: 'http://127.0.0.1:4320', mode: 'remote', remoteKind: 'url' }

const rowsByHost: Record<string, unknown[]> = {
  [primary.baseUrl]: [
    { id: 'box-default-1', profile: 'default', last_active: 3 },
    { id: 'box-gamma-1', profile: 'gamma', last_active: 2 }
  ],
  [lan.baseUrl]: [{ id: 'lan-eps-1', profile: 'eps', last_active: 1 }]
}

async function getJson(descriptor: unknown, path: string) {
  const rows = rowsByHost[(descriptor as { baseUrl: string }).baseUrl] ?? []

  return path.startsWith('/api/profiles/sessions') ? { sessions: rows, total: rows.length } : { sessions: [], total: 0 }
}

describe('registry session sources with the SSH primary reused', () => {
  it('keeps the primary SSH gateway in the All-profiles aggregate after a named profile rides it', async () => {
    // Selecting box/gamma: gamma rides the default backend, which is the primary.
    const gamma = await resolveSharedSshProfileBackend({ id: 'box', kind: 'ssh' }, 'gamma', {
      ensureDefault: async () => ({ ...primary, connectionId: 'box' }),
      listProfiles: async () => ({ profiles: [{ name: 'default' }, { name: 'gamma' }] })
    })

    expect(gamma?.baseUrl).toBe(primary.baseUrl)

    const sources = await collectRegistrySessionSources(
      {
        connections: [
          { id: 'local', kind: 'local' },
          { id: 'box', kind: 'ssh' },
          { id: 'lan', kind: 'remote' }
        ],
        pool: [['conn:lan::default', { connectionPromise: Promise.resolve(lan) }]],
        scopePrefix: id => `conn:${id}::`,
        reusedPrimary: async id => (id === 'box' ? primary : null),
        installId: async id => `install-${id}`,
        sharedProfiles: sharedSshProfiles
      },
      true
    )

    expect(hasPinnedRegistrySessionSource('box', 'gamma', sources, false)).toBe(true)

    const merged: unknown[] = []
    await mergeRegistrySessionRows(sources, new URLSearchParams('limit=50'), getJson, merged, {})

    const scopes = merged.map(row => {
      const { connection_id, id } = row as { connection_id: string; id: string }

      return `${connection_id}/${id}`
    })

    expect(scopes.sort()).toEqual(['box/box-default-1', 'box/box-gamma-1', 'lan/lan-eps-1'])
  })
})
