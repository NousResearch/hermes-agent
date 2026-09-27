import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assembleSidebarSessionSlices,
  buildSidebarSessionSliceParams,
  fetchPrimaryProfileSessions,
  fetchRegistrySessionRows,
  fetchRemoteProfileSessions,
  findRemoteOwnerProfileForSession,
  mergeProfileSessionWindow,
  pathWithRemoteOwnerScope,
  remoteProfileQueryScope,
  spliceRegistrySessionRows,
  tagRegistrySessionResponse,
  tagRemoteSessionRows
} from './profile-session-routing'

test('remote sidebar slices all follow the selected profile', () => {
  const slices = buildSidebarSessionSliceParams(
    new URLSearchParams({
      recents_profile: 'work-vps',
      recents_limit: '30',
      cron_limit: '40',
      messaging_limit: '50',
      recents_exclude: 'cron,signal',
      messaging_exclude: 'desktop,cron'
    })
  )

  assert.equal(slices.recents.get('profile'), 'work-vps')
  assert.equal(slices.cron.get('profile'), 'work-vps')
  assert.equal(slices.messaging.get('profile'), 'work-vps')
  assert.equal(slices.recents.get('exclude_sources'), 'cron,signal')
  assert.equal(slices.cron.get('source'), 'cron')
  assert.equal(slices.messaging.get('exclude_sources'), 'desktop,cron')
})

test('remote sidebar slices preserve the explicit all-profiles scope', () => {
  const slices = buildSidebarSessionSliceParams(new URLSearchParams({ recents_profile: 'all' }))

  assert.deepEqual(
    Object.values(slices).map(params => params.get('profile')),
    ['all', 'all', 'all']
  )
})

test('remote sidebar slices fall back to the all-profiles scope', () => {
  for (const searchParams of [new URLSearchParams(), new URLSearchParams({ recents_profile: '   ' })]) {
    const slices = buildSidebarSessionSliceParams(searchParams)

    assert.deepEqual(
      Object.values(slices).map(params => params.get('profile')),
      ['all', 'all', 'all']
    )
  }
})

test('primary session reads use the profile-aware request path', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const expected = { sessions: [{ id: 'session-1' }], total: 1, profile_totals: { default: 1 } }

  const result = await fetchPrimaryProfileSessions(
    new URLSearchParams({ profile: 'default', limit: '20' }),
    async (profile, path) => {
      calls.push({ profile, path })

      return expected
    }
  )

  assert.deepEqual(calls, [{ profile: null, path: '/api/profiles/sessions?profile=default&limit=20' }])
  assert.equal(result, expected)
})

// A failed primary read is an empty page the renderer must NOT treat as the
// profile's truth: without errors[] it replaced the sidebar with "No sessions"
// and nothing said why (#67600). Report the failed scope the same way the
// backend reports a failed profile scan.
test('primary session reads report a failed read in errors', async () => {
  const result = await fetchPrimaryProfileSessions(new URLSearchParams({ profile: 'default' }), async () => {
    throw new Error('remote unavailable')
  })

  assert.deepEqual(result, {
    sessions: [],
    total: 0,
    profile_totals: {},
    errors: [{ profile: 'default', error: 'remote unavailable' }]
  })
})

test('a failed unified primary read names the whole aggregate', async () => {
  const result = await fetchPrimaryProfileSessions(new URLSearchParams({ limit: '20' }), async () => {
    throw new Error('timed out')
  })

  assert.deepEqual(result.errors, [{ profile: 'all', error: 'timed out' }])
})

test('reassembled sidebar slices keep each slice errors', () => {
  const failed = [{ profile: 'default', error: 'timed out' }]

  const result = assembleSidebarSessionSlices(
    { sessions: [], total: 0, profile_totals: {}, errors: failed },
    { sessions: [{ id: 'cron-1' }], total: 1 },
    { sessions: [], total: 0, errors: failed }
  )

  assert.deepEqual(result.recents.errors, failed)
  assert.equal(result.cron.errors, undefined)
  assert.deepEqual(result.messaging.errors, failed)
  assert.deepEqual(result.cron.sessions, [{ id: 'cron-1' }])
})

test('remote session reads split oversized sidebar windows into API-safe pages', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const rows = Array.from({ length: 250 }, (_, index) => ({ id: `session-${index}` }))

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '300', offset: '0', order: 'updated' }),
    async (profile, path) => {
      calls.push({ profile, path })
      const url = new URL(path, 'http://desktop.test')
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))

      if (limit > 100) {
        throw new Error(`remote /api/sessions rejects limit ${limit}`)
      }

      return {
        sessions: rows.slice(offset, offset + limit),
        total: rows.length,
        limit,
        offset
      }
    }
  )

  assert.deepEqual(calls, [
    { profile: 'remote-work', path: '/api/sessions?limit=100&offset=0&order=updated&profile=remote-work' },
    { profile: 'remote-work', path: '/api/sessions?limit=100&offset=100&order=updated&profile=remote-work' },
    { profile: 'remote-work', path: '/api/sessions?limit=50&offset=200&order=updated&profile=remote-work' }
  ])
  assert.equal(result.sessions.length, 250)
  assert.equal(result.total, 250)
  assert.equal(result.limit, 300)
  assert.equal(result.offset, 0)
  assert.deepEqual(
    result.sessions.map(row => (row as { id: string }).id),
    rows.map(row => row.id)
  )
})

test('remote paging preserves offsets and deduplicates pinned backfill rows', async () => {
  const calls: string[] = []

  const rows = Array.from({ length: 240 }, (_, index) => ({
    id: `session-${index}`,
    pinned: index === 20 || index === 200
  }))

  const pinned = rows.filter(row => row.pinned)

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '150', offset: '80' }),
    async (_profile, path) => {
      calls.push(path)
      const url = new URL(path, 'http://desktop.test')
      const limit = Number(url.searchParams.get('limit'))
      const offset = Number(url.searchParams.get('offset'))
      const window = rows.slice(offset, offset + limit)
      const windowIds = new Set(window.map(row => row.id))

      return {
        sessions: [...window, ...pinned.filter(row => !windowIds.has(row.id))],
        total: rows.length,
        limit,
        offset
      }
    }
  )

  assert.deepEqual(calls, [
    '/api/sessions?limit=100&offset=80&profile=remote-work',
    '/api/sessions?limit=50&offset=180&profile=remote-work'
  ])
  assert.deepEqual(
    result.sessions.map(row => (row as { id: string }).id),
    [...rows.slice(80, 230).map(row => row.id), 'session-20']
  )
})

test('remote paging treats malformed totals as unknown instead of truncating the result', async () => {
  const rows = Array.from({ length: 250 }, (_, index) => ({ id: `session-${index}` }))

  for (const malformedTotal of [null, '', false, 100.5]) {
    const calls: string[] = []

    const result = await fetchRemoteProfileSessions(
      'remote-work',
      new URLSearchParams({ limit: '300', offset: '0' }),
      async (_profile, path) => {
        calls.push(path)
        const url = new URL(path, 'http://desktop.test')
        const limit = Number(url.searchParams.get('limit'))
        const offset = Number(url.searchParams.get('offset'))

        return {
          sessions: rows.slice(offset, offset + limit),
          total: malformedTotal,
          limit,
          offset
        }
      }
    )

    assert.deepEqual(calls, [
      '/api/sessions?limit=100&offset=0&profile=remote-work',
      '/api/sessions?limit=100&offset=100&profile=remote-work',
      '/api/sessions?limit=100&offset=200&profile=remote-work'
    ])
    assert.equal(result.sessions.length, 250)
    assert.equal(result.total, 250)
  }
})

test('merged profile windows retain pinned rows outside the recency window', () => {
  const rows = [
    { id: 'recent-default', profile: 'default', pinned: false },
    { id: 'shared-id', profile: 'default', pinned: false },
    { id: 'recent-remote', profile: 'remote-work', pinned: false },
    { id: 'shared-id', profile: 'remote-work', pinned: true },
    { id: 'old-remote', profile: 'remote-work', pinned: true },
    { id: 'old-unpinned', profile: 'remote-work', pinned: false }
  ]

  assert.deepEqual(mergeProfileSessionWindow(rows, 0, 3), [rows[0], rows[1], rows[2], rows[3], rows[4]])
})

test('remote session reads keep small requests on one call', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []
  const expected = { sessions: [{ id: 'session-1' }], total: 1, limit: 20, offset: 0 }

  const result = await fetchRemoteProfileSessions(
    'remote-work',
    new URLSearchParams({ profile: 'remote-work', limit: '20', offset: '0' }),
    async (profile, path) => {
      calls.push({ profile, path })

      return expected
    }
  )

  assert.deepEqual(calls, [{ profile: 'remote-work', path: '/api/sessions?limit=20&offset=0&profile=remote-work' }])
  assert.equal(result, expected)
})

test('registry sources: ssh backends are read natively and rows tagged with connection + profile', async () => {
  const calls: Array<{ descriptor: unknown; path: string }> = []

  const rows = await fetchRegistrySessionRows(
    [
      {
        connectionId: 'gw-spark',
        kind: 'ssh',
        backends: [
          { descriptor: 'spark-desc', profileLabel: 'research' },
          { descriptor: 'spark-desc-2', profileLabel: '' }
        ]
      }
    ],
    new URLSearchParams({ limit: '20', offset: '0', profile: 'all' }),
    async (descriptor, path) => {
      calls.push({ descriptor, path })

      return { sessions: [{ id: `s-${descriptor}`, message_count: 3 }], total: 1 }
    }
  )

  assert.equal(calls.length, 2)
  // The remote serves its own state.db: no profile param forwarded.
  assert.ok(calls.every(({ path }) => path.startsWith('/api/sessions?') && !path.includes('profile=')))
  // Hidden Bot Mode chats must stay hidden — include_hidden is never requested.
  assert.ok(calls.every(({ path }) => !path.includes('include_hidden')))

  assert.deepEqual(
    rows.map(row => [(row as any).id, (row as any).connection_id, (row as any).profile]),
    [
      ['s-spark-desc', 'gw-spark', 'research'],
      ['s-spark-desc-2', 'gw-spark', 'default']
    ]
  )
  assert.ok(rows.every(row => (row as any).is_default_profile === false))
})

test('registry sources: shared remote hosts read the cross-profile aggregate once', async () => {
  const calls: string[] = []

  const rows = await fetchRegistrySessionRows(
    [
      {
        connectionId: 'gw-cloud',
        kind: 'remote',
        backends: [{ descriptor: 'cloud-desc', profileLabel: null }]
      }
    ],
    new URLSearchParams({ limit: '20', offset: '0' }),
    async (_descriptor, path) => {
      calls.push(path)

      return {
        sessions: [
          { id: 'r-1', profile: 'hermes-claude' },
          { id: 'r-2', profile: '' }
        ],
        total: 2
      }
    }
  )

  assert.equal(calls.length, 1)
  assert.ok(calls[0].startsWith('/api/profiles/sessions?'))
  assert.ok(calls[0].includes('profile=all'))
  assert.ok(!calls[0].includes('include_hidden'))

  // The remote's own profile stamps survive; missing stamps get 'default'.
  assert.deepEqual(
    rows.map(row => [(row as any).id, (row as any).profile, (row as any).connection_id]),
    [
      ['r-1', 'hermes-claude', 'gw-cloud'],
      ['r-2', 'default', 'gw-cloud']
    ]
  )
})

test('registry-pinned session responses retain their owning connection', () => {
  const sidebar = tagRegistrySessionResponse(
    '/api/profiles/sessions/sidebar?recents_profile=default',
    {
      recents: { sessions: [{ id: 'remote-chat', profile: 'default' }] },
      cron: { sessions: [{ id: 'remote-cron', profile: 'default' }] },
      messaging: { sessions: [] }
    },
    'test-amnezia'
  ) as any

  assert.equal(sidebar.recents.sessions[0].connection_id, 'test-amnezia')
  assert.equal(sidebar.cron.sessions[0].connection_id, 'test-amnezia')

  const aggregate = tagRegistrySessionResponse(
    '/api/profiles/sessions?profile=all',
    { sessions: [{ id: 'remote-profile-chat', profile: 'research' }] },
    'test-amnezia'
  ) as any

  assert.equal(aggregate.sessions[0].connection_id, 'test-amnezia')

  const single = tagRegistrySessionResponse(
    '/api/sessions/remote-chat?profile=default',
    { id: 'remote-chat', profile: 'default' },
    'test-amnezia'
  ) as any

  assert.equal(single.connection_id, 'test-amnezia')
})

test('registry response ownership tagging ignores non-session payloads and transcript messages', () => {
  const status = { ok: true }
  const messages = { messages: [{ id: 'message-1' }], session_id: 'remote-chat' }

  assert.equal(tagRegistrySessionResponse('/api/status', status, 'test-amnezia'), status)
  assert.equal(tagRegistrySessionResponse('/api/sessions/remote-chat/messages', messages, 'test-amnezia'), messages)
  assert.equal((messages.messages[0] as any).connection_id, undefined)
})

test('registry sources: an older shared host without the aggregator falls back to its flat list', async () => {
  const calls: string[] = []

  const rows = await fetchRegistrySessionRows(
    [{ connectionId: 'gw-old', kind: 'remote', backends: [{ descriptor: 'old-desc', profileLabel: null }] }],
    new URLSearchParams({ limit: '20' }),
    async (_descriptor, path) => {
      calls.push(path)

      if (path.startsWith('/api/profiles/sessions')) {
        throw new Error('404: No such API endpoint')
      }

      return { sessions: [{ id: 'legacy-1' }], total: 1 }
    }
  )

  assert.equal(calls.length, 2)
  assert.ok(calls[1].startsWith('/api/sessions?'))
  assert.deepEqual(
    rows.map(row => [(row as any).id, (row as any).profile]),
    [['legacy-1', 'default']]
  )
})

test('registry sources: a dead gateway contributes nothing instead of breaking the list', async () => {
  const rows = await fetchRegistrySessionRows(
    [
      { connectionId: 'gw-dead', kind: 'ssh', backends: [{ descriptor: 'dead', profileLabel: 'x' }] },
      { connectionId: 'gw-live', kind: 'ssh', backends: [{ descriptor: 'live', profileLabel: 'y' }] }
    ],
    new URLSearchParams({ limit: '10' }),
    async descriptor => {
      if (descriptor === 'dead') {
        throw new Error('ECONNREFUSED')
      }

      return { sessions: [{ id: 'ok-1' }], total: 1 }
    }
  )

  assert.deepEqual(
    rows.map(row => (row as any).id),
    ['ok-1']
  )
})

test('splice: registry rows dedupe by id and extend per-profile totals', () => {
  const merged: unknown[] = [
    { id: 'local-1', profile: 'default', last_active: 100 },
    { id: 'dupe', profile: 'work', last_active: 90 }
  ]

  const totals: Record<string, number> = { default: 1, work: 1 }

  const { added } = spliceRegistrySessionRows(
    merged,
    [
      { id: 'dupe', profile: 'work', connection_id: 'gw-1', last_active: 95 },
      { id: 'remote-1', profile: 'hermes-claude', connection_id: 'gw-1', last_active: 120 },
      { id: 'remote-2', connection_id: 'gw-1', last_active: 110 }
    ],
    totals
  )

  assert.equal(added, 2)
  assert.deepEqual(
    merged.map(row => (row as any).id),
    ['local-1', 'dupe', 'remote-1', 'remote-2']
  )
  assert.equal(totals['hermes-claude'], 1)
  assert.equal(totals.default, 2) // untagged registry row counts under default
  assert.equal(totals.work, 1) // deduped row does not double-count
})

test('finds the remote owner profile for a hint-less session read (#85834)', async () => {
  const owner = await findRemoteOwnerProfileForSession('sess-remote', ['vps-a', 'vps-b'], async profile => {
    if (profile === 'vps-b') {
      return { sessions: [{ id: 'sess-remote' }] } as never
    }

    return { sessions: [{ id: 'other' }] } as never
  })

  assert.equal(owner, 'vps-b')
})

test('remote owner lookup matches a compression lineage root id too', async () => {
  const owner = await findRemoteOwnerProfileForSession('root-1', ['vps-a'], async () => {
    return { sessions: [{ id: 'tip-2', _lineage_root_id: 'root-1' }] } as never
  })

  assert.equal(owner, 'vps-a')
})

test('remote owner lookup returns null when no remote lists the id or remotes fail', async () => {
  const missing = await findRemoteOwnerProfileForSession('sess-x', ['vps-a'], async () => {
    return { sessions: [{ id: 'other' }] } as never
  })

  assert.equal(missing, null)

  const dead = await findRemoteOwnerProfileForSession('sess-x', ['vps-a'], async () => {
    throw new Error('remote unavailable')
  })

  assert.equal(dead, null)

  const noRemotes = await findRemoteOwnerProfileForSession('sess-x', [], async () => {
    throw new Error('never called')
  })

  assert.equal(noRemotes, null)
})

// #64999: two Desktop profile scopes can point at the SAME multi-profile
// remote backend. The list read must name the requested profile scope so the
// backend opens that profile's state.db — an unscoped read returns the
// backend's launch-profile rows, and the old unconditional relabel stamped
// them as whichever local scope happened to ask.
test('remote session reads carry the profile scope against a multi-profile backend', async () => {
  const calls: Array<{ profile: string | null; path: string }> = []

  await fetchRemoteProfileSessions(
    'wife',
    new URLSearchParams({ profile: 'wife', limit: '20', offset: '0' }),
    async (profile, path) => {
      calls.push({ profile, path })

      return { sessions: [{ id: 's-1', profile: 'wife' }], total: 1, limit: 20, offset: 0 }
    }
  )

  assert.deepEqual(calls, [{ profile: 'wife', path: '/api/sessions?limit=20&offset=0&profile=wife' }])
})

// The same read for the OTHER scope sharing the backend names ITS scope — the
// backend's launch profile (say `dad`) is never read for a `wife` request.
test('two scopes sharing one multi-profile backend each read their own profile', async () => {
  const seenProfiles: string[] = []

  for (const scope of ['wife', 'dad']) {
    const result = await fetchRemoteProfileSessions(
      scope,
      new URLSearchParams({ profile: scope, limit: '20', offset: '0' }),
      async (_profile, path) => {
        const url = new URL(path, 'http://desktop.test')
        const servedProfile = url.searchParams.get('profile') || 'launch-profile'
        seenProfiles.push(servedProfile)

        return { sessions: [{ id: `s-${servedProfile}`, profile: servedProfile }], total: 1 }
      }
    )

    assert.equal((result.sessions[0] as { profile: string }).profile, scope)
  }

  assert.deepEqual(seenProfiles, ['wife', 'dad'])
})

// A remote that rejects the scope (400/404: profile does not exist there) is
// the legacy single-launch-profile shape — fall back to its own database.
test('remote session reads fall back to the unscoped list when the remote rejects the profile scope', async () => {
  const calls: string[] = []

  const result = await fetchRemoteProfileSessions(
    'wife',
    new URLSearchParams({ profile: 'wife', limit: '20', offset: '0' }),
    async (_profile, path) => {
      calls.push(path)

      if (path.includes('profile=wife')) {
        const error: any = new Error("404: Profile 'wife' does not exist.")
        error.statusCode = 404
        throw error
      }

      return { sessions: [{ id: 's-1' }], total: 1 }
    }
  )

  assert.deepEqual(calls, ['/api/sessions?limit=20&offset=0&profile=wife', '/api/sessions?limit=20&offset=0'])
  assert.equal((result.sessions[0] as { id: string }).id, 's-1')
})

// Auth/transport/5xx failures are real errors — the unscoped fallback must
// not swallow them.
test('remote session reads propagate non-scope errors without the fallback', async () => {
  const calls: string[] = []

  await assert.rejects(
    fetchRemoteProfileSessions(
      'wife',
      new URLSearchParams({ profile: 'wife', limit: '20', offset: '0' }),
      async (_profile, path) => {
        calls.push(path)
        const error: any = new Error('503: Service Unavailable')
        error.statusCode = 503
        throw error
      }
    ),
    /503/
  )

  assert.deepEqual(calls, ['/api/sessions?limit=20&offset=0&profile=wife'])
})

// A managed-SSH override can map the Desktop label to the remote's own
// profile name (remoteProfile); the scope sent on the wire is the alias.
test('remote session reads use the managed-SSH remoteProfile alias as the scope', async () => {
  const calls: string[] = []

  await fetchRemoteProfileSessions(
    'mara',
    new URLSearchParams({ profile: 'mara', limit: '20', offset: '0' }),
    async (_profile, path) => {
      calls.push(path)

      return { sessions: [], total: 0 }
    },
    { remoteProfileAlias: 'dixie' }
  )

  assert.deepEqual(calls, ['/api/sessions?limit=20&offset=0&profile=dixie'])
})

// Every concrete scope — `default` included — is named on the wire: a
// multi-profile backend's launch profile is not necessarily `default`, so an
// unscoped read could relaunch the bug for the default scope too.
test('remote session reads for the default scope name it explicitly', async () => {
  const calls: string[] = []

  await fetchRemoteProfileSessions(
    'default',
    new URLSearchParams({ profile: 'default', limit: '20', offset: '0' }),
    async (_profile, path) => {
      calls.push(path)

      return { sessions: [], total: 0 }
    }
  )

  assert.deepEqual(calls, ['/api/sessions?limit=20&offset=0&profile=default'])
})

// The remote's own profile stamp is authoritative; the splice must only
// backfill unowned rows instead of relabeling every one (#64999).
test('remote list rows keep the remote profile stamp instead of the desktop scope label', () => {
  const rows: Array<Record<string, unknown>> = [
    { id: 'w-1', profile: 'wife' },
    { id: 'd-1', profile: 'dad' },
    { id: 'legacy-1' },
    { id: 'legacy-2', profile: '' }
  ]

  tagRemoteSessionRows(rows as unknown[], 'wife')

  assert.equal(rows[0].profile, 'wife')
  assert.equal(rows[0].is_default_profile, false)
  assert.equal(rows[1].profile, 'dad') // NOT relabeled to wife
  assert.equal(rows[1].is_default_profile, undefined)
  assert.equal(rows[2].profile, 'wife') // unowned row backfilled
  assert.equal(rows[2].is_default_profile, false)
  assert.equal(rows[3].profile, 'wife')
})

// A per-session read on a shared multi-profile backend must open the owning
// profile's state.db — the same scope the list read sends.
test('per-session remote reads carry the owner profile scope', () => {
  assert.equal(
    pathWithRemoteOwnerScope('/api/sessions/s-1?limit=50', 'wife'),
    '/api/sessions/s-1?limit=50&profile=wife'
  )
  assert.equal(pathWithRemoteOwnerScope('/api/sessions/s-1', 'wife'), '/api/sessions/s-1?profile=wife')
  // Existing pagination params survive the scope.
  assert.equal(
    pathWithRemoteOwnerScope('/api/sessions/s-1/messages?limit=100&offset=200', 'wife'),
    '/api/sessions/s-1/messages?limit=100&offset=200&profile=wife'
  )
  // A legacy single-profile scope keeps the path bare.
  assert.equal(pathWithRemoteOwnerScope('/api/sessions/s-1', ''), '/api/sessions/s-1')
})

// The scope helper itself: alias wins, every concrete scope is named.
test('remoteProfileQueryScope resolves the wire scope for a profile override', () => {
  assert.equal(remoteProfileQueryScope('wife'), 'wife')
  assert.equal(remoteProfileQueryScope('mara', 'dixie'), 'dixie')
  assert.equal(remoteProfileQueryScope('mara', 'default'), 'mara')
  assert.equal(remoteProfileQueryScope('default'), 'default')
  assert.equal(remoteProfileQueryScope('default', 'dixie'), 'dixie')
  assert.equal(remoteProfileQueryScope(''), '')
})
