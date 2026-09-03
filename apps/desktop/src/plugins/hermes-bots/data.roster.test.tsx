/**
 * The multi-source roster merge, driven through the real `useRoster` query.
 *
 * The union agent roster (`host.agents`) enumerates EVERY registered
 * connection — including the gateway that just answered `profiles.list`. The
 * merge exists to fold the active source's agents into the rich rows it
 * already has, and to append only the genuinely-other sources. Getting that
 * classification wrong is the whole bug family this suite pins:
 *
 *  - #88344 — a remote-primary desktop listed every bot twice, because the
 *    active gateway's own agents were appended as phantom rows;
 *  - the live-id override — after the user activates a non-primary source,
 *    `profiles.list` answers from THAT source, so classifying against the
 *    registry primary duplicated the active source's agents all over again;
 *  - #88828 + #88697 composed — a primary registered under two addresses
 *    collapses to one union row carrying the PRIMARY's connection id, and the
 *    boot descriptor now reports that same id, so those rows must annotate;
 *  - connect-on-demand — SSH sources drop off the union the moment their
 *    tunnel is not the live gateway, so clicking the local agent emptied Bot
 *    Mode until previously-painted rows were carried forward.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $lastRoster, useRoster } from './data'
import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    agents: vi.fn(),
    profileRoutes: undefined as unknown,
    request: vi.fn(),
    requestProfile: vi.fn(),
    retainProfileSocket: vi.fn(),
    state: { connectionId: { get: vi.fn(() => 'local') }, profile: { get: () => 'default' } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { useQuery } = await import('@tanstack/react-query')

  return {
    atom,
    host: hostMock,
    queryClient: { getQueryData: vi.fn(), invalidateQueries: vi.fn() },
    useQuery,
    useValue: (store: { get: () => unknown }) => store.get()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

interface UnionAgent {
  connectionId: string
  connectionKind: string
  connectionLabel?: string
  handle: string
  profile: string
}

interface Union {
  agents: UnionAgent[]
  primaryConnectionId?: string
  sources?: Array<{ connectionId: string; error?: string; kind: string; reachable?: boolean }>
}

/** Gateway rows carry a session id on `last_session` that the plugin's
 *  `SessionPreview` type deliberately does not model; the merge must carry it
 *  through untouched, so the fixtures keep it. */
interface RowFixture extends Omit<Partial<RosterRow>, 'last_session'> {
  last_session?: { id?: string; last_active?: number; preview?: string }
}

/** Run the real roster query once and hand back the merged rows. */
async function mergedRoster(
  local: { profiles: RowFixture[] },
  union: Union | null,
  liveConnectionId: null | string = 'local'
): Promise<RowFixture[]> {
  hostMock.state.connectionId.get.mockReturnValue(liveConnectionId as string)
  hostMock.request.mockResolvedValue(local)

  if (union) {
    hostMock.agents.mockResolvedValue(union)
  } else {
    hostMock.agents.mockRejectedValue(new Error('no union roster on this build'))
  }

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  const { result } = renderHook(() => useRoster(), { wrapper })

  await waitFor(() => expect(result.current.data).toBeTruthy())

  return (result.current.data?.profiles ?? []) as RowFixture[]
}

const identities = (rows: RowFixture[]) => rows.map(row => `${row.connectionId}:${row.name}`)

beforeEach(() => {
  vi.clearAllMocks()
  $lastRoster.set([])
})

afterEach(() => {
  $lastRoster.set([])
})

describe('no union roster', () => {
  it('retains the polled profile socket for the query observer lifetime', async () => {
    const release = vi.fn()
    hostMock.retainProfileSocket.mockReturnValueOnce(release)
    hostMock.request.mockResolvedValue({ profiles: [] })
    hostMock.agents.mockResolvedValue({ agents: [], sources: [] })

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    const { result, unmount } = renderHook(() => useRoster(), { wrapper })

    await waitFor(() => expect(result.current.data).toBeTruthy())
    expect(hostMock.retainProfileSocket).toHaveBeenCalledWith('default')

    unmount()
    expect(release).toHaveBeenCalledOnce()
  })

  it('leaves the local list exactly as it was', async () => {
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 's1', last_active: 1 }, name: 'default' }] },
      null
    )

    expect(rows).toHaveLength(1)
    expect(rows[0].name).toBe('default')
    expect(rows[0].last_session?.id).toBe('s1')
  })

  it('appends nothing when the union is empty', async () => {
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 's1', last_active: 1 }, name: 'default' }] },
      { agents: [] }
    )

    expect(identities(rows)).toEqual(['undefined:default'])
  })
})

describe('the active source annotates; other sources append', () => {
  it('keeps rich fields on the annotated row and tags the rest by source', async () => {
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 's1', last_active: 1 }, name: 'research' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'research-this-device',
            profile: 'research'
          },
          {
            connectionId: 'homelab',
            connectionKind: 'remote',
            connectionLabel: 'Homelab',
            handle: 'research-homelab',
            profile: 'research'
          },
          {
            connectionId: 'homelab',
            connectionKind: 'remote',
            connectionLabel: 'Homelab',
            handle: 'coder',
            profile: 'coder'
          }
        ]
      }
    )

    expect(rows).toHaveLength(3)

    const annotated = rows.find(row => row.name === 'research' && !row.remoteSource)!

    expect(annotated.last_session?.id).toBe('s1')
    expect(annotated.handle).toBe('research-this-device')
    expect(annotated.sourceScoped).toBe(true)
    expect(annotated.remoteSource).toBeUndefined()

    expect(rows.find(row => row.name === 'research' && row.remoteSource)).toMatchObject({
      connectionId: 'homelab',
      connectionLabel: 'Homelab',
      handle: 'research-homelab'
    })
    expect(rows.find(row => row.name === 'coder')).toMatchObject({ handle: 'coder', remoteSource: true })
  })

  it('never invents a thin row for an active-source profile profiles.list did not return', async () => {
    // An older backend mid-refresh: skip rather than paint a bot that has no
    // rich row behind it.
    const rows = await mergedRoster(
      { profiles: [{ name: 'default' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'ghost',
            profile: 'ghost'
          }
        ]
      }
    )

    expect(rows.map(row => row.name)).toEqual(['default'])
  })

  it('renders duplicate source and local identities once', async () => {
    const rows = await mergedRoster(
      {
        profiles: [
          { last_session: { id: 'newest', last_active: 2 }, name: 'default' },
          { last_session: { id: 'stale', last_active: 1 }, name: 'default' }
        ]
      },
      {
        agents: [
          { connectionId: 'local', connectionKind: 'local', handle: 'default-this-device', profile: 'default' },
          { connectionId: 'local', connectionKind: 'local', handle: 'default-this-device', profile: 'default' },
          {
            connectionId: 'homelab',
            connectionKind: 'remote',
            connectionLabel: 'Homelab',
            handle: 'default-homelab',
            profile: 'default'
          },
          {
            connectionId: 'homelab',
            connectionKind: 'remote',
            connectionLabel: 'Homelab',
            handle: 'default-homelab',
            profile: 'default'
          }
        ]
      }
    )

    expect(rows).toHaveLength(2)
    expect(rows[0].last_session?.id).toBe('newest')
    expect(rows[0].handle).toBe('default-this-device')
    expect(rows[1].connectionId).toBe('homelab')
  })

  it('follows the ACTIVE remote source, so the local twin is the appended one', async () => {
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 'remote-session', last_active: 1 }, name: 'default' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          },
          {
            connectionId: 'work',
            connectionKind: 'remote',
            connectionLabel: 'Work',
            handle: 'default-work',
            profile: 'default'
          }
        ]
      },
      'work'
    )

    const active = rows.find(row => row.connectionId === 'work')!

    expect(active.remoteSource).toBeUndefined()
    expect(active.sourceScoped).toBe(true)
    expect(active.last_session?.id).toBe('remote-session')
    expect(rows.find(row => row.connectionId === 'local')).toMatchObject({
      remoteSource: true,
      sourceScoped: true
    })
  })

  it('annotates every active-gateway agent instead of duplicating them (#88344)', async () => {
    const primary = '10-244-108-128-9119'

    const rows = await mergedRoster(
      {
        profiles: [
          { last_session: { id: 's-default', last_active: 1 }, name: 'default' },
          { last_session: { id: 's-dev', last_active: 1 }, name: 'dev' }
        ]
      },
      {
        agents: [
          // The ACTIVE remote gateway itself — same identities as the rich
          // rows, so it must annotate in place.
          {
            connectionId: primary,
            connectionKind: 'remote',
            connectionLabel: '10.244.108.128:9119',
            handle: `default-${primary}`,
            profile: 'default'
          },
          {
            connectionId: primary,
            connectionKind: 'remote',
            connectionLabel: '10.244.108.128:9119',
            handle: 'dev',
            profile: 'dev'
          },
          // A genuinely separate source with a same-named profile keeps its row.
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          },
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'agent-mentor',
            profile: 'agent-mentor'
          }
        ],
        primaryConnectionId: primary
      },
      primary
    )

    // 2 annotated rows + 2 other-source rows = 4, NOT 6.
    expect(rows).toHaveLength(4)
    expect(rows.filter(row => row.remoteSource)).toHaveLength(2)

    const annotated = rows.find(row => row.name === 'default' && !row.remoteSource)!

    expect(annotated.last_session?.id).toBe('s-default')
    expect(annotated.handle).toBe(`default-${primary}`)
    expect(rows.find(row => row.name === 'default' && row.remoteSource)).toMatchObject({
      connectionId: 'local',
      handle: 'default-this-device'
    })
  })

  it('keeps single-source behavior on a primary-local desktop', async () => {
    const rows = await mergedRoster(
      { profiles: [{ name: 'default' }, { name: 'dev' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          },
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'dev',
            profile: 'dev'
          }
        ],
        primaryConnectionId: 'local'
      }
    )

    expect(rows).toHaveLength(2)
    expect(rows.filter(row => row.remoteSource)).toHaveLength(0)
    expect(rows[0].handle).toBe('default-this-device')
  })

  it('lets the live id beat primaryConnectionId', async () => {
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 's1', last_active: 1 }, name: 'default' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          },
          {
            connectionId: 'vps',
            connectionKind: 'remote',
            connectionLabel: 'VPS',
            handle: 'default-vps',
            profile: 'default'
          }
        ],
        primaryConnectionId: 'local'
      },
      'vps'
    )

    expect(rows).toHaveLength(2)
    expect(rows.find(row => row.last_session)?.handle).toBe('default-vps')
    expect(rows.filter(row => row.remoteSource).map(row => row.connectionId)).toEqual(['local'])
  })

  it('composes the install_id collapse with the live id, with no re-append (#88828 + #88697)', async () => {
    const rows = await mergedRoster(
      {
        profiles: [
          { last_session: { id: 's-default', last_active: 1 }, name: 'default' },
          { last_session: { id: 's-dev', last_active: 1 }, name: 'dev' }
        ]
      },
      {
        agents: [
          // Post-#88828 union: the tailscale twin of the primary collapsed
          // into these rows — one per profile, keyed to the PRIMARY id.
          {
            connectionId: 'spark-lan',
            connectionKind: 'remote',
            connectionLabel: 'Spark',
            handle: 'default-spark',
            profile: 'default'
          },
          {
            connectionId: 'spark-lan',
            connectionKind: 'remote',
            connectionLabel: 'Spark',
            handle: 'dev',
            profile: 'dev'
          },
          // A real second backend survives the collapse as its own row.
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          }
        ],
        primaryConnectionId: 'spark-lan'
      },
      'spark-lan'
    )

    // Pre-fix (live id null) this was 5: both primary rows re-appended.
    expect(rows).toHaveLength(3)
    expect(rows.filter(row => row.remoteSource).map(row => row.connectionId)).toEqual(['local'])
    expect(rows.find(row => row.name === 'default' && !row.remoteSource)).toMatchObject({
      connectionId: 'spark-lan',
      handle: 'default-spark'
    })
  })
})

describe('a null live id', () => {
  it('does not treat the registry primary as active on a local window', async () => {
    // Clicking the local agent leaves host.state.connectionId null while the
    // registry primary stays on Spark. That must not skip Spark's bots or
    // invent a second "This device" shadow of default.
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 'this-chat', last_active: 1 }, name: 'default' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'default-this-device',
            profile: 'default'
          },
          { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', handle: 'bob', profile: 'bob' },
          { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', handle: 'kai', profile: 'kai' },
          { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', handle: 'rook', profile: 'rook' }
        ],
        primaryConnectionId: 'spark'
      },
      null
    )

    expect(rows.filter(row => row.name === 'default')).toHaveLength(1)
    expect(rows.find(row => row.name === 'default')?.last_session?.id).toBe('this-chat')
    expect(rows.filter(row => row.remoteSource && row.connectionId === 'local')).toHaveLength(0)
    expect(
      rows
        .filter(row => row.remoteSource)
        .map(row => row.name)
        .sort()
    ).toEqual(['bob', 'kai', 'rook'])
  })

  it('infers a matching remote primary when the local inventory does not match', async () => {
    // Legacy remote descriptors carry mode:'remote' but no connectionId, so
    // the host state reads null even though profiles.list is answering from
    // the registry primary.
    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 'noah-chat', last_active: 1 }, name: 'default' }] },
      {
        agents: [
          {
            connectionId: 'local',
            connectionKind: 'local',
            connectionLabel: 'This device',
            handle: 'archie',
            profile: 'archie'
          },
          {
            connectionId: 'noah',
            connectionKind: 'remote',
            connectionLabel: 'Noah',
            handle: 'default',
            profile: 'default'
          }
        ],
        primaryConnectionId: 'noah'
      },
      null
    )

    expect(rows).toHaveLength(2)

    const inferred = rows.find(row => row.name === 'default')!

    expect(inferred.connectionId).toBe('noah')
    expect(inferred.remoteSource).toBeUndefined()
    expect(rows.find(row => row.name === 'archie')).toMatchObject({ connectionId: 'local', remoteSource: true })
  })
})

describe('connect-on-demand sources', () => {
  const previouslyPainted = (connectionId: string) =>
    [
      { last_session: { id: 'this-chat', last_active: 1 }, name: 'default' },
      {
        connectionId,
        connectionKind: 'ssh',
        connectionLabel: 'Spark',
        handle: 'bob',
        name: 'bob',
        remoteSource: true,
        sourceScoped: true
      }
    ] as RosterRow[]

  const localOnlyUnion = (sources: Union['sources']): Union => ({
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        handle: 'default',
        profile: 'default'
      }
    ],
    primaryConnectionId: 'local',
    sources
  })

  it('carries a previously painted remote row through an empty union', async () => {
    $lastRoster.set(previouslyPainted('spark'))

    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 'this-chat', last_active: 1 }, name: 'default' }] },
      localOnlyUnion([{ connectionId: 'spark', error: 'connect-on-demand', kind: 'ssh' }])
    )

    expect(rows.find(row => row.name === 'bob' && row.connectionId === 'spark')).toMatchObject({ remoteSource: true })
    expect(rows.filter(row => row.name === 'default')).toHaveLength(1)
  })

  it('does not resurrect a row whose connection was removed from the registry', async () => {
    $lastRoster.set(previouslyPainted('gone'))

    const rows = await mergedRoster(
      { profiles: [{ last_session: { id: 'this-chat', last_active: 1 }, name: 'default' }] },
      localOnlyUnion([{ connectionId: 'local', kind: 'local' }])
    )

    expect(rows.find(row => row.connectionId === 'gone')).toBeUndefined()
  })
})

describe('a stalled profiles.list cannot pin the spinner forever', () => {
  it('gives up after bounded retries and surfaces the error', async () => {
    // `retry: true` keeps React Query in isLoading until the first success, so
    // a stalled profiles.list (live state.db write lock, SSH flap) left the
    // Bots sidebar on a spinner with no error card. The 5s refetchInterval and
    // the gateway-open effect already recover drops.
    hostMock.state.connectionId.get.mockReturnValue('local')
    hostMock.request.mockRejectedValue(new Error('state.db is locked'))

    const client = new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } })

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(() => useRoster(), { wrapper })

    await waitFor(() => expect(result.current.isError).toBe(true), { timeout: 5000 })

    expect(result.current.isLoading).toBe(false)
    expect(hostMock.request.mock.calls.length).toBeGreaterThan(1)
  })
})
