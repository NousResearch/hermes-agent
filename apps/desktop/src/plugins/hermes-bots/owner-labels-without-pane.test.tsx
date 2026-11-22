/**
 * A bot chat's tab captions with its bot's name, not the shared "Bot Chat"
 * registry title (#99152) — including tiles restored at boot while the Bots
 * pane has never mounted. The pane strip renders background tabs lazily, so
 * after a reload on the Sessions sidebar tab `BotsPane` (and every effect in
 * it) does not exist; the roster is still fetched without it (`primeRoster`
 * on gateway open, the composer's `useRoster`). The labels must come from
 * that fetch, and the stored title must stay exactly "Bot Chat".
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The REAL shell caption the tab strip, tile and window title all read.
// eslint-disable-next-line no-restricted-imports
import {
  $workspaceOwnerLabels,
  setWorkspaceOwnerLabel,
  workspaceOwnerTitle
} from '@/components/pane-shell/workspace-scope'

import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    agents: vi.fn(),
    profileRoutes: undefined as unknown,
    request: vi.fn(),
    requestProfile: vi.fn(),
    setWorkspaceOwnerLabel: undefined as unknown,
    state: { connectionId: { get: vi.fn(() => 'local') }, profile: { get: () => 'default' } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')
  const { QueryClient: Client, useQuery } = await import('@tanstack/react-query')

  return {
    atom,
    host: hostMock,
    queryClient: new Client({ defaultOptions: { queries: { retry: false } } }),
    useQuery,
    useValue: (store: { get: () => unknown }) => store.get()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

import { queryClient } from '@hermes/plugin-sdk'

import { $botMeta, $lastRoster, primeRoster, useRoster } from './data'

function bot(name: string, title: string): RosterRow {
  return { name, ui_meta: { 'hermes-bots': { title } } } as RosterRow
}

/** The union agent roster a single-machine desktop reports: every profile on
 *  the local connection. It is what turns a row's owner key into the
 *  `bot:local::<profile>` the persisted tiles carry. */
function localUnion(profiles: string[]) {
  return {
    agents: profiles.map(profile => ({
      connectionId: 'local',
      connectionKind: 'local',
      connectionLabel: 'This device',
      handle: profile,
      profile
    })),
    primaryConnectionId: 'local',
    sources: [{ connectionId: 'local', kind: 'local', reachable: true }]
  }
}

/** Exactly what `hermes.desktop.sessionTiles.v2` restores for a canonical chat. */
function restoredTile(ownerKey: string) {
  return { workspaceMode: 'bots' as const, workspaceOwnerKey: ownerKey, workspaceTabTitle: 'Bot Chat' }
}

beforeEach(() => {
  vi.clearAllMocks()
  queryClient.clear()
  $lastRoster.set([])
  $botMeta.set({})
  $workspaceOwnerLabels.set({})
  // The real host seam: the SDK's setWorkspaceOwnerLabel is the shell's.
  hostMock.setWorkspaceOwnerLabel = setWorkspaceOwnerLabel
  hostMock.agents.mockResolvedValue(localUnion(['default', 'tranquillity-chief']))
})

describe('bot tab captions without the Bots pane', () => {
  it('captions restored bot tiles with the bot name once the roster is fetched', async () => {
    hostMock.request.mockResolvedValue({
      profiles: [bot('default', 'Hermes Prime'), bot('tranquillity-chief', 'TQ Chief')]
    })

    // Before any roster: the placeholder, as today.
    expect(workspaceOwnerTitle('Bot Chat', restoredTile('bot:local::tranquillity-chief'))).toBe('Bot Chat')

    // The boot prime that runs on gateway open; BotsPane never mounts.
    await primeRoster()

    expect(workspaceOwnerTitle('Bot Chat', restoredTile('bot:local::default'))).toBe('Hermes Prime')
    expect(workspaceOwnerTitle('Bot Chat', restoredTile('bot:local::tranquillity-chief'))).toBe('TQ Chief')
    // Presentation only: a side thread in the same workspace keeps its title,
    // and a Sessions tab titled "Bot Chat" is untouched.
    expect(workspaceOwnerTitle('Plan the launch', restoredTile('bot:local::default'))).toBe('Plan the launch')
    expect(workspaceOwnerTitle('Bot Chat', { workspaceMode: 'sessions' })).toBe('Bot Chat')
  })

  it('never sends a rename for the canonical chat while captioning it', async () => {
    hostMock.request.mockResolvedValue({ profiles: [bot('default', 'Hermes Prime')] })

    await primeRoster()

    const methods = hostMock.request.mock.calls.map(call => call[0])
    expect(methods).toEqual(['profiles.list'])
    expect(hostMock.requestProfile).not.toHaveBeenCalled()
  })

  it('a cancelled, older roster fetch does not overwrite a newer bot name', async () => {
    const deferred: Array<(value: unknown) => void> = []

    hostMock.request.mockImplementation(
      () =>
        new Promise(resolve => {
          deferred.push(resolve)
        })
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    const { result } = renderHook(() => useRoster(), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      )
    })

    await waitFor(() => expect(deferred).toHaveLength(1))
    deferred[0]!({ profiles: [bot('tranquillity-chief', 'TQ Chief')] })
    await waitFor(() => expect(result.current.data?.profiles).toHaveLength(1))

    // One poll goes out and stalls; a newer one supersedes it. React Query
    // cancels the stalled fetch for its observers, but that query function
    // still runs to completion — and with it any side effect it carries.
    await new Promise(resolve => setTimeout(resolve, 5))
    void result.current.refetch()
    await waitFor(() => expect(deferred).toHaveLength(2))
    await new Promise(resolve => setTimeout(resolve, 5))
    void result.current.refetch({ cancelRefetch: true })
    await waitFor(() => expect(deferred).toHaveLength(3))

    deferred[2]!({ profiles: [bot('tranquillity-chief', 'TQ Chief (renamed)')] })
    await waitFor(() =>
      expect(workspaceOwnerTitle('Bot Chat', restoredTile('bot:local::tranquillity-chief'))).toBe('TQ Chief (renamed)')
    )

    deferred[1]!({ profiles: [bot('tranquillity-chief', 'TQ Chief')] })
    await new Promise(resolve => setTimeout(resolve, 20))

    expect(workspaceOwnerTitle('Bot Chat', restoredTile('bot:local::tranquillity-chief'))).toBe('TQ Chief (renamed)')
  })
})
