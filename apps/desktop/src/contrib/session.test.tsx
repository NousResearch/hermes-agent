import { act, cleanup, render, screen } from '@testing-library/react'
import { createElement, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeSessionId, $selectedStoredSessionId, $sessions, _resetSessionOwnerHintsForTests } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { registry } from './registry'
import * as sessionApi from './session'
import { resolveSessionContributionContext } from './session-context'
import { SessionContributions } from './session-contributions'

it('resolves the addressed tile or row, not the focused runtime, and fails closed on ambiguous ids', () => {
  expect(resolveSessionContributionContext).toBeTypeOf('function')
  _resetSessionOwnerHintsForTests()
  $activeSessionId.set('runtime-a')
  $selectedStoredSessionId.set('stored-a')
  $sessions.set([
    { id: 'stored-a', profile: 'alice', connection_id: 'local' },
    { id: 'stored-b', profile: 'bob', connection_id: 'remote' }
  ] as SessionInfo[])
  $sessionTiles.set([
    { storedSessionId: 'stored-b', runtimeId: 'runtime-b', ownerRoute: { connectionId: 'remote', profile: 'bob' } }
  ])
  expect(resolveSessionContributionContext({ runtimeSessionId: 'runtime-b' })).toMatchObject({
    runtimeSessionId: 'runtime-b',
    storedSessionId: 'stored-b',
    profile: 'bob',
    connectionId: 'remote'
  })
  expect(resolveSessionContributionContext({ storedSessionId: 'missing' })).toBeNull()
  $sessions.set([
    { id: 'collision', profile: 'alice', connection_id: 'local' },
    { id: 'collision', profile: 'alice', connection_id: 'remote' }
  ] as SessionInfo[])
  expect(resolveSessionContributionContext({ storedSessionId: 'collision' })).toBeNull()
  expect(resolveSessionContributionContext({ storedSessionId: 'collision', row: $sessions.get()[1] })).toMatchObject({
    runtimeSessionId: null,
    storedSessionId: 'collision',
    profile: 'alice',
    connectionId: 'remote'
  })
})

it('does not repaint a session contribution on unrelated streamed state changes', async () => {
  const renderBadge = vi.fn(() => <span>badge</span>)
  disposers.push(
    registry.register({ id: 'perf', area: sessionApi.SESSION_AREAS.tileBadge, data: { render: renderBadge } })
  )
  $sessionTiles.set([
    { storedSessionId: 'own', runtimeId: 'own-runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  render(<SessionContributions area={sessionApi.SESSION_AREAS.tileBadge} storedSessionId="own" />)
  expect(renderBadge).toHaveBeenCalledOnce()
  renderBadge.mockClear()
  await act(async () => {
    $sessionStates.set({ unrelated: createClientSessionState('unrelated') })
    await new Promise(resolve => setTimeout(resolve, 0))
  })
  expect(renderBadge).not.toHaveBeenCalled()
})

const disposers: Array<() => void> = []
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

it('renders independent session contributions with hooks and removes them on disposal', () => {
  expect(sessionApi.SESSION_AREAS).toBeDefined()
  const area = sessionApi.SESSION_AREAS.listBadge
  const first = { runtimeSessionId: 'runtime-a', storedSessionId: 'stored-a', profile: 'alice', connectionId: 'local' }
  const second = { runtimeSessionId: null, storedSessionId: 'stored-b', profile: 'bob', connectionId: 'remote' }

  const dispose = registry.register({
    id: 'demo',
    area,
    data: {
      render: function SessionBadge({ session }: sessionApi.SessionContributionProps) {
        const [label] = useState(session.profile)

        return createElement('span', {}, `${label}:${session.connectionId}:${session.storedSessionId}`)
      }
    }
  })

  disposers.push(dispose)
  render(
    <>
      <sessionApi.SessionContributionSlot area={area} session={first} />
      <sessionApi.SessionContributionSlot area={area} session={second} />
    </>
  )
  expect(screen.getByText('alice:local:stored-a')).toBeTruthy()
  expect(screen.getByText('bob:remote:stored-b')).toBeTruthy()
  act(dispose)
  expect(screen.queryByText('alice:local:stored-a')).toBeNull()
})
