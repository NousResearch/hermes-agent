import { afterEach, expect, it, vi } from 'vitest'

import { markFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { readKey } from '@/lib/storage'

import {
  $firstBuildConnections,
  type FirstBuildConnectorPart,
  flushFirstBuildNote,
  openFirstBuildLinks
} from './first-build-connectors'

const part: FirstBuildConnectorPart = {
  toolCallId: 'connect-batch',
  toolName: 'manage_connections',
  args: { action: 'connect', connectors: ['gmail', 'googlecalendar', 'notion'] },
  result: {
    results: [
      { connector: 'gmail', status: 'initiated', connect_url: 'https://connect.example/gmail' },
      { connector: 'googlecalendar', status: 'initiated', connect_url: 'https://connect.example/calendar' },
      { connector: 'notion', status: 'active' }
    ]
  }
}

afterEach(() => {
  window.localStorage.clear()
  $firstBuildConnections.set({})
})

it('opens the entire first-build batch once and reports its links in result order', async () => {
  markFirstBuildSession('build')

  const open = vi.fn(async () => {
    expect(readKey(`hermes.onboarding.links-opened.v1.${part.toolCallId}`)).toBe('1')
  })

  const opening = openFirstBuildLinks('build', part, { open })

  expect(open.mock.calls).toHaveLength(2)
  await opening
  expect(open).toHaveBeenNthCalledWith(1, 'https://connect.example/gmail')
  expect(open).toHaveBeenNthCalledWith(2, 'https://connect.example/calendar')
  expect($firstBuildConnections.get().build.pendingNote).toBe('[setup] links opened for gmail, googlecalendar')
  expect($firstBuildConnections.get().build.rows.map(row => row.phase)).toEqual(['waiting', 'waiting', 'connected'])

  // The note waits for an idle session, then goes once.
  const submit = vi.fn().mockReturnValue(true)
  flushFirstBuildNote('build', part.toolCallId, true, submit)
  expect(submit).not.toHaveBeenCalled()
  flushFirstBuildNote('build', part.toolCallId, false, submit)
  flushFirstBuildNote('build', part.toolCallId, false, submit)
  expect(submit.mock.calls).toEqual([['[setup] links opened for gmail, googlecalendar']])
  expect($firstBuildConnections.get().build.pendingNote).toBeUndefined()

  // Rehydration loses renderer state but retains the durable open receipt.
  $firstBuildConnections.set({})
  await openFirstBuildLinks('build', part, { open })
  expect(open).toHaveBeenCalledTimes(2)
  expect($firstBuildConnections.get().build.pendingNote).toBeUndefined()

  const state = $firstBuildConnections.get().build
  $firstBuildConnections.setKey('build', {
    ...state,
    rows: state.rows.map(row => ({ ...row, phase: 'connected' }))
  })
  await openFirstBuildLinks(
    'build',
    { ...part, toolCallId: 'reconnect-batch', args: { ...part.args, action: 'reconnect' } },
    { open: vi.fn(async () => {}) }
  )
  expect($firstBuildConnections.get().build.rows.map(row => row.phase)).toEqual(['waiting', 'waiting', 'connected'])
})

it('drops the held note when the model has already moved past the connect', async () => {
  markFirstBuildSession('build')
  await openFirstBuildLinks('build', part, { open: vi.fn(async () => {}) })
  const submit = vi.fn().mockReturnValue(true)

  flushFirstBuildNote('build', 'wait-call', false, submit)
  expect(submit).not.toHaveBeenCalled()
  expect($firstBuildConnections.get().build.pendingNote).toBeUndefined()
})

it('keeps ordinary sessions click-driven and retains links when the browser bridge is absent', async () => {
  markFirstBuildSession('build')
  const open = vi.fn(async () => {})
  await openFirstBuildLinks('ordinary', part, { open })
  expect(open).not.toHaveBeenCalled()
  expect($firstBuildConnections.get().ordinary).toBeUndefined()

  await openFirstBuildLinks('build', part, {})
  expect($firstBuildConnections.get().build.pendingNote).toBeUndefined()
  expect(readKey(`hermes.onboarding.links-opened.v1.${part.toolCallId}`)).toBeNull()
  expect($firstBuildConnections.get().build.rows[0].connectUrl).toBe('https://connect.example/gmail')
})
