import { afterEach, expect, it, vi } from 'vitest'

import { markFirstBuildSession } from '@/app/contrib/handoff-receipt'
import { readKey } from '@/lib/storage'
import type { ConnectorFlowRow } from '@/store/connector-flow'
import { $firstBuildConnections, startFirstBuild } from '@/store/first-build-connectors'

import { buildConnectionStartMessage, canStartWithConnections } from './first-build-start'

const rows: ConnectorFlowRow[] = [
  { connector: 'gmail', phase: 'connected' },
  { connector: 'googlecalendar', phase: 'waiting' },
  { connector: 'notion', phase: 'timeout' }
]

afterEach(() => {
  window.localStorage.clear()
  $firstBuildConnections.set({})
})

it('names connected and skipped apps in first person, including all and none connected', () => {
  expect(buildConnectionStartMessage(rows)).toBe('Start with Gmail connected. I skipped Google Calendar and Notion.')
  expect(buildConnectionStartMessage(rows.map(row => ({ ...row, phase: 'connected' })))).toBe(
    'Start with Gmail, Google Calendar and Notion connected.'
  )
  expect(buildConnectionStartMessage(rows.slice(0, 2).map(row => ({ ...row, phase: 'waiting' })))).toBe(
    'Start without connections. I skipped Gmail and Google Calendar.'
  )
})

it('allows a pending wait to start once and retains that decision after rehydration', () => {
  const part = {
    type: 'tool-call' as const,
    toolCallId: 'wait',
    toolName: 'manage_connections',
    args: { action: 'wait' }
  }

  expect(canStartWithConnections(part)).toBe(true)
  expect(canStartWithConnections({ ...part, result: { status: 'pending' } })).toBe(true)

  for (const status of ['timeout', 'connected', 'interrupted']) {
    expect(canStartWithConnections({ ...part, result: { status } })).toBe(false)
  }

  markFirstBuildSession('build')
  const state = { toolCallId: part.toolCallId, rows, started: false }
  $firstBuildConnections.setKey('build', state)
  const submit = vi.fn().mockReturnValue(true)
  startFirstBuild('build', submit)
  expect(submit.mock.calls).toEqual([[buildConnectionStartMessage(rows)]])
  expect($firstBuildConnections.get().build.started).toBe(true)
  expect(readKey('hermes.onboarding.started.v1.build')).toBe('1')

  $firstBuildConnections.setKey('build', state)
  startFirstBuild('build', submit)
  expect(submit).toHaveBeenCalledTimes(1)
})
