import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { StatusItemRow } from '@/app/chat/composer/status-stack/status-row'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $backgroundStatusBySession,
  dismissBackgroundProcess,
  reconcileBackgroundProcesses,
  resetSessionBackground
} from '@/store/composer-status'
import { $sidebarRowMeta, resetSidebarView } from '@/store/layout'
import { $sessions } from '@/store/session'
import { $sessionDotStateById, showsRunningArc } from '@/store/session-dot-state'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'
import { makeSessionInfo } from '@/test/session-info'

import { SidebarSessionActivity } from './session-activity'

const child: SubagentProgress = {
  id: 'child',
  parentId: null,
  goal: 'Fix the build',
  status: 'running',
  taskCount: 1,
  taskIndex: 0,
  startedAt: 1,
  updatedAt: 1,
  filesRead: [],
  filesWritten: [],
  stream: []
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessions.set([])
  $backgroundStatusBySession.set({})
  resetSessionBackground('runtime')
  $subagentsBySession.set({})
  resetSidebarView()
})

it('reserves monitoring for background-only work and returns to working for delegated or parent follow-up', () => {
  $sidebarRowMeta.set(['activity'])
  $sessions.set([makeSessionInfo({ id: 'stored', message_count: 2 })])
  const idle = createClientSessionState('stored')

  const process = {
    id: 'watch',
    type: 'background' as const,
    state: 'running' as const,
    title: '# Watching CI',
    awaitingNotification: true
  }

  publishSessionState('runtime', { ...idle, busy: true })
  $backgroundStatusBySession.set({ runtime: [process] })
  render(<SidebarSessionActivity sessionId="stored" />)
  const status = () => $sessionDotStateById.get().stored ?? 'idle'

  expect(status()).toBe('working')
  expect(screen.queryByRole('img')).toBeNull()
  act(() => publishSessionState('runtime', idle))
  expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
  expect(showsRunningArc(status())).toBe(false)

  act(() => $subagentsBySession.set({ runtime: [child] }))
  expect(status()).toBe('working')
  expect(showsRunningArc(status())).toBe(true)
  expect(screen.queryByRole('img')).toBeNull()
  act(() => $subagentsBySession.set({ runtime: [{ ...child, status: 'completed' }] }))
  expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()

  act(() => publishSessionState('runtime', { ...idle, busy: true }))
  expect(screen.queryByRole('img')).toBeNull()
  act(() => $backgroundStatusBySession.set({ runtime: [{ ...process, state: 'done' }] }))
  expect(status()).toBe('working')
  act(() => publishSessionState('runtime', idle))
  expect(status()).toBe('unread')
  expect(screen.queryByRole('img')).toBeNull()
})

it('uses notification contracts rather than process names to distinguish awaited work from a leftover server', () => {
  $sidebarRowMeta.set(['activity'])
  $sessions.set([makeSessionInfo({ id: 'stored', message_count: 2 })])
  publishSessionState('runtime', createClientSessionState('stored'))
  const process = { session_id: 'process', command: '# Watching CI', status: 'running' }
  reconcileBackgroundProcesses('runtime', [process])
  render(<SidebarSessionActivity sessionId="stored" />)
  expect(screen.queryByRole('img')).toBeNull()
  expect($sessionDotStateById.get().stored).not.toBe('background')

  act(() => reconcileBackgroundProcesses('runtime', [{ ...process, notify_on_complete: true }]))
  expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
  act(() => reconcileBackgroundProcesses('runtime', [{ ...process, watch_patterns: ['ready'], watch_hit: false }]))
  expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()
  act(() => reconcileBackgroundProcesses('runtime', [{ ...process, watch_patterns: ['ready'], watch_hit: true }]))
  expect(screen.queryByRole('img')).toBeNull()
  expect($backgroundStatusBySession.get().runtime?.[0]?.state).toBe('running')
  act(() =>
    reconcileBackgroundProcesses('runtime', [
      { ...process, status: 'exited', exit_code: 0, notification_pending: true }
    ])
  )
  expect(screen.getByRole('img', { name: 'Watching CI' })).toBeTruthy()

  const statusRow = () => (
    <StatusItemRow
      item={$backgroundStatusBySession.get().runtime![0]!}
      onDismiss={id => dismissBackgroundProcess('runtime', id)}
    />
  )

  const row = render(statusRow())
  expect(screen.queryByRole('button', { name: 'Dismiss' })).toBeNull()
  act(() =>
    reconcileBackgroundProcesses('runtime', [
      { ...process, status: 'exited', exit_code: 0, notification_pending: false }
    ])
  )
  row.rerender(statusRow())
  expect(screen.queryByRole('img')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
  expect($backgroundStatusBySession.get().runtime ?? []).toEqual([])
})
