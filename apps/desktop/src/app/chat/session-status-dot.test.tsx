import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $interfaceMode } from '@/store/interface-mode'
import { $sessions, $unreadFinishedSessionIds } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { SessionStatusDot } from './session-status-dot'

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessions.set([])
  $unreadFinishedSessionIds.set([])
  $interfaceMode.set('advanced')
})

it('reserves Simple dots for unread messages from the backend or the live runtime', () => {
  $interfaceMode.set('simple')
  const view = render(<SessionStatusDot storedSessionId="chat" />)
  expect(view.container.querySelector('.rounded-full')).toBeNull()
  act(() => $sessions.set([makeSessionInfo({ id: 'chat', unread: true })]))
  expect(screen.getByRole('status').className).toContain('rounded-full')
  act(() => $sessions.set([makeSessionInfo({ id: 'chat', unread: false })]))
  expect(screen.queryByRole('status')).toBeNull()
  act(() => $unreadFinishedSessionIds.set(['chat']))
  expect(screen.getByRole('status').className).toContain('rounded-full')
  act(() => $unreadFinishedSessionIds.set([]))
  expect(screen.queryByRole('status')).toBeNull()
  view.rerender(<SessionStatusDot storedSessionId={null} />)
  expect(view.container.querySelector('.rounded-full')).toBeNull()
})

it('keeps running activity visible as an icon and preserves Advanced idle dots', () => {
  $interfaceMode.set('simple')
  const view = render(<SessionStatusDot storedSessionId="chat" />)
  act(() => publishSessionState('runtime', { ...createClientSessionState('chat'), busy: true }))
  expect(screen.getByRole('status').querySelector('.codicon-loading')).toBeTruthy()
  expect(view.container.querySelector('.rounded-full')).toBeNull()
  act(() => clearAllSessionStates())
  act(() => $interfaceMode.set('advanced'))
  expect(view.container.querySelector('.rounded-full')).toBeTruthy()
})

it('keeps a request for user input visible without an unread dot', () => {
  $interfaceMode.set('simple')
  const view = render(<SessionStatusDot storedSessionId="chat" />)
  act(() => publishSessionState('runtime', { ...createClientSessionState('chat'), busy: true, needsInput: true }))
  expect(screen.getByRole('status').querySelector('.codicon-warning')).toBeTruthy()
  expect(view.container.querySelector('.rounded-full')).toBeNull()
})
