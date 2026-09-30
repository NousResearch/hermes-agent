import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { $desktopNotifyIncomingMessages } from './desktop-notify-incoming'
import { resetInboundMessageCountsForTests } from './messaging-inbound-notify'
import {
  dispatchNativeNotification,
  NATIVE_NOTIFICATION_KINDS,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from './native-notifications'
import { __resetNativeNotifyBaselineForTests } from './notify-baseline'
import { $messagingSessions, setActiveSessionId } from './session'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notify = vi.fn().mockResolvedValue(true)

function setWindowState({ focused = true, hidden = false }: { focused?: boolean; hidden?: boolean }) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => focused })
}

let counter = 0

// Unique session id per test dodges the per-(kind,session) throttle so each
// assertion starts clean (same trick as native-notifications.test.ts).
function freshMessagingSession(): string {
  counter += 1

  return `tg-${counter}`
}

const session = (id: string, over: Partial<SessionInfo> = {}): SessionInfo =>
  makeSessionInfo({ id, source: 'telegram', ...over })

beforeEach(() => {
  notify.mockClear()
  desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
  setNativeNotifyEnabled(true)
  setNativeNotifyKind('message', true)
  $desktopNotifyIncomingMessages.set(true)
  setActiveSessionId(null)
  setWindowState({ focused: false, hidden: true })
  __resetNativeNotifyBaselineForTests()
  resetInboundMessageCountsForTests()
  $messagingSessions.set([])
})

afterEach(() => {
  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('message kind registration', () => {
  it('registers the message kind with the native notification pipeline', () => {
    expect(NATIVE_NOTIFICATION_KINDS).toContain('message')
  })
})

describe('inbound messaging notifications', () => {
  it('seeds on first sight and notifies only for a later count rise', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10, preview: 'hello' })])
    expect(notify).not.toHaveBeenCalled()

    $messagingSessions.set([session(id, { message_count: 14, preview: 'anyone there?' })])
    expect(notify).toHaveBeenCalledTimes(1)
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'message', sessionId: id, body: 'anyone there?' })
    )
  })

  it('ignores an unchanged count', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10 })])
    $messagingSessions.set([session(id, { message_count: 10 })])
    expect(notify).not.toHaveBeenCalled()
  })

  it('is suppressed while the window is focused', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10 })])
    setWindowState({ focused: true, hidden: false })
    $messagingSessions.set([session(id, { message_count: 11 })])
    expect(notify).not.toHaveBeenCalled()
  })

  it('is suppressed when the config key is off (default-off contract)', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10 })])
    $desktopNotifyIncomingMessages.set(false)
    $messagingSessions.set([session(id, { message_count: 11 })])
    expect(notify).not.toHaveBeenCalled()
  })

  it('is suppressed when the per-kind preference is off', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10 })])
    setNativeNotifyKind('message', false)
    $messagingSessions.set([session(id, { message_count: 11 })])
    expect(notify).not.toHaveBeenCalled()
  })

  it('falls back to the generic body when the preview is empty', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10, preview: null })])
    $messagingSessions.set([session(id, { message_count: 11, preview: null })])
    expect(notify).toHaveBeenCalledTimes(1)
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'message', sessionId: id, body: 'An inbound message arrived on a messaging session.' })
    )
  })

  it('follows a compression rotation that re-projects a lower count', () => {
    const id = freshMessagingSession()
    $messagingSessions.set([session(id, { message_count: 10 })])
    // Rotation: the tip re-projects with a smaller count before new messages land.
    $messagingSessions.set([session(id, { message_count: 2 })])
    expect(notify).not.toHaveBeenCalled()

    $messagingSessions.set([session(id, { message_count: 3 })])
    expect(notify).toHaveBeenCalledTimes(1)
  })
})

describe('dispatchNativeNotification message gating', () => {
  it('fires for a non-active session while away', () => {
    const id = freshMessagingSession()
    setActiveSessionId('on-screen')
    expect(dispatchNativeNotification({ kind: 'message', sessionId: id, title: 'New message' })).toBe(true)
  })

  it('does not require the session to be the active one', () => {
    const id = freshMessagingSession()
    setActiveSessionId(id)
    expect(dispatchNativeNotification({ kind: 'message', sessionId: id, title: 'New message' })).toBe(true)
  })

  it('requires a session id', () => {
    expect(dispatchNativeNotification({ kind: 'message', title: 'New message' })).toBe(false)
    expect(notify).not.toHaveBeenCalled()
  })
})
