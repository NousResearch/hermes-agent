import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NATIVE_NOTIFICATION_KINDS, setNativeNotifyEnabled, setNativeNotifyKind } from '@/store/native-notifications'
import { __resetNativeNotifyBaselineForTests } from '@/store/notify-baseline'
import { clearApprovalRequest } from '@/store/prompts'
import { setMessagingSessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const notify = vi.fn().mockResolvedValue(true)

const session = (id: string, title: string): SessionInfo =>
  ({ id, message_count: 1, source: 'desktop', started_at: 0, title }) as SessionInfo

describe('background prompt notifications', () => {
  beforeEach(() => {
    notify.mockClear()
    desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
    Object.defineProperty(window.document, 'hidden', { configurable: true, value: false })
    Object.defineProperty(window.document, 'hasFocus', { configurable: true, value: () => true })
    setNativeNotifyEnabled(true)

    for (const kind of NATIVE_NOTIFICATION_KINDS) {
      setNativeNotifyKind(kind, true)
    }

    __resetNativeNotifyBaselineForTests()
    setSessions([])
    setMessagingSessions([])
  })

  afterEach(() => {
    cleanup()
    clearApprovalRequest()
    setSessions([])
    setMessagingSessions([])

    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    } else {
      delete desktopWindow.hermesDesktop
    }
  })

  it('names the off-screen chat that is waiting for approval', () => {
    const stream = renderMessageStream('foreground')
    setSessions([session('background-approval', 'Release checklist')])

    act(() =>
      stream.handleEvent({
        payload: {
          command: 'npm run test',
          description: 'Run the test suite',
          request_id: 'approval-1'
        },
        session_id: 'background-approval',
        type: 'approval.request'
      })
    )

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'approval',
        sessionId: 'background-approval',
        title: expect.stringMatching(/ · Release checklist$/)
      })
    )
  })

  it('names a messaging-platform chat from the separate session list', () => {
    const stream = renderMessageStream('foreground')
    setMessagingSessions([session('telegram-approval', 'Release room')])

    act(() =>
      stream.handleEvent({
        payload: {
          command: 'npm run test',
          description: 'Run the test suite',
          request_id: 'approval-2'
        },
        session_id: 'telegram-approval',
        type: 'approval.request'
      })
    )

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'approval',
        sessionId: 'telegram-approval',
        title: expect.stringMatching(/ · Release room$/)
      })
    )
  })
})
