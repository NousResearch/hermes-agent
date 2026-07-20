import { cleanup, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  getRememberedSessionId,
  setRememberedRoute,
  setRememberedSessionId,
  setSessions
} from '@/store/session'

import { NEW_CHAT_ROUTE, sessionRoute } from '../../routes'

import { useDesktopIntegrations } from './use-desktop-integrations'

vi.mock('@/app/chat/close-tab', () => ({
  closeActiveTab: vi.fn()
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

vi.mock('@/store/native-notifications', () => ({
  respondToApprovalAction: vi.fn()
}))

vi.mock('@/store/session-sync', () => ({
  onSessionsChanged: vi.fn(() => undefined)
}))

vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))

vi.mock('@/store/windows', () => ({
  isSecondaryWindow: vi.fn(() => false)
}))

function makeSession(id: string, isActive = false): { _lineage_root_id?: null | string; id: string; is_active: boolean } {
  return { id, is_active: isActive, _lineage_root_id: null }
}

interface HarnessProps {
  locationPathname: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  refreshSessions: () => Promise<unknown> | unknown
}

function Harness(props: HarnessProps): ReactNode {
  useDesktopIntegrations({
    chatOpen: true,
    hasPreview: false,
    locationPathname: props.locationPathname,
    navigate: props.navigate,
    refreshSessions: props.refreshSessions,
    resumeExhaustedSessionId: null,
    routedSessionId: null,
    runtimeIdByStoredSessionId: { current: new Map() }
  })

  return null
}

describe('useDesktopIntegrations', () => {
  afterEach(() => {
    cleanup()
    setRememberedRoute(null)
    setRememberedSessionId(null)
    setSessions([] as any)
    vi.restoreAllMocks()
  })

  it('falls back to the most recent live session when the remembered session id is stale on cold boot', async () => {
    setRememberedRoute(null)
    setRememberedSessionId('dead-session')

    const navigate = vi.fn()

    const refreshSessions = vi.fn(async () => {
      setSessions([makeSession('live-session', true), makeSession('older-session', false)] as any)
    })

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} refreshSessions={refreshSessions} />)

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(sessionRoute('live-session'), { replace: true })
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(getRememberedSessionId()).toBe('live-session')
  })

  it('lands on new chat when the remembered session and backend session list are both empty', async () => {
    setRememberedRoute(null)
    setRememberedSessionId('dead-session')

    const navigate = vi.fn()

    const refreshSessions = vi.fn(async () => {
      setSessions([] as any)
    })

    render(<Harness locationPathname={NEW_CHAT_ROUTE} navigate={navigate} refreshSessions={refreshSessions} />)

    await waitFor(() => {
      expect(navigate).toHaveBeenCalledWith(NEW_CHAT_ROUTE, { replace: true })
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
    expect(getRememberedSessionId()).toBeNull()
  })
})
