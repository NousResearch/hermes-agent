import { cleanup, render } from '@testing-library/react'
import { type MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getRememberedRoute, getRememberedSessionId, setRememberedRoute, setRememberedSessionId } from '@/store/session'

import { useDesktopIntegrations } from './use-desktop-integrations'

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ respondToApprovalAction: vi.fn() }))
vi.mock('@/store/session-sync', () => ({ onSessionsChanged: vi.fn(() => () => undefined) }))
vi.mock('@/store/updates', () => ({
  openUpdatesWindow: vi.fn(),
  startUpdatePoller: vi.fn(),
  stopUpdatePoller: vi.fn()
}))
vi.mock('@/store/windows', () => ({ isSecondaryWindow: vi.fn(() => false) }))
vi.mock('../../chat/composer/focus', () => ({
  requestComposerFocus: vi.fn(),
  requestComposerInsert: vi.fn()
}))

const runtimeIds = { current: new Map<string, string>() } as MutableRefObject<Map<string, string>>

interface RememberedSession {
  _lineage_root_id?: null | string
  id: string
  profile?: string
}

interface HarnessProps {
  activeProfile: string
  locationPathname?: string
  navigate: (to: string, options?: { replace?: boolean }) => void
  profileReady: boolean
  resumeExhaustedSessionId?: null | string
  routedSessionId?: null | string
  sessions?: RememberedSession[]
}

function Harness({
  activeProfile,
  locationPathname = '/',
  navigate,
  profileReady,
  resumeExhaustedSessionId = null,
  routedSessionId = null,
  sessions = []
}: HarnessProps) {
  useDesktopIntegrations({
    activeProfile,
    chatOpen: true,
    hasPreview: false,
    locationPathname,
    navigate,
    profileReady,
    refreshSessions: () => undefined,
    resumeExhaustedSessionId,
    routedSessionId,
    runtimeIdByStoredSessionId: runtimeIds,
    sessions
  })

  return null
}

beforeEach(() => {
  window.localStorage.clear()
  vi.clearAllMocks()
})

afterEach(() => cleanup())

describe('profile-scoped remembered navigation', () => {
  it('waits for profile adoption and restores the adopted profile route before its session', () => {
    setRememberedRoute('/artifacts', 'default')
    setRememberedSessionId('default-session', 'default')
    setRememberedRoute('/skills', 'coder')
    setRememberedSessionId('coder-session', 'coder')
    const navigate = vi.fn()
    const view = render(<Harness activeProfile="default" navigate={navigate} profileReady={false} />)

    expect(navigate).not.toHaveBeenCalled()

    view.rerender(<Harness activeProfile="coder" navigate={navigate} profileReady />)

    expect(navigate).toHaveBeenCalledOnce()
    expect(navigate).toHaveBeenCalledWith('/skills', { replace: true })
  })

  it('leaves a profile with no history on new chat instead of inheriting default history', () => {
    setRememberedRoute('/skills', 'default')
    setRememberedSessionId('default-session', 'default')
    const navigate = vi.fn()

    render(<Harness activeProfile="fresh-profile" navigate={navigate} profileReady />)

    expect(navigate).not.toHaveBeenCalled()
  })

  it('falls back to an owned active-profile session when its remembered route is an overlay', () => {
    setRememberedRoute('/settings', 'coder')
    setRememberedSessionId('coder-session', 'coder')
    const navigate = vi.fn()

    render(
      <Harness
        activeProfile="coder"
        navigate={navigate}
        profileReady
        sessions={[{ id: 'coder-session', profile: 'coder' }]}
      />
    )

    expect(navigate).toHaveBeenCalledOnce()
    expect(navigate).toHaveBeenCalledWith('/coder-session', { replace: true })
  })

  it('persists and clears both navigation values only for the active profile', () => {
    setRememberedRoute('/skills', 'default')
    setRememberedSessionId('default-session', 'default')
    const navigate = vi.fn()
    const sessions = [{ id: 'coder-session', profile: 'coder' }]

    const view = render(
      <Harness
        activeProfile="coder"
        locationPathname="/coder-session"
        navigate={navigate}
        profileReady
        routedSessionId="coder-session"
        sessions={sessions}
      />
    )

    expect(getRememberedRoute('coder')).toBe('/coder-session')
    expect(getRememberedSessionId('coder')).toBe('coder-session')
    expect(getRememberedRoute('default')).toBe('/skills')
    expect(getRememberedSessionId('default')).toBe('default-session')

    view.rerender(
      <Harness
        activeProfile="coder"
        locationPathname="/coder-session"
        navigate={navigate}
        profileReady
        resumeExhaustedSessionId="coder-session"
        routedSessionId="coder-session"
        sessions={sessions}
      />
    )

    expect(getRememberedSessionId('coder')).toBeNull()
    expect(getRememberedRoute('coder')).toBeNull()
    expect(getRememberedSessionId('default')).toBe('default-session')
    expect(getRememberedRoute('default')).toBe('/skills')
  })

  it('restores the active owner when duplicate IDs exist in multiple profiles', () => {
    setRememberedRoute('/shared-session', 'coder')
    setRememberedSessionId('shared-session', 'coder')
    const navigate = vi.fn()

    render(
      <Harness
        activeProfile="coder"
        navigate={navigate}
        profileReady
        sessions={[
          { id: 'shared-session', profile: 'default' },
          { id: 'shared-session', profile: 'coder' }
        ]}
      />
    )

    expect(navigate).toHaveBeenCalledOnce()
    expect(navigate).toHaveBeenCalledWith('/shared-session', { replace: true })
  })

  it('does not persist unresolved or wrong-profile routed sessions', () => {
    setRememberedRoute('/skills', 'coder')
    setRememberedSessionId('safe-session', 'coder')
    const navigate = vi.fn()

    const view = render(
      <Harness
        activeProfile="coder"
        locationPathname="/unknown-session"
        navigate={navigate}
        profileReady
        routedSessionId="unknown-session"
      />
    )

    expect(getRememberedSessionId('coder')).toBe('safe-session')
    expect(getRememberedRoute('coder')).toBe('/skills')

    view.rerender(
      <Harness
        activeProfile="coder"
        locationPathname="/foreign-session"
        navigate={navigate}
        profileReady
        routedSessionId="foreign-session"
        sessions={[{ id: 'foreign-session', profile: 'default' }]}
      />
    )

    expect(getRememberedSessionId('coder')).toBe('safe-session')
    expect(getRememberedRoute('coder')).toBe('/skills')
  })
})
