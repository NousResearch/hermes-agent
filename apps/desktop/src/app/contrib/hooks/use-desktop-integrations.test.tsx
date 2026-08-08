import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { _resetLegacyDiscardForTests } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { useDesktopIntegrations } from './use-desktop-integrations'

// Pure-jsdom localStorage (no nanostores persistence module needed — the
// production functions write directly to window.localStorage through the
// persistString/storedString helpers in @/lib/storage, which in jsdom resolves
// to the real localStorage global).
// We import the hook and drive it with explicit rx-stores/props to exercise the
// profile-ready gate, ownership validation, and legacy-key discard.

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const session = (over: Partial<SessionInfo> = {}): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

describe('useDesktopIntegrations', () => {
  let navigate: ReturnType<typeof vi.fn<(...args: unknown[]) => void>>

  beforeEach(() => {
    window.localStorage.clear()
    _resetLegacyDiscardForTests()
    navigate = vi.fn()

    // Stub the desktop bridge so the hook's useEffect callbacks don't try to
    // reach real Electron IPC. The established desktop-test pattern assigns a
    // plain object to window.hermesDesktop rather than using vi.spyOn.
    desktopWindow.hermesDesktop = {
      setPreviewShortcutActive: vi.fn(),
      onOpenUpdatesRequested: vi.fn(),
      onFocusSession: vi.fn(),
      onNotificationAction: vi.fn(),
      onDeepLink: vi.fn(),
      signalDeepLinkReady: vi.fn(),
      onClosePreviewRequested: vi.fn(),
      onOpenFolderRequested: vi.fn()
    } as unknown as Window['hermesDesktop']
  })

  afterEach(() => {
    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    }

    vi.restoreAllMocks()
  })

  function render({
    activeProfile = 'default',
    locationPathname = '/',
    profileReady = false,
    resumeExhaustedSessionId = null as string | null,
    routedSessionId = null as string | null,
    sessions = [] as readonly SessionInfo[]
  } = {}) {
    return renderHook(
      ({
        activeProfile,
        locationPathname,
        profileReady,
        resumeExhaustedSessionId,
        routedSessionId,
        sessions
      }: {
        activeProfile: string
        locationPathname: string
        profileReady: boolean
        resumeExhaustedSessionId: string | null
        routedSessionId: string | null
        sessions: readonly SessionInfo[]
      }) =>
        useDesktopIntegrations({
          activeProfile,
          chatOpen: false,
          hasPreview: false,
          locationPathname,
          navigate,
          profileReady,
          refreshSessions: vi.fn(),
          resumeExhaustedSessionId,
          routedSessionId,
          runtimeIdByStoredSessionId: { current: new Map() },
          sessions
        }),
      {
        initialProps: {
          activeProfile,
          locationPathname,
          profileReady,
          resumeExhaustedSessionId,
          routedSessionId,
          sessions
        }
      }
    )
  }

  describe('profile-ready gate', () => {
    it('does NOT restore before profileReady is true', () => {
      // Set remembered state, but profileReady=false.
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/remembered-session')
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'remembered-session')

      render({ profileReady: false })

      // no navigation should have occurred
      expect(navigate).not.toHaveBeenCalled()
    })

    it('restores on profileReady when remembered route exists and owns the session', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/remembered-session')

      const sessions = [session({ id: 'remembered-session', profile: 'default' })]

      render({ profileReady: true, sessions })

      expect(navigate).toHaveBeenCalledWith('/remembered-session', { replace: true })
    })

    it('restores remembered session id when no remembered route exists', () => {
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'remembered-session')

      const sessions = [session({ id: 'remembered-session', profile: 'default' })]

      render({ profileReady: true, sessions })

      // sessionRoute('remembered-session') = '/remembered-session'
      expect(navigate).toHaveBeenCalledWith('/remembered-session', { replace: true })
    })

    it('preserves remembered New Chat instead of restoring a stale session', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/')
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'remembered-session')

      const sessions = [session({ id: 'remembered-session', profile: 'default' })]

      render({ profileReady: true, sessions })

      expect(navigate).not.toHaveBeenCalled()
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBeNull()
    })

    it('preserves remembered New Chat after waiting for sessions to load', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/')
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'remembered-session')

      const result = render({ profileReady: true, sessions: [] })

      expect(navigate).not.toHaveBeenCalled()
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBe('remembered-session')

      result.rerender({
        activeProfile: 'default',
        locationPathname: '/',
        profileReady: true,
        resumeExhaustedSessionId: null,
        routedSessionId: null,
        sessions: [session({ id: 'remembered-session', profile: 'default' })]
      })

      expect(navigate).not.toHaveBeenCalled()
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBeNull()
    })

    it('waits for sessions before validating a remembered session route', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/remembered-session')

      const result = render({ profileReady: true, sessions: [] })

      expect(navigate).not.toHaveBeenCalled()
      expect(window.localStorage.getItem('hermes.desktop.lastRoute.profile.default')).toBe('/remembered-session')

      result.rerender({
        activeProfile: 'default',
        locationPathname: '/',
        profileReady: true,
        resumeExhaustedSessionId: null,
        routedSessionId: null,
        sessions: [session({ id: 'remembered-session', profile: 'default' })]
      })

      expect(navigate).toHaveBeenCalledWith('/remembered-session', { replace: true })
    })
  })

  describe('ownership validation', () => {
    it('refuses to restore a session route owned by another profile', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/ai-session')

      const sessions = [session({ id: 'ai-session', profile: 'ai-engineer' })]

      // The route belongs to ai-engineer; active profile is default.
      // No navigation should happen — wrong owner.
      render({ activeProfile: 'default', profileReady: true, sessions })

      expect(navigate).not.toHaveBeenCalled()
    })

    it('refuses to restore a session id owned by another profile', () => {
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'ai-session')
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/ai-session')

      const sessions = [session({ id: 'ai-session', profile: 'ai-engineer' })]

      render({ activeProfile: 'default', profileReady: true, sessions })

      // Both route and fallback session id are owned by another profile.
      expect(navigate).not.toHaveBeenCalled()
    })

    it('clears stale remembered route owned by wrong profile', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.ai-engineer', '/ai-session')

      const sessions = [session({ id: 'ai-session', profile: 'ai-engineer' })]

      render({ activeProfile: 'ai-engineer', profileReady: true, sessions })

      // The route and session match the active profile — should restore.
      expect(navigate).toHaveBeenCalledWith('/ai-session', { replace: true })
    })
  })

  describe('two profiles with distinct sessions', () => {
    it('restores profile A session when profile A is active', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.coder', '/coder-session')

      const sessions = [
        session({ id: 'coder-session', profile: 'coder' }),
        session({ id: 'ops-session', profile: 'ops' })
      ]

      render({ activeProfile: 'coder', profileReady: true, sessions })

      expect(navigate).toHaveBeenCalledWith('/coder-session', { replace: true })
    })

    it('does NOT bleed profile A session into profile B', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.coder', '/coder-session')

      const sessions = [session({ id: 'coder-session', profile: 'coder' })]

      // ops profile is active but has no own remembered route
      render({
        activeProfile: 'ops',
        profileReady: true,
        sessions
      })

      // No navigation — coder's remembered route doesn't belong to ops.
      expect(navigate).not.toHaveBeenCalled()
    })
  })

  describe('legacy key behavior', () => {
    it('discards legacy global keys on read and does NOT restore from them', () => {
      // Simulate a pre-per-profile install.
      window.localStorage.setItem('hermes.desktop.lastSessionId', 'legacy-session')
      window.localStorage.setItem('hermes.desktop.lastRoute', '/session/legacy-session')

      // Profile contexts without matching sessions.
      const sessions = [session({ id: 'legacy-session', profile: 'default' })]

      render({ profileReady: true, sessions })

      // Legacy keys must be discarded.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId')).toBeNull()
      expect(window.localStorage.getItem('hermes.desktop.lastRoute')).toBeNull()

      // And no navigation should happen (the per-profile keys were empty).
      expect(navigate).not.toHaveBeenCalled()
    })
  })

  describe('stale-result suppression during profile switch', () => {
    it('remembers route for the new profile after switch, not the old one', () => {
      const sessions = [
        session({ id: 'coder-session', profile: 'coder' }),
        session({ id: 'ops-session', profile: 'ops' })
      ]

      // Render with coder active and navigate to a session.
      const { rerender } = render({
        activeProfile: 'coder',
        locationPathname: '/coder-session',
        profileReady: true,
        routedSessionId: 'coder-session',
        sessions
      })

      // The coder session should be persisted under coder's key.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.coder')).toBe('coder-session')

      // Now switch to ops.
      rerender({
        activeProfile: 'ops',
        locationPathname: '/ops-session',
        profileReady: true,
        resumeExhaustedSessionId: null,
        routedSessionId: 'ops-session',
        sessions
      })

      // The ops session should now be persisted under ops's key.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.ops')).toBe('ops-session')

      // Coder's remembered session should still be there.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.coder')).toBe('coder-session')
    })

    it('does NOT overwrite remembered state when session ownership fails validation', () => {
      // Simulate an async restore result arriving for a route that doesn't
      // own the active profile.
      const sessions = [session({ id: 'coder-session', profile: 'coder' })]

      // Active profile is ops, but the routed session belongs to coder.
      render({
        activeProfile: 'ops',
        locationPathname: '/',
        profileReady: true,
        routedSessionId: 'coder-session', // wrong profile!
        sessions
      })

      // No session should be remembered for the active profile.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.ops')).toBeNull()
    })
  })

  describe('route-scoped restoration', () => {
    it('restores a non-session route like /skills', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/skills')

      const sessions = [session({ id: 'some-session', profile: 'default' })]

      render({ profileReady: true, sessions })

      // /skills is not a session route — no ownership validation needed.
      expect(navigate).toHaveBeenCalledWith('/skills', { replace: true })
    })

    it('does NOT restore overlay routes (settings/command-center)', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/settings')

      render({ profileReady: true, sessions: [] })

      // Overlay routes should not be restored.
      expect(navigate).not.toHaveBeenCalled()
    })

    it('does NOT persist overlay routes for next boot', () => {
      const { rerender } = render({
        activeProfile: 'default',
        locationPathname: '/settings',
        profileReady: true,
        routedSessionId: null,
        sessions: []
      })

      // Remembering effect fires on route change.
      rerender({
        activeProfile: 'default',
        locationPathname: '/settings',
        profileReady: true,
        resumeExhaustedSessionId: null,
        routedSessionId: null,
        sessions: []
      })

      // Overlay routes must NOT be persisted.
      expect(window.localStorage.getItem('hermes.desktop.lastRoute.profile.default')).toBeNull()
    })
  })

  describe('exhausted session cleanup', () => {
    it('clears remembered session id when the exhausted session matches', () => {
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'exhausted')

      const sessions = [session({ id: 'exhausted', profile: 'default' })]

      render({
        profileReady: true,
        resumeExhaustedSessionId: 'exhausted',
        sessions
      })

      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBeNull()
    })

    it('clears remembered route when it carries the exhausted session', () => {
      window.localStorage.setItem('hermes.desktop.lastRoute.profile.default', '/exhausted')

      const sessions = [session({ id: 'exhausted', profile: 'default' })]

      render({
        profileReady: true,
        resumeExhaustedSessionId: 'exhausted',
        sessions
      })

      expect(window.localStorage.getItem('hermes.desktop.lastRoute.profile.default')).toBeNull()
    })

    it('does NOT clear exhausted when profileReady is false', () => {
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'exhausted')

      render({
        profileReady: false,
        resumeExhaustedSessionId: 'exhausted',
        sessions: []
      })

      // profileReady=false gates the cleanup effect.
      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBe('exhausted')
    })

    it('does NOT clear remembered state when exhausted id does not match', () => {
      window.localStorage.setItem('hermes.desktop.lastSessionId.profile.default', 'other-session')

      render({
        profileReady: true,
        resumeExhaustedSessionId: 'exhausted',
        sessions: [session({ id: 'other-session', profile: 'default' })]
      })

      expect(window.localStorage.getItem('hermes.desktop.lastSessionId.profile.default')).toBe('other-session')
    })
  })
})
