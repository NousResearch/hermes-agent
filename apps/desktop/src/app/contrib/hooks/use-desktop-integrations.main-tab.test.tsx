import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $showAllProfiles } from '@/store/profile'
import { $selectedStoredSessionId, _resetLegacyDiscardForTests, _resetSessionOwnerHintsForTests } from '@/store/session'
import { VIEW_MAIN_KEY } from '@/store/session-tile-storage'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../../../test/session-info'

import { useDesktopIntegrations } from './use-desktop-integrations'

// The first tab (workspace pane) reopens from the per-view memory the other
// tabs use, not from the sidebar's recent page or the gateway profile.

const { probeStoredSession } = vi.hoisted(() => ({
  probeStoredSession: vi.fn<(id: string) => Promise<unknown>>()
}))

vi.mock('@/app/session/hooks/use-session-actions/utils', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  probeStoredSession
}))

vi.mock('@/app/chat/close-tab', () => ({ closeActiveTab: vi.fn(() => true) }))
vi.mock('@/store/mcp-deeplink-install', () => ({ requestMcpInstallFromDeepLink: vi.fn() }))
vi.mock('@/store/plugin-catalog-install', () => ({ requestPluginCatalogInstallFromDeepLink: vi.fn() }))
vi.mock('@/store/plugin-install-request', () => ({ openPluginInstallRequest: vi.fn() }))

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const session = (over: Partial<SessionInfo> = {}): SessionInfo => makeSessionInfo({ id: 'live', ...over })

// 50 rows = the sidebar's recent page; the first tab's chat is older.
const page = Array.from({ length: 50 }, (_, i) => session({ id: `recent-${i}`, profile: 'default' }))

const ALL = '__all__'
const LEGACY_LAST = 'hermes.desktop.lastSessionId.profile.default'

const rememberMain = (view: string, id: null | string) =>
  window.localStorage.setItem(VIEW_MAIN_KEY, JSON.stringify({ [view]: id }))

const viewMain = (view: string) =>
  (JSON.parse(window.localStorage.getItem(VIEW_MAIN_KEY) ?? '{}') as Record<string, unknown>)[view]

describe('first tab cold-start restore', () => {
  let navigate: ReturnType<typeof vi.fn<(...args: unknown[]) => void>>

  beforeEach(() => {
    window.localStorage.clear()
    _resetLegacyDiscardForTests()
    _resetSessionOwnerHintsForTests({ storage: true })
    $selectedStoredSessionId.set(null)
    $showAllProfiles.set(true)
    probeStoredSession.mockReset()
    navigate = vi.fn()
    desktopWindow.hermesDesktop = {
      setPreviewShortcutActive: vi.fn(),
      onOpenUpdatesRequested: vi.fn(),
      onFocusSession: vi.fn(),
      onNotificationAction: vi.fn(),
      onNotificationActivate: vi.fn(),
      onDeepLink: vi.fn(),
      signalDeepLinkReady: vi.fn(),
      onClosePreviewRequested: vi.fn(),
      onOpenFolderRequested: vi.fn(),
      logLine: vi.fn(),
      api: vi.fn(async () => {
        throw new Error('no api stub for this test')
      })
    } as unknown as Window['hermesDesktop']
  })

  afterEach(() => {
    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    }

    $showAllProfiles.set(false)
    $selectedStoredSessionId.set(null)
    vi.restoreAllMocks()
  })

  const render = (sessions: readonly SessionInfo[]) =>
    renderHook(() =>
      useDesktopIntegrations({
        activeProfile: 'default',
        chatOpen: false,
        hasPreview: false,
        locationPathname: '/',
        navigate,
        profileReady: true,
        refreshSessions: vi.fn(),
        resumeExhaustedSessionId: null,
        resumeLastSession: true,
        routedSessionId: null,
        runtimeIdByStoredSessionId: { current: new Map() },
        sessions
      })
    )

  it('reopens a chat older than the sidebar page instead of a new session', async () => {
    rememberMain(ALL, 'old-chat')
    probeStoredSession.mockResolvedValue({ session: session({ id: 'old-chat', profile: 'default' }), status: 'found' })

    render(page)

    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/old-chat', { replace: true }))
    expect(viewMain(ALL)).toBe('old-chat')
  })

  it('reopens another profile chat in "All profiles"', () => {
    rememberMain(ALL, 'metatron-chat')

    render([...page, session({ id: 'metatron-chat', profile: 'metatron' })])

    expect(navigate).toHaveBeenCalledWith('/metatron-chat', { replace: true })
    expect(probeStoredSession).not.toHaveBeenCalled()
  })

  it('a view whose first tab was a new session stays a new session', () => {
    rememberMain(ALL, null)
    window.localStorage.setItem(LEGACY_LAST, 'recent-3')

    render(page)

    expect(navigate).not.toHaveBeenCalled()
  })

  it('forgets a chat the backend no longer has', async () => {
    rememberMain(ALL, 'deleted-chat')
    probeStoredSession.mockResolvedValue({ status: 'gone' })

    render(page)

    await waitFor(() => expect(viewMain(ALL)).toBeNull())
    expect(navigate).not.toHaveBeenCalled()
  })

  it('keeps the memory when the backend cannot answer yet', async () => {
    rememberMain(ALL, 'old-chat')
    probeStoredSession.mockResolvedValue({ status: 'inconclusive' })

    render(page)

    await waitFor(() => expect(probeStoredSession).toHaveBeenCalled())
    await Promise.resolve()
    expect(navigate).not.toHaveBeenCalled()
    expect(viewMain(ALL)).toBe('old-chat')
  })

  it('a chat the person opened during the lookup wins', async () => {
    rememberMain(ALL, 'old-chat')
    let answer: (value: unknown) => void = () => undefined
    probeStoredSession.mockReturnValue(new Promise(resolve => (answer = resolve)))

    render(page)
    $selectedStoredSessionId.set('recent-1')
    answer({ session: session({ id: 'old-chat', profile: 'default' }), status: 'found' })

    await waitFor(() => expect(probeStoredSession).toHaveBeenCalled())
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(navigate).not.toHaveBeenCalledWith('/old-chat', { replace: true })
  })

  it('a delegate child reopens as its parent', async () => {
    rememberMain(ALL, 'child')
    probeStoredSession.mockResolvedValue({
      session: session({ id: 'child', parent_session_id: 'parent', profile: 'default', source: 'subagent' }),
      status: 'found'
    })

    render(page)

    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/parent', { replace: true }))
  })

  it('waits for the session list before deciding', () => {
    rememberMain(ALL, 'old-chat')

    render([])

    expect(navigate).not.toHaveBeenCalled()
    expect(probeStoredSession).not.toHaveBeenCalled()
  })

  it('a view with no memory yet keeps the per-profile restore', () => {
    $showAllProfiles.set(false)
    window.localStorage.setItem(LEGACY_LAST, 'recent-3')

    render(page)

    expect(navigate).toHaveBeenCalledWith('/recent-3', { replace: true })
  })
})
