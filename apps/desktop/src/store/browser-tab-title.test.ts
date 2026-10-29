import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { type BrowserTabStatus, browserTabTitle, installBrowserTabTitle } from './browser-tab-title'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unlistedSessionOwnerRows,
  $unreadFinishedSessionIds,
  applySessionTitle,
  setSessions
} from './session'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { $sessionSeenCounts, $unreadFinishedMarkers } from './session-unread'

const row = (over: Partial<SessionInfo>): SessionInfo => makeSessionInfo({ message_count: 3, ...over })

const state = (storedId: string, over: { busy?: boolean; needsInput?: boolean } = {}) => ({
  ...createClientSessionState(storedId),
  ...over
})

const tabTitle = (over: Partial<BrowserTabStatus> = {}) =>
  browserTabTitle({ caption: 'Fix login', needsInput: false, unread: 0, working: false, ...over })

let uninstall = () => {}

function reset() {
  uninstall()

  uninstall = () => {}
  clearAllSessionStates()
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $selectedStoredSessionId.set(null)
  $sessionSeenCounts.set({})
  $unreadFinishedMarkers.set({})
  $unreadFinishedSessionIds.set([])
  delete document.documentElement.dataset.hermesDesktopHost
  window.history.replaceState(null, '', '/')
  document.title = 'Hermes'
}

describe('browserTabTitle', () => {
  it('lets a blocking prompt speak over a running turn, behind the unread count', () => {
    const idle = tabTitle()
    const working = tabTitle({ working: true })
    const blocked = tabTitle({ needsInput: true, working: true })

    expect(idle).toContain('Fix login')
    // Each status marks the front of the idle title; a blocking prompt replaces
    // the running mark rather than joining it.
    expect(working).not.toBe(idle)
    expect(working.endsWith(idle)).toBe(true)
    expect(blocked).not.toBe(working)
    expect(blocked.endsWith(idle)).toBe(true)
    expect(blocked).toBe(tabTitle({ needsInput: true }))

    const counted = tabTitle({ needsInput: true, unread: 2 })
    expect(counted.endsWith(blocked)).toBe(true)
    expect(counted.slice(0, -blocked.length)).toContain('2')

    // Without a caption the title falls back to the app name alone.
    const bare = tabTitle({ caption: '' })
    expect(bare).not.toBe('')
    expect(idle.endsWith(bare)).toBe(true)
    expect(tabTitle({ caption: '', unread: 1 }).endsWith(bare)).toBe(true)
  })
})

describe('installBrowserTabTitle', () => {
  beforeEach(reset)
  afterEach(reset)

  it('follows the focused session and background status in a browser host', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    setSessions([row({ id: 's1', title: 'Fix login' }), row({ id: 's2', profile: 'writer', title: 'Draft essay' })])
    $selectedStoredSessionId.set('s1')
    uninstall = installBrowserTabTitle()

    expect(document.title).toBe(tabTitle())

    publishSessionState('r1', state('s1', { busy: true }))
    expect(document.title).toBe(tabTitle({ working: true }))

    // Another session blocks on an approval: the whole window needs the user.
    publishSessionState('r2', state('s2', { busy: true, needsInput: true }))
    expect(document.title).toBe(tabTitle({ needsInput: true, working: true }))

    // It finishes unwatched; the focused turn ends too, while being looked at.
    publishSessionState('r2', state('s2'))
    publishSessionState('r1', state('s1'))
    expect(document.title).toBe(tabTitle({ unread: 1 }))

    // Opening it reads it; a non-default profile names its owner.
    $selectedStoredSessionId.set('s2')
    $unreadFinishedSessionIds.set([])
    // Nothing is left to mark ahead of the caption, which names the owner.
    expect(document.title.startsWith('Draft essay')).toBe(true)
    expect(document.title).toContain('writer')
  })

  it('never puts unsent or untitled text in the tab', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    setSessions([row({ id: 's1', preview: 'my bank password is', title: null })])
    $selectedStoredSessionId.set('s1')
    uninstall = installBrowserTabTitle()

    expect(document.title).toBe('Hermes')
  })

  it.each([$cronSessions, $messagingSessions, $unlistedSessionOwnerRows])(
    'names a focused session outside the recents list',
    slice => {
      slice.set([row({ id: 'hidden', profile: 'writer', title: 'Scheduled report' })])
      $selectedStoredSessionId.set('hidden')
      uninstall = installBrowserTabTitle()

      expect(document.title).toBe('Scheduled report — writer · Hermes')
      applySessionTitle('hidden', 'Renamed report')
      expect(document.title).toBe('Renamed report — writer · Hermes')
    }
  )

  it.each(['hud', 'overlay', 'quick', 'wake', 'browser', 'secondary'])(
    'leaves the %s auxiliary window title untouched',
    win => {
      window.history.replaceState(null, '', `/?win=${win}`)
      document.title = `Hermes ${win}`
      setSessions([row({ id: 's1', title: 'Fix login' })])
      $selectedStoredSessionId.set('s1')
      uninstall = installBrowserTabTitle()
      setSessions([row({ id: 's1', title: 'Renamed session' })])

      expect(document.title).toBe(`Hermes ${win}`)
    }
  )

  it('follows auto titles, manual renames and focused tiles rather than background sessions', () => {
    setSessions([row({ id: 's1', title: null }), row({ id: 's2', profile: 'writer', title: 'Essay' })])
    $selectedStoredSessionId.set('s1')
    uninstall = installBrowserTabTitle()
    expect(document.title).toBe('Hermes')

    // Both backend auto-title events and manual rename receipts use this reducer.
    applySessionTitle('s1', 'Generated title')
    expect(document.title).toBe('Generated title · Hermes')
    applySessionTitle('s1', 'My chosen title')
    expect(document.title).toBe('My chosen title · Hermes')

    $layoutTree.set(group(['workspace', 'session-tile:s2'], { active: 'session-tile:s2', id: 'main' }))
    $activeTreeGroup.set('main')
    expect(document.title).toBe('Essay — writer · Hermes')
    applySessionTitle('s1', 'Background rename')
    expect(document.title).toBe('Essay — writer · Hermes')
    applySessionTitle('s2', 'Revised essay')
    expect(document.title).toBe('Revised essay — writer · Hermes')

    $layoutTree.set(group(['workspace', 'session-tile:s2'], { active: 'workspace', id: 'main' }))
    expect(document.title).toBe('Background rename · Hermes')
    $selectedStoredSessionId.set(null)
    expect(document.title).toBe('Hermes')
  })

  it('stops updating the document when uninstalled', () => {
    setSessions([row({ id: 's1', title: 'First title' })])
    $selectedStoredSessionId.set('s1')
    uninstall = installBrowserTabTitle()
    uninstall()
    applySessionTitle('s1', 'Later title')

    expect(document.title).toBe('First title · Hermes')
  })

  it('names the native window after the focused session without browser status marks', () => {
    setSessions([row({ id: 's1', title: 'Fix login' })])
    $selectedStoredSessionId.set('s1')
    uninstall = installBrowserTabTitle()
    publishSessionState('r1', state('s1', { busy: true }))

    expect(document.title).toBe('Fix login · Hermes')
  })
})
