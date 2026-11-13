import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { $selectedStoredSessionId, $sessions, applySessionTitle, setSessions } from './session'
import { clearAllSessionStates, publishSessionState } from './session-states'
import { installWindowTitle, windowTitle, type WindowTitleStatus } from './window-title'

// The real `isSecondaryWindow()` reads the window's query string once and
// caches it, which no test can undo — hold the answer here instead.
const opened = vi.hoisted(() => ({ secondary: true }))

vi.mock('./windows', () => ({
  isBrowserWindow: () => false,
  isSecondaryWindow: () => opened.secondary
}))

const row = (over: Partial<SessionInfo>): SessionInfo => makeSessionInfo({ message_count: 3, ...over })

const state = (storedId: string, over: { busy?: boolean; needsInput?: boolean } = {}) => ({
  ...createClientSessionState(storedId),
  ...over
})

const title = (over: Partial<WindowTitleStatus> = {}) =>
  windowTitle({ caption: 'Fix login', needsInput: false, working: false, ...over })

let uninstall = () => {}

function reset() {
  uninstall()

  uninstall = () => {}
  opened.secondary = true
  clearAllSessionStates()
  $sessions.set([])
  $selectedStoredSessionId.set(null)
  delete document.documentElement.dataset.hermesDesktopHost
  document.title = 'Hermes'
}

describe('windowTitle', () => {
  it('marks the front of the name and lets a blocking prompt speak over a turn', () => {
    const idle = title()
    const working = title({ working: true })
    const blocked = title({ needsInput: true, working: true })

    expect(idle).toBe('Fix login')
    expect(working).toBe(`● ${idle}`)
    expect(blocked).toBe(`⚠ ${idle}`)
    expect(blocked).toBe(title({ needsInput: true }))

    // An untitled session still names the app, never an empty window title.
    expect(title({ caption: '' })).toBe('Hermes')
    expect(title({ caption: '', working: true })).toBe('● Hermes')
  })
})

describe('installWindowTitle', () => {
  beforeEach(reset)
  afterEach(reset)

  it('follows the session this window shows, and only that session', () => {
    setSessions([row({ id: 's1', title: 'Fix login' }), row({ id: 's2', title: 'Draft essay' })])
    $selectedStoredSessionId.set('s1')
    uninstall = installWindowTitle()

    expect(document.title).toBe(title())

    // The titler writes the auto-title after the turn and the gateway pushes it.
    applySessionTitle('s1', 'Fix the flaky test')
    expect(document.title).toBe('Fix the flaky test')

    publishSessionState('r1', state('s1', { busy: true }))
    expect(document.title).toBe(title({ caption: 'Fix the flaky test', working: true }))

    // A background session's status belongs to ITS window, not this one.
    publishSessionState('r2', state('s2', { busy: true, needsInput: true }))
    expect(document.title).toBe(title({ caption: 'Fix the flaky test', working: true }))

    // This window's own block outranks its running turn.
    publishSessionState('r1', state('s1', { busy: true, needsInput: true }))
    expect(document.title).toBe(title({ caption: 'Fix the flaky test', needsInput: true }))

    publishSessionState('r1', state('s1'))
    expect(document.title).toBe('Fix the flaky test')
  })

  it('leaves an untitled session named after the app', () => {
    setSessions([row({ id: 's1', preview: 'my bank password is', title: null })])
    $selectedStoredSessionId.set('s1')
    uninstall = installWindowTitle()

    expect(document.title).toBe('Hermes')
  })

  it('leaves the primary window and a browser tab alone', () => {
    setSessions([row({ id: 's1', title: 'Fix login' })])
    $selectedStoredSessionId.set('s1')

    opened.secondary = false
    uninstall = installWindowTitle()
    publishSessionState('r1', state('s1', { busy: true }))
    expect(document.title).toBe('Hermes')
    uninstall()

    // A Webapp pop-out is `?win=secondary` too, but its tab title is the
    // browser-tab-title form, which owns the count and the whole-window mark.
    opened.secondary = true
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    uninstall = installWindowTitle()
    publishSessionState('r1', state('s1', { busy: true }))
    expect(document.title).toBe('Hermes')
  })
})
