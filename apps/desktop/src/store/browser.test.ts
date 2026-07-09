import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $browserCurrentSessionHasBrowser,
  $browserCurrentState,
  $browserDriveCommand,
  $browserSessionRegistry,
  closeCurrentBrowserSession,
  driveBrowser,
  normalizeBrowserUrl,
  openBrowserRail,
  resetBrowserRegistryForTests,
  setBrowserSessionState
} from './browser'
import { RIGHT_RAIL_BROWSER_TAB_ID } from './layout'
import { $rightRailActiveTabId, PREVIEW_PANE_ID } from './layout'
import { $paneOpen } from './panes'
import { $activeSessionId, $selectedStoredSessionId } from './session'

describe('browser rail store', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-07-05T00:00:00Z'))
    resetBrowserRegistryForTests()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('normalizes browser URLs safely', () => {
    expect(normalizeBrowserUrl('example.com')).toBe('https://example.com/')
    expect(normalizeBrowserUrl('https://example.com/path')).toBe('https://example.com/path')
    expect(normalizeBrowserUrl('about:blank')).toBe('about:blank')
    expect(normalizeBrowserUrl('javascript:alert(1)')).toBe('about:blank')
  })

  it('stores browser state per session', () => {
    setBrowserSessionState('session-1', { title: 'One', url: 'example.com' })
    setBrowserSessionState('session-2', { title: 'Two', url: 'https://nousresearch.com' })

    expect($browserSessionRegistry.get()['session-1']).toMatchObject({
      title: 'One',
      url: 'https://example.com/'
    })
    expect($browserSessionRegistry.get()['session-2']).toMatchObject({
      title: 'Two',
      url: 'https://nousresearch.com/'
    })
  })

  it('opens the browser rail and selects the browser tab', () => {
    $activeSessionId.set('session-1')

    openBrowserRail('example.com')

    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_BROWSER_TAB_ID)
    expect($browserCurrentState.get()).toMatchObject({ url: 'https://example.com/' })
    expect($browserCurrentSessionHasBrowser.get()).toBe(true)
  })

  it('records external drive commands and opens the browser rail', () => {
    $activeSessionId.set('session-1')

    driveBrowser({ action: 'navigate', title: 'Example', url: 'example.com' })

    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_BROWSER_TAB_ID)
    expect($browserCurrentState.get()).toMatchObject({ title: 'Example', url: 'https://example.com/' })
    expect($browserDriveCommand.get()).toMatchObject({
      action: 'navigate',
      id: 1,
      sessionId: 'session-1',
      title: 'Example',
      url: 'example.com'
    })
  })

  it('does not treat sessions without browser records as open browsers', () => {
    $activeSessionId.set('session-1')
    openBrowserRail('example.com')

    $activeSessionId.set('session-2')

    expect($browserCurrentSessionHasBrowser.get()).toBe(false)
    expect($browserCurrentState.get()).toMatchObject({ url: 'about:blank' })
    expect($browserSessionRegistry.get()['session-2']).toBeUndefined()
  })

  it('removes the current session browser when its browser tab closes', () => {
    $activeSessionId.set('session-1')
    openBrowserRail('example.com')

    expect(closeCurrentBrowserSession()).toBe(true)
    expect($browserCurrentSessionHasBrowser.get()).toBe(false)
    expect($browserSessionRegistry.get()['session-1']).toBeUndefined()
  })
})
