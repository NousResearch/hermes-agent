import { describe, expect, it } from 'vitest'

import { HERMES_WINDOW_TITLE, shouldShowChatHeader, windowTitleForChat } from './header-state'

describe('shouldShowChatHeader', () => {
  it('keeps the clickable title header for loaded session pop-out windows', () => {
    expect(
      shouldShowChatHeader({
        activeSessionId: 'runtime-1',
        isRoutedSessionView: false,
        selectedSessionId: 'stored-1'
      })
    ).toBe(true)
  })

  it('keeps the header while a routed secondary session is still resolving', () => {
    expect(
      shouldShowChatHeader({
        activeSessionId: null,
        isRoutedSessionView: true,
        selectedSessionId: null
      })
    ).toBe(true)
  })

  it('hides the header only for a truly empty draft', () => {
    expect(
      shouldShowChatHeader({
        activeSessionId: null,
        isRoutedSessionView: false,
        selectedSessionId: null
      })
    ).toBe(false)
  })
})

describe('windowTitleForChat', () => {
  it('uses the session title in the native window title', () => {
    expect(windowTitleForChat('Mission Control')).toBe('Mission Control — Hermes')
  })

  it('falls back to Hermes for empty titles', () => {
    expect(windowTitleForChat('   ')).toBe(HERMES_WINDOW_TITLE)
  })
})
