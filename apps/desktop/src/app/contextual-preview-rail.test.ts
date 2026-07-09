import { describe, expect, it } from 'vitest'

import { shouldShowContextualPreviewRail } from './contextual-preview-rail'

describe('contextual preview rail visibility', () => {
  it('stays hidden for the new-chat route even when stale preview tabs exist', () => {
    expect(
      shouldShowContextualPreviewRail({
        currentView: 'chat',
        hasPreviewTarget: true,
        routedSessionId: null,
        selectedStoredSessionId: null
      })
    ).toBe(false)
  })

  it.each(['settings', 'command-center', 'agents', 'cron', 'profiles', 'starmap'] as const)(
    'stays hidden behind the full-screen %s overlay',
    currentView => {
      expect(
        shouldShowContextualPreviewRail({
          currentView,
          hasPreviewTarget: true,
          routedSessionId: 'session-1',
          selectedStoredSessionId: null
        })
      ).toBe(false)
    }
  )

  it.each(['skills', 'messaging', 'artifacts'] as const)(
    'stays hidden on the non-chat %s workspace view',
    currentView => {
      expect(
        shouldShowContextualPreviewRail({
          currentView,
          hasPreviewTarget: true,
          routedSessionId: 'session-1',
          selectedStoredSessionId: null
        })
      ).toBe(false)
    }
  )

  it('is visible for an actual chat session route with a preview or editor target', () => {
    expect(
      shouldShowContextualPreviewRail({
        currentView: 'chat',
        hasPreviewTarget: true,
        routedSessionId: 'session-1',
        selectedStoredSessionId: null
      })
    ).toBe(true)
  })

  it('is hidden for real chat sessions until there is a preview or editor target', () => {
    expect(
      shouldShowContextualPreviewRail({
        currentView: 'chat',
        hasPreviewTarget: false,
        routedSessionId: 'session-1',
        selectedStoredSessionId: null
      })
    ).toBe(false)
  })
})
