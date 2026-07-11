import { describe, expect, it } from 'vitest'

import { shouldShowPreviewWorkspace } from './preview-workspace-context'

describe('shouldShowPreviewWorkspace', () => {
  it('requires chat context, a selected session, and at least one surface', () => {
    expect(
      shouldShowPreviewWorkspace({
        chatOpen: true,
        hasPreviewSurfaces: true,
        routedSessionId: 'session-1',
        selectedStoredSessionId: null
      })
    ).toBe(true)
    expect(
      shouldShowPreviewWorkspace({
        chatOpen: true,
        hasPreviewSurfaces: true,
        routedSessionId: null,
        selectedStoredSessionId: 'stored-1'
      })
    ).toBe(true)
    expect(
      shouldShowPreviewWorkspace({
        chatOpen: true,
        hasPreviewSurfaces: true,
        routedSessionId: null,
        selectedStoredSessionId: null
      })
    ).toBe(false)
    expect(
      shouldShowPreviewWorkspace({
        chatOpen: false,
        hasPreviewSurfaces: true,
        routedSessionId: 'session-1',
        selectedStoredSessionId: null
      })
    ).toBe(false)
    expect(
      shouldShowPreviewWorkspace({
        chatOpen: true,
        hasPreviewSurfaces: false,
        routedSessionId: 'session-1',
        selectedStoredSessionId: null
      })
    ).toBe(false)
  })
})
