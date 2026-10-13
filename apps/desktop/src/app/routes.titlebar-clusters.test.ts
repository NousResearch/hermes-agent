import { describe, expect, it } from 'vitest'

import { appViewForPath, CAPABILITIES_ROUTE, hidesFixedTitlebarClusters, isOverlayView, isWorkspacePageRoute } from './routes'

describe('hidesFixedTitlebarClusters', () => {
  it('hides clusters on contributed full pages and overlays', () => {
    expect(hidesFixedTitlebarClusters('extension')).toBe(true)
    expect(hidesFixedTitlebarClusters('settings')).toBe(true)
    // Capabilities floats over the chat like Settings, so the chat's titlebar
    // clusters step aside for it too.
    expect(hidesFixedTitlebarClusters('capabilities')).toBe(true)
  })

  it('keeps clusters on chat and first-party workspace pages', () => {
    expect(hidesFixedTitlebarClusters('chat')).toBe(false)
  })
})

describe('capabilities is an overlay', () => {
  it('classifies the route as an overlay, not a workspace page', () => {
    expect(appViewForPath(CAPABILITIES_ROUTE)).toBe('capabilities')
    expect(isOverlayView('capabilities')).toBe(true)
    expect(isWorkspacePageRoute(CAPABILITIES_ROUTE)).toBe(false)
    expect(isWorkspacePageRoute(`${CAPABILITIES_ROUTE}?tab=connectors`)).toBe(false)
  })
})
