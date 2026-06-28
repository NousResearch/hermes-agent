import { describe, expect, it } from 'vitest'

import { appViewForPath, routeSessionId } from './routes'

describe('desktop routes', () => {
  it('does not treat dashboard plugin routes as chat session ids', () => {
    const pluginRoutes = new Set(['/kanban', '/achievements'])

    expect(routeSessionId('/kanban', pluginRoutes)).toBeNull()
    expect(routeSessionId('/achievements', pluginRoutes)).toBeNull()
    expect(appViewForPath('/kanban', pluginRoutes)).toBe('plugin')
    expect(appViewForPath('/achievements', pluginRoutes)).toBe('plugin')
  })

  it('continues to route unknown single-segment paths as chat sessions', () => {
    const pluginRoutes = new Set(['/kanban'])

    expect(routeSessionId('/abc123', pluginRoutes)).toBe('abc123')
    expect(appViewForPath('/abc123', pluginRoutes)).toBe('chat')
  })
})
