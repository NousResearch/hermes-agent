import { describe, expect, it } from 'vitest'

import { appViewForPath, KANBAN_ROUTE, routeSessionId } from './routes'

describe('desktop routes', () => {
  it('treats the Kanban dashboard plugin route as a reserved app view', () => {
    expect(KANBAN_ROUTE).toBe('/kanban')
    expect(appViewForPath(KANBAN_ROUTE)).toBe('kanban')
    expect(routeSessionId(KANBAN_ROUTE)).toBeNull()
  })
})
