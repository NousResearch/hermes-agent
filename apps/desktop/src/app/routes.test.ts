import { describe, expect, it } from 'vitest'

import { routeSessionProfile, sessionRoute } from './routes'

describe('profile-routed session paths', () => {
  it('round-trips the owning profile through the route query', () => {
    const route = sessionRoute('telegram/session', 'ubuntu server')
    const [, search = ''] = route.split('?')

    expect(route).toBe('/telegram%2Fsession?profile=ubuntu%20server')
    expect(routeSessionProfile(`?${search}`)).toBe('ubuntu server')
  })

  it('keeps legacy routes unchanged when no profile is supplied', () => {
    expect(sessionRoute('session-1')).toBe('/session-1')
    expect(routeSessionProfile('')).toBeNull()
    expect(routeSessionProfile('?profile=%20%20')).toBeNull()
  })
})
