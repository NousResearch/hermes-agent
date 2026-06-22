import { describe, expect, it } from 'vitest'

import { appViewForPath, COMPANION_ROUTE, routeSessionId } from './routes'

describe('desktop routes', () => {
  it('maps the companion route to the companion view', () => {
    expect(appViewForPath(COMPANION_ROUTE)).toBe('companion')
  })

  it('does not treat the companion route as a session id', () => {
    expect(routeSessionId(COMPANION_ROUTE)).toBeNull()
  })
})
