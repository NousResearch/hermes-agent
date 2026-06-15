import { describe, expect, it } from 'vitest'

import { APP_ROUTES, appViewForPath, WORKFLOW_ROUTE } from './routes'

describe('workflow route', () => {
  it('maps /workflow to the workflow app view', () => {
    expect(WORKFLOW_ROUTE).toBe('/workflow')
    expect(APP_ROUTES).toContainEqual({ id: 'workflow', path: WORKFLOW_ROUTE, view: 'workflow' })
    expect(appViewForPath(WORKFLOW_ROUTE)).toBe('workflow')
  })
})
