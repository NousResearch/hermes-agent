import { describe, expect, it } from 'vitest'

import { ACCOUNT_ROUTE, APP_ROUTES, appViewForPath, WORKFLOW_ROUTE } from './routes'

describe('workflow route', () => {
  it('maps /workflow to the workflow app view', () => {
    expect(WORKFLOW_ROUTE).toBe('/workflow')
    expect(APP_ROUTES).toContainEqual({ id: 'workflow', path: WORKFLOW_ROUTE, view: 'workflow' })
    expect(appViewForPath(WORKFLOW_ROUTE)).toBe('workflow')
  })
})

describe('account route', () => {
  it('maps /account to the EasyHermes account management view', () => {
    expect(ACCOUNT_ROUTE).toBe('/account')
    expect(APP_ROUTES).toContainEqual({ id: 'account', path: ACCOUNT_ROUTE, view: 'account' })
    expect(appViewForPath(ACCOUNT_ROUTE)).toBe('account')
  })
})
