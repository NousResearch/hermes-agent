import { describe, expect, it } from 'vitest'
import { appExecutionAliasExtensions } from './before-build.mjs'

describe('MSIX alias population', () => {
  it('omits execution aliases when no payload launchers are present', () => {
    expect(appExecutionAliasExtensions([])).toBe('')
  })
})
