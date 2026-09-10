import { describe, expect, it } from 'vitest'

import { namespaceExportNames } from './runtime'

describe('runtime SDK namespace preparation', () => {
  it('tolerates an uninitialized namespace from a production import cycle', () => {
    expect(namespaceExportNames(undefined)).toEqual([])
    expect(namespaceExportNames(null)).toEqual([])
  })

  it('preserves valid named exports while excluding default', () => {
    expect(namespaceExportNames({ default: {}, host: {}, cn: () => undefined })).toEqual(['host', 'cn'])
  })
})
