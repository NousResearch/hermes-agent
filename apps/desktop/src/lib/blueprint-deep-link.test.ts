import { describe, expect, it } from 'vitest'

import { buildBlueprintDeepLinkCommand } from './blueprint-deep-link'

describe('buildBlueprintDeepLinkCommand', () => {
  it('formats valid blueprint deep links for composer insertion', () => {
    expect(
      buildBlueprintDeepLinkCommand({
        kind: 'blueprint',
        name: 'morning-brief',
        params: {
          time: '08:00',
          label: 'daily brief'
        }
      })
    ).toBe('/blueprint morning-brief time=08:00 label="daily brief"')
  })

  it('drops unsafe names and rejects empty names after sanitization', () => {
    expect(buildBlueprintDeepLinkCommand({ kind: 'blueprint', name: '../bad name', params: {} })).toBe(
      '/blueprint badname'
    )
    expect(buildBlueprintDeepLinkCommand({ kind: 'blueprint', name: '💥', params: {} })).toBeNull()
  })

  it('skips unsafe parameter keys and strips control characters from values', () => {
    expect(
      buildBlueprintDeepLinkCommand({
        kind: 'blueprint',
        name: 'daily',
        params: {
          ok_key: 'hello\n/evil "quoted"',
          'bad key': 'should-not-appear',
          'bad=key': 'should-not-appear',
          good2: 'plain\u0000value'
        }
      })
    ).toBe('/blueprint daily ok_key="hello/evil \\"quoted\\"" good2=plainvalue')
  })

  it('ignores non-blueprint payloads', () => {
    expect(buildBlueprintDeepLinkCommand(null)).toBeNull()
    expect(buildBlueprintDeepLinkCommand({ kind: 'other', name: 'daily', params: {} })).toBeNull()
  })
})
