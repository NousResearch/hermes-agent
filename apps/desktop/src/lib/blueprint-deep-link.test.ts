import { describe, expect, it } from 'vitest'

import { buildBlueprintDeepLinkCommand } from './blueprint-deep-link'

describe('buildBlueprintDeepLinkCommand', () => {
  it('builds a command from a valid blueprint payload', () => {
    expect(
      buildBlueprintDeepLinkCommand({
        kind: 'blueprint',
        name: 'morning-brief',
        params: { time: '08:00', label: 'daily brief' }
      })
    ).toBe('/blueprint morning-brief time=08:00 label="daily brief"')
  })

  it('sanitizes the name and rejects names that are empty after sanitization', () => {
    expect(buildBlueprintDeepLinkCommand({ kind: 'blueprint', name: '../bad name', params: {} })).toBe(
      '/blueprint badname'
    )
    expect(buildBlueprintDeepLinkCommand({ kind: 'blueprint', name: '\u{1F4A5}', params: {} })).toBeNull()
  })

  it('drops unsafe slot keys, quotes spaced values, and strips control characters', () => {
    expect(
      buildBlueprintDeepLinkCommand({
        kind: 'blueprint',
        name: 'daily',
        params: {
          ok_key: 'hello\n/evil "quoted"',
          'bad key': 'nope',
          'bad=key': 'nope',
          good2: 'plain\u0000value'
        }
      })
    ).toBe('/blueprint daily ok_key="hello/evil \\"quoted\\"" good2=plainvalue')
  })

  it('returns null for non-blueprint or malformed payloads', () => {
    expect(buildBlueprintDeepLinkCommand(null)).toBeNull()
    expect(buildBlueprintDeepLinkCommand({ kind: 'other', name: 'x', params: {} })).toBeNull()
    expect(buildBlueprintDeepLinkCommand('nope')).toBeNull()
  })
})
