import { describe, expect, it } from 'vitest'

import {
  normalizeHermesOpenString,
  pathFromHermesDeepLink,
  pathFromOpenDeepLink,
  resolveHermesOpenPath
} from './hermes-open-target'

describe('normalizeHermesOpenString', () => {
  it('accepts hash-router paths and strips a leading hash', () => {
    expect(normalizeHermesOpenString('/index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeHermesOpenString('#/index-network/intent/1')).toBe('/index-network/intent/1')
  })

  it('maps plugin-scoped hermes:// deep links to the same path', () => {
    expect(normalizeHermesOpenString('hermes://index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeHermesOpenString('hermes://index-network/intent/1?focus=true')).toBe(
      '/index-network/intent/1?focus=true'
    )
  })

  it('maps hermes://open/… deep links by stripping the open host', () => {
    expect(normalizeHermesOpenString('hermes://open/index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeHermesOpenString('hermes://open/settings/plugins')).toBe('/settings/plugins')
  })

  it('rejects reserved hermes kinds and unsafe paths', () => {
    expect(normalizeHermesOpenString('hermes://blueprint/morning-brief')).toBeNull()
    expect(normalizeHermesOpenString('hermes://plugin/install')).toBeNull()
    expect(normalizeHermesOpenString('https://example.com/x')).toBeNull()
    expect(normalizeHermesOpenString('/../etc/passwd')).toBeNull()
    expect(normalizeHermesOpenString('index-network')).toBeNull()
  })
})

describe('resolveHermesOpenPath', () => {
  it('merges structured path + params', () => {
    expect(resolveHermesOpenPath({ path: '/index-network/intent/1', params: { focus: 'true' } })).toBe(
      '/index-network/intent/1?focus=true'
    )
  })

  it('resolves href the same as a bare string', () => {
    expect(resolveHermesOpenPath({ href: 'hermes://index-network/intent/1' })).toBe('/index-network/intent/1')
  })
})

describe('pathFromHermesDeepLink', () => {
  it('builds the navigate path from a plugin-scoped deep-link payload', () => {
    expect(pathFromHermesDeepLink('index-network', 'intent/1')).toBe('/index-network/intent/1')
  })

  it('builds the navigate path from hermes://open/… payloads', () => {
    expect(pathFromOpenDeepLink('index-network/intent/1')).toBe('/index-network/intent/1')
    expect(pathFromHermesDeepLink('open', 'agent/42')).toBe('/agent/42')
  })

  it('ignores reserved kinds', () => {
    expect(pathFromHermesDeepLink('blueprint', 'morning-brief')).toBeNull()
    expect(pathFromHermesDeepLink('plugin', 'install')).toBeNull()
  })
})
