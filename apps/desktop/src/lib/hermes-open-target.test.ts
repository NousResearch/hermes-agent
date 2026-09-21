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

  it('maps plugin-scoped app-protocol deep links to the same path', () => {
    expect(normalizeHermesOpenString('aievolution-jarvis://index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeHermesOpenString('aievolution-jarvis://index-network/intent/1?focus=true')).toBe(
      '/index-network/intent/1?focus=true'
    )
  })

  it('maps app-protocol open deep links by stripping the open host', () => {
    expect(normalizeHermesOpenString('aievolution-jarvis://open/index-network/intent/1')).toBe(
      '/index-network/intent/1'
    )
    expect(normalizeHermesOpenString('aievolution-jarvis://open/settings/plugins')).toBe('/settings/plugins')
  })

  it('rejects reserved app-protocol kinds, unrelated schemes, and unsafe paths', () => {
    expect(normalizeHermesOpenString('aievolution-jarvis://blueprint/morning-brief')).toBeNull()
    expect(normalizeHermesOpenString('aievolution-jarvis://plugin/install')).toBeNull()
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
    expect(resolveHermesOpenPath({ href: 'aievolution-jarvis://index-network/intent/1' })).toBe(
      '/index-network/intent/1'
    )
  })
})

describe('pathFromHermesDeepLink', () => {
  it('builds the navigate path from a plugin-scoped deep-link payload', () => {
    expect(pathFromHermesDeepLink('index-network', 'intent/1')).toBe('/index-network/intent/1')
  })

  it('builds the navigate path from app-protocol open payloads', () => {
    expect(pathFromOpenDeepLink('index-network/intent/1')).toBe('/index-network/intent/1')
    expect(pathFromHermesDeepLink('open', 'agent/42')).toBe('/agent/42')
  })

  it('ignores reserved kinds', () => {
    expect(pathFromHermesDeepLink('blueprint', 'morning-brief')).toBeNull()
    expect(pathFromHermesDeepLink('plugin', 'install')).toBeNull()
  })
})
