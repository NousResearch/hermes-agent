import { describe, expect, it } from 'vitest'

import {
  normalizeRabbitOpenString,
  pathFromRabbitDeepLink,
  pathFromOpenDeepLink,
  resolveRabbitOpenPath
} from './rabbit-open-target'

describe('normalizeRabbitOpenString', () => {
  it('accepts hash-router paths and strips a leading hash', () => {
    expect(normalizeRabbitOpenString('/index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeRabbitOpenString('#/index-network/intent/1')).toBe('/index-network/intent/1')
  })

  it('maps plugin-scoped rabbit:// deep links to the same path', () => {
    expect(normalizeRabbitOpenString('rabbit://index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeRabbitOpenString('rabbit://index-network/intent/1?focus=true')).toBe(
      '/index-network/intent/1?focus=true'
    )
  })

  it('maps rabbit://open/… deep links by stripping the open host', () => {
    expect(normalizeRabbitOpenString('rabbit://open/index-network/intent/1')).toBe('/index-network/intent/1')
    expect(normalizeRabbitOpenString('rabbit://open/settings/plugins')).toBe('/settings/plugins')
  })

  it('rejects reserved rabbit kinds and unsafe paths', () => {
    expect(normalizeRabbitOpenString('rabbit://blueprint/morning-brief')).toBeNull()
    expect(normalizeRabbitOpenString('rabbit://plugin/install')).toBeNull()
    expect(normalizeRabbitOpenString('https://example.com/x')).toBeNull()
    expect(normalizeRabbitOpenString('/../etc/passwd')).toBeNull()
    expect(normalizeRabbitOpenString('index-network')).toBeNull()
  })
})

describe('resolveRabbitOpenPath', () => {
  it('merges structured path + params', () => {
    expect(resolveRabbitOpenPath({ path: '/index-network/intent/1', params: { focus: 'true' } })).toBe(
      '/index-network/intent/1?focus=true'
    )
  })

  it('resolves href the same as a bare string', () => {
    expect(resolveRabbitOpenPath({ href: 'rabbit://index-network/intent/1' })).toBe('/index-network/intent/1')
  })
})

describe('pathFromRabbitDeepLink', () => {
  it('builds the navigate path from a plugin-scoped deep-link payload', () => {
    expect(pathFromRabbitDeepLink('index-network', 'intent/1')).toBe('/index-network/intent/1')
  })

  it('builds the navigate path from rabbit://open/… payloads', () => {
    expect(pathFromOpenDeepLink('index-network/intent/1')).toBe('/index-network/intent/1')
    expect(pathFromRabbitDeepLink('open', 'agent/42')).toBe('/agent/42')
  })

  it('ignores reserved kinds', () => {
    expect(pathFromRabbitDeepLink('blueprint', 'morning-brief')).toBeNull()
    expect(pathFromRabbitDeepLink('plugin', 'install')).toBeNull()
  })
})
