import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import {
  behaviorKeyAction,
  behaviorKeys,
  behaviorRequestPath,
  composerBehavior,
  notifyBehaviorLifecycle
} from './composer-behavior'
const off: (() => void)[] = []

const add = (data: unknown, source = 'plugin:neo-next-prompt') =>
  off.push(registry.register({ id: `test-${off.length}`, source, area: 'composer.behavior.v2', data }))

afterEach(() => {
  while (off.length) {
    off.pop()?.()
  }
})
describe('minimal composer bridge contract', () => {
  it('uses native behavior when absent, wrong protocol, ambiguous or provider throws', () => {
    expect(composerBehavior()).toBeUndefined()
    add({ protocol: 3, create: () => ({}) })
    expect(composerBehavior()).toBeUndefined()
    add({
      protocol: 2,
      create: () => ({}),
      keyAction: () => {
        throw Error('fixture')
      }
    })
    expect(behaviorKeyAction({ key: 'Enter', busy: true })).toBeUndefined()
    add({ protocol: 2, create: () => ({}) })
    expect(composerBehavior()).toBeUndefined()
  })
  it('routes only validated actions and read-only key labels', () => {
    add({ protocol: 2, create: () => ({}), keyAction: () => 'queue', readOnlyKeys: { 'composer.queue': ['enter'] } })
    expect(behaviorKeyAction({ key: 'Enter', busy: true })).toBe('queue')
    expect(behaviorKeys('composer.queue', ['mod+enter'])).toEqual(['enter'])
  })
  it('binds REST namespace to registry provenance and rejects escape/query paths', () => {
    expect(behaviorRequestPath('plugin:neo-next-prompt', '/suggestion')).toBe('/api/plugins/neo-next-prompt/suggestion')

    for (const path of ['/../config', '/%2e%2e/config', '/suggestion?profile=other', 'https://example.invalid']) {
      expect(() => behaviorRequestPath('plugin:neo-next-prompt', path)).toThrow()
    }

    expect(() => behaviorRequestPath('core', '/suggestion')).toThrow()
  })
  it('delivers profile cleanup to the plugin instead of owning its storage', () => {
    const events: unknown[] = []
    add({ protocol: 2, create: () => ({}), lifecycle: (event: unknown) => events.push(event) })
    notifyBehaviorLifecycle({ kind: 'drop', profile: 'default', connectionId: 'ga401' })
    expect(events).toEqual([{ kind: 'drop', profile: 'default', connectionId: 'ga401' }])
  })
})
