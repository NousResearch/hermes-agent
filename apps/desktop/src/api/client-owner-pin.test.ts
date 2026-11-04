import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ambientIsLegacyRemote,
  ambientOwnerConnectionId,
  exactOwnerFor,
  sessionFileOwnerScope,
  sessionReadOwnerPin,
  setApiRequestConnection,
  setApiRequestLocalMode,
  setApiRequestProfile,
  setSessionOwnerResolver
} from './client'
import type * as ClientModule from './client'

// The module's local mode starts UNKNOWN (null). Boot-window cells import a
// FRESH module instance so they hold regardless of test order.
async function freshClient(): Promise<typeof ClientModule> {
  vi.resetModules()

  return import('./client')
}

beforeEach(() => {
  setSessionOwnerResolver(null)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

describe('H3/H4 boot window (mode unknown)', () => {
  it('ambientIsLegacyRemote is false while the mode is unknown; ambient id stays undefined', async () => {
    const c = await freshClient()
    expect(c.ambientIsLegacyRemote()).toBe(false)
    expect(c.ambientOwnerConnectionId()).toBeUndefined()
  })

  it('H4 boot branch applies the base read rule: a non-local owner pins, local does not', async () => {
    const c = await freshClient()
    expect(c.ownerConnectionPin({ connectionId: 'conn-r', profile: 'p' })).toBe('conn-r')
    expect(c.ownerConnectionPin({ connectionId: 'local', profile: 'p' })).toBeNull()
    expect(c.ownerConnectionPin(null)).toBeNull()
    expect(c.ownerConnectionPin(undefined)).toBeNull()
  })

  it('sessionReadOwnerPin keeps base behaviour during boot', async () => {
    const c = await freshClient()
    c.setSessionOwnerResolver(() => ({ connectionId: 'conn-r', profile: 'rp' }))
    expect(c.sessionReadOwnerPin('sess-boot')).toEqual({ connectionId: 'conn-r', profile: 'rp' })
    c.setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'lp' }))
    expect(c.sessionReadOwnerPin('sess-boot')).toEqual({})
  })

  it('a mode set on another instance does not leak into a fresh one', async () => {
    setApiRequestLocalMode(false)
    const c = await freshClient()
    expect(c.ambientIsLegacyRemote()).toBe(false)
  })
})

describe('H3 three-state local mode', () => {
  it('tag null + mode false is legacy remote; mode true names local; a tag is never legacy remote', () => {
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    expect(ambientIsLegacyRemote()).toBe(true)
    expect(ambientOwnerConnectionId()).toBeUndefined()
    setApiRequestLocalMode(true)
    expect(ambientIsLegacyRemote()).toBe(false)
    expect(ambientOwnerConnectionId()).toBe('local')
    setApiRequestConnection('reg-R')
    expect(ambientIsLegacyRemote()).toBe(false)
    expect(ambientOwnerConnectionId()).toBe('reg-R')
  })
})

describe('H2 exactOwnerFor', () => {
  it('row rung wins over the resolver; a missing row profile defaults', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'resolver-conn', profile: 'xp' }))
    expect(exactOwnerFor('sess-o', { connection_id: 'row-conn', profile: 'rowprof' })).toEqual({
      connectionId: 'row-conn',
      profile: 'rowprof'
    })
    expect(exactOwnerFor('sess-o', { connection_id: 'row-conn' })).toEqual({
      connectionId: 'row-conn',
      profile: 'default'
    })
    expect(exactOwnerFor('sess-o', { connection_id: '   ', profile: 'rowprof' })).toEqual({
      connectionId: 'resolver-conn',
      profile: 'xp'
    })
  })

  it('resolver rung needs a non-empty trimmed connectionId; bare strings and unknowns yield null', () => {
    setSessionOwnerResolver(() => ({ connectionId: '  resolver-conn  ', profile: ' sp ', targetProfile: ' tp ' }))
    expect(exactOwnerFor('sess-o')).toEqual({ connectionId: 'resolver-conn', profile: 'sp', targetProfile: 'tp' })
    setSessionOwnerResolver(() => ({ profile: 'p-only' }))
    expect(exactOwnerFor('sess-o')).toBeNull()
    setSessionOwnerResolver(() => 'bare-profile')
    expect(exactOwnerFor('sess-o')).toBeNull()
    setSessionOwnerResolver(() => null)
    expect(exactOwnerFor('sess-o')).toBeNull()
    setSessionOwnerResolver(null)
    expect(exactOwnerFor('sess-o')).toBeNull()
  })

  it('T8 lineage root/tip: the row rung matches both id forms', () => {
    const tipRow = { connection_id: 'conn-lin', profile: 'lin' }
    expect(exactOwnerFor('root-id', tipRow)).toEqual({ connectionId: 'conn-lin', profile: 'lin' })
    expect(exactOwnerFor('tip-id', tipRow)).toEqual({ connectionId: 'conn-lin', profile: 'lin' })
  })
})

describe('T2 sessionReadOwnerPin topology table', () => {
  const owners: Record<string, (() => unknown) | null> = {
    none: null,
    bare: () => 'bare-prof',
    local: () => ({ connectionId: 'local', profile: 'lp' }),
    R: () => ({ connectionId: 'reg-R', profile: 'rp' }),
    other: () => ({ connectionId: 'conn-other', profile: 'op' })
  }

  const cases: Array<[string, string, Record<string, string>]> = [
    ['local', 'none', {}],
    ['local', 'bare', {}],
    ['local', 'local', {}],
    ['local', 'R', { connectionId: 'reg-R', profile: 'rp' }],
    ['local', 'other', { connectionId: 'conn-other', profile: 'op' }],
    ['registry', 'none', {}],
    ['registry', 'bare', {}],
    ['registry', 'local', { connectionId: 'local', profile: 'lp' }],
    ['registry', 'R', {}],
    ['registry', 'other', { connectionId: 'conn-other', profile: 'op' }],
    ['legacy', 'none', {}],
    ['legacy', 'bare', {}],
    ['legacy', 'local', { connectionId: 'local', profile: 'lp' }],
    ['legacy', 'R', { connectionId: 'reg-R', profile: 'rp' }],
    ['legacy', 'other', { connectionId: 'conn-other', profile: 'op' }]
  ]

  for (const [primary, ownerKey, expected] of cases) {
    it(`${primary} primary × ${ownerKey} owner ⇒ ${JSON.stringify(expected)}`, () => {
      setApiRequestConnection(primary === 'registry' ? 'reg-R' : null)
      setApiRequestLocalMode(primary === 'local')
      setSessionOwnerResolver(owners[ownerKey] as never)
      expect(sessionReadOwnerPin('sess-t2')).toEqual(expected)
    })
  }

  it('a caller that already pinned a connection object gets {} verbatim (base guard)', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    setApiRequestConnection('reg-R')
    setApiRequestLocalMode(false)
    expect(sessionReadOwnerPin('sess-t2', { connectionId: 'reg-R', profile: 'rp' })).toEqual({})
  })
})

describe('T15 file scope + read pin under legacy remote (D1 read fix)', () => {
  it('legacy remote: a local owner pins reads and file scope to local; local primary unchanged', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'lp' }))
    setApiRequestConnection(null)
    setApiRequestLocalMode(false)
    expect(sessionReadOwnerPin('sess-t15')).toEqual({ connectionId: 'local', profile: 'lp' })
    expect(sessionFileOwnerScope('sess-t15')).toEqual({ connectionId: 'local', profile: 'lp' })
    setApiRequestLocalMode(true)
    expect(sessionReadOwnerPin('sess-t15')).toEqual({})
    expect(sessionFileOwnerScope('sess-t15')).toEqual({ profile: 'lp' })
  })
})

describe('T19 sessionFileOwnerScope keeps the base profile-only branches', () => {
  const shapes: Array<[string, () => unknown, Record<string, string> | null]> = [
    ['bare string', () => 'work ', { profile: 'work' }],
    ['empty string', () => '', null],
    ['profile-only object', () => ({ profile: 'work' }), { profile: 'work' }],
    ['unknown owner', () => null, null]
  ]

  for (const primary of ['local', 'registry', 'legacy'] as const) {
    for (const [name, resolver, expected] of shapes) {
      it(`${primary} primary: ${name} ⇒ ${JSON.stringify(expected)} (base output)`, () => {
        setApiRequestConnection(primary === 'registry' ? 'reg-R' : null)
        setApiRequestLocalMode(primary === 'local')
        setSessionOwnerResolver(resolver as never)
        expect(sessionFileOwnerScope('sess-t19')).toEqual(expected)
      })
    }
  }

  it('object owner with a connection: the pin follows the topology; the base rule shows otherwise', () => {
    setApiRequestConnection('reg-R')
    setApiRequestLocalMode(false)
    setSessionOwnerResolver(() => ({ connectionId: 'conn-other', profile: 'op' }))
    expect(sessionFileOwnerScope('sess-t19')).toEqual({ connectionId: 'conn-other', profile: 'op' })
    setSessionOwnerResolver(() => ({ connectionId: 'reg-R', profile: 'rp' }))
    expect(sessionFileOwnerScope('sess-t19')).toEqual({ profile: 'rp' })
    setSessionOwnerResolver(() => ({ connectionId: 'conn-r', profile: 'rp' }))
    expect(sessionFileOwnerScope('sess-t19')).toEqual({ connectionId: 'conn-r', profile: 'rp' })
    setSessionOwnerResolver(() => ({ connectionId: 'conn-x' }))
    expect(sessionFileOwnerScope('sess-t19')).toBeNull()
  })
})
