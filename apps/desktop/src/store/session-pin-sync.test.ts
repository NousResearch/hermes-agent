import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, pinned: boolean, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  setSessionPinnedRemote: (id: string, pinned: boolean, profile?: null | string) => patch(id, pinned, profile)
}))

import { $pinnedSessionIds } from '@/store/layout'
import { $sessions } from '@/store/session'

import { watchSessionPins, __resetSessionPinSyncForTests } from './session-pin-sync'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

const flush = () => Promise.resolve()

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
  // Attach the listeners once — module state is process-global.
  watchSessionPins()
})

beforeEach(() => {
  __resetSessionPinSyncForTests()
  $sessions.set([])
  $pinnedSessionIds.set([])
  patch.mockClear()
  patch.mockImplementation(() => Promise.resolve({ ok: true }))
})

afterEach(() => {
  __resetSessionPinSyncForTests()
  $sessions.set([])
  $pinnedSessionIds.set([])
})

describe('watchSessionPins', () => {
  it('mirrors a new pin as pinned=true with the row profile', async () => {
    $sessions.set([row('a', { profile: 'work' })])
    $pinnedSessionIds.set(['a'])
    await flush()

    expect(patch).toHaveBeenCalledWith('a', true, 'work')
  })

  it('mirrors an unpin as pinned=false', async () => {
    $sessions.set([row('b')])
    $pinnedSessionIds.set(['b'])
    await flush()
    patch.mockClear()

    $pinnedSessionIds.set([])
    await flush()

    expect(patch).toHaveBeenCalledWith('b', false, undefined)
  })

  it('defers a pin whose row is not loaded, then flushes once it appears', async () => {
    $pinnedSessionIds.set(['c'])
    await flush()
    // No row yet -> nothing sent.
    expect(patch).not.toHaveBeenCalled()

    $sessions.set([row('c', { profile: 'p2' })])
    await flush()

    expect(patch).toHaveBeenCalledWith('c', true, 'p2')
  })

  it('matches a pin id against the lineage root', async () => {
    // pin id is the lineage root; the live row carries it as _lineage_root_id.
    $sessions.set([row('tip', { _lineage_root_id: 'root' })])
    $pinnedSessionIds.set(['root'])
    await flush()

    expect(patch).toHaveBeenCalledWith('root', true, undefined)
  })

  it('does not re-PATCH an already-mirrored pin on unrelated session updates', async () => {
    $sessions.set([row('d')])
    $pinnedSessionIds.set(['d'])
    await flush()
    patch.mockClear()

    // A session-list refresh that doesn't change the pinned set.
    $sessions.set([row('d'), row('e')])
    await flush()

    expect(patch).not.toHaveBeenCalled()
  })
})

describe('watchSessionPins remote pull', () => {
  it('adopts a pin another app made', async () => {
    $sessions.set([row('remote', { pinned: true })])
    await flush()

    expect($pinnedSessionIds.get()).toContain('remote')
  })

  it('adopts a remote pin on the durable lineage root, not the live tip', async () => {
    $sessions.set([row('tip', { _lineage_root_id: 'root', pinned: true })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual(['root'])
  })

  it('does not echo an adopted pin back as a redundant write', async () => {
    $sessions.set([row('adopted', { pinned: true })])
    await flush()

    expect(patch).not.toHaveBeenCalled()
  })

  it('drops a local pin the server reports as unpinned', async () => {
    $pinnedSessionIds.set(['gone'])
    $sessions.set([row('gone', { pinned: true })])
    await flush()
    // Server has confirmed the pin; sticky intent is cleared.
    $sessions.set([row('gone', { pinned: true })])
    await flush()
    patch.mockClear()

    // Another app unpinned it; our next refresh carries the new truth.
    $sessions.set([row('gone', { pinned: false })])
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('gone')
  })

  it('leaves the local set alone when the backend omits the flag', async () => {
    $pinnedSessionIds.set(['legacy'])
    // No `pinned` key at all — a runtime predating the column.
    $sessions.set([row('legacy')])
    await flush()

    expect($pinnedSessionIds.get()).toContain('legacy')
  })

  it('ignores a stale page that contradicts a write still in flight', async () => {
    let settle: (v: { ok: boolean }) => void = () => {}

    patch.mockImplementationOnce(() => new Promise(resolve => (settle = resolve)))

    $sessions.set([row('race')])
    $pinnedSessionIds.set(['race'])
    await flush()
    expect(patch).toHaveBeenCalledWith('race', true, undefined)

    // A list request issued before the PATCH lands still says pinned=false.
    // Honouring it would silently undo the pin the user just made.
    $sessions.set([row('race', { pinned: false })])
    await flush()

    expect($pinnedSessionIds.get()).toContain('race')

    settle({ ok: true })
    await flush()
    await flush()

    // After ack, lagging false pages still must not strip a just-pinned chat.
    $sessions.set([row('race', { pinned: false }), row('other')])
    await flush()
    expect($pinnedSessionIds.get()).toContain('race')

    // Server confirmation clears sticky; later remote unpin can win.
    $sessions.set([row('race', { pinned: true }), row('other')])
    await flush()
    $sessions.set([row('race', { pinned: false }), row('other')])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('race')
  })

  it('does not re-adopt a just-unpinned chat from a still-true server page', async () => {
    // Repro: user unpins while the session list still carries pinned=true.
    // Pull used to run before the unpin write, put the id back, and the
    // sidebar Pinned section never dropped the row.
    $sessions.set([row('sticky', { pinned: true, profile: 'dewey' })])
    $pinnedSessionIds.set(['sticky'])
    await flush()
    $sessions.set([row('sticky', { pinned: true, profile: 'dewey' })])
    await flush()
    patch.mockClear()

    let settle: (v: { ok: boolean }) => void = () => {}
    patch.mockImplementationOnce(() => new Promise(resolve => (settle = resolve)))

    $pinnedSessionIds.set([])
    await flush()

    expect(patch).toHaveBeenCalledWith('sticky', false, 'dewey')
    // Stale list refresh while the unpin PATCH is still in flight.
    $sessions.set([row('sticky', { pinned: true, profile: 'dewey' })])
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('sticky')

    settle({ ok: true })
    await flush()
    // Server still lagging one more tick must not bounce the pin back.
    $sessions.set([row('sticky', { pinned: true, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('sticky')

    // Once the backend agrees, stay unpinned.
    $sessions.set([row('sticky', { pinned: false, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('sticky')
  })

  it('unpins a lineage-root id even when the live tip still reports pinned', async () => {
    $sessions.set([row('tip', { _lineage_root_id: 'root', pinned: true, profile: 'dewey' })])
    $pinnedSessionIds.set(['root'])
    await flush()
    $sessions.set([row('tip', { _lineage_root_id: 'root', pinned: true, profile: 'dewey' })])
    await flush()
    patch.mockClear()

    $pinnedSessionIds.set([])
    await flush()

    expect(patch).toHaveBeenCalledWith('root', false, 'dewey')
    expect($pinnedSessionIds.get()).not.toContain('root')
    expect($pinnedSessionIds.get()).not.toContain('tip')

    // Stale tip page must not re-pin under the durable root.
    $sessions.set([row('tip', { _lineage_root_id: 'root', pinned: true, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('does not strip a just-pinned chat from a still-false server page', async () => {
    // Symmetric race to the sticky-unpin bug: user pins while the list still
    // carries pinned=false. Pull must not delete the local pin before/while
    // the PATCH lands.
    $sessions.set([row('fresh', { pinned: false, profile: 'dewey' })])
    await flush()
    patch.mockClear()

    let settle: (v: { ok: boolean }) => void = () => {}
    patch.mockImplementationOnce(() => new Promise(resolve => (settle = resolve)))

    $pinnedSessionIds.set(['fresh'])
    await flush()

    expect(patch).toHaveBeenCalledWith('fresh', true, 'dewey')
    expect($pinnedSessionIds.get()).toContain('fresh')

    // Stale list refresh while the pin PATCH is still in flight.
    $sessions.set([row('fresh', { pinned: false, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('fresh')

    settle({ ok: true })
    await flush()

    // Still lagging false after ack must not strip the pin.
    $sessions.set([row('fresh', { pinned: false, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('fresh')

    // Server confirmation clears sticky; pin remains via local + server true.
    $sessions.set([row('fresh', { pinned: true, profile: 'dewey' })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('fresh')
  })
})
