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

import { flushSessionPinWrites, watchSessionPins } from './session-pin-sync'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

// Drain the microtask queue fully: per-id write chains settle across several
// hops, so a bare Promise.resolve() no longer flushes a write to completion.
const flush = () => new Promise<void>(resolve => setTimeout(resolve, 0))

beforeAll(() => {
  ;(globalThis as { window?: unknown }).window ??= {}
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {}
  // Attach the listeners once — module state is process-global.
  watchSessionPins()
})

beforeEach(() => {
  $sessions.set([])
  $pinnedSessionIds.set([])
  patch.mockClear()
})

afterEach(async () => {
  $sessions.set([])
  $pinnedSessionIds.set([])
  await flush()
  // A deferred re-assert (skipped while its chain was in flight) only fires on
  // a later $sessions change; poke one more pass so nothing lands inside the
  // next test's window (module-level mirrored/pending/unpinPending persist).
  $sessions.set([row('__teardown__')])
  await flush()
  $sessions.set([])
  await flush()
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

  it('does not revert a fresh local pin while the loaded row is still stale (#74570)', async () => {
    // The row is already loaded and says pinned=false when the user pins.
    // The pin listener fires reconcile synchronously — before any PATCH — and
    // the stale row must not win over the local intent.
    $sessions.set([row('fresh', { pinned: false })])
    await flush()
    patch.mockClear()

    $pinnedSessionIds.set(['fresh'])
    await flush()

    expect($pinnedSessionIds.get()).toContain('fresh')
    expect(patch).toHaveBeenCalledWith('fresh', true, undefined)
  })

  it('does not revert a fresh local unpin while the loaded row still says pinned (#74570)', async () => {
    // Adopt a server-side pin first, so it's held locally and mirrored.
    $sessions.set([row('sticky', { pinned: true })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('sticky')
    patch.mockClear()

    // User unpins while the loaded row still says pinned=true.
    $pinnedSessionIds.set([])
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('sticky')
    expect(patch).toHaveBeenCalledWith('sticky', false, undefined)
  })

  it('keeps a deferred pin (row not yet loaded) when a stale page finally arrives', async () => {
    $pinnedSessionIds.set(['deferred'])
    await flush()
    expect(patch).not.toHaveBeenCalled()

    // The page that loads the row still predates our intent.
    $sessions.set([row('deferred', { pinned: false })])
    await flush()

    expect($pinnedSessionIds.get()).toContain('deferred')
    expect(patch).toHaveBeenCalledWith('deferred', true, undefined)
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

    // Once the write is acked, later server truth is honoured again.
    settle({ ok: true })
    await flush()
    await flush()

    $sessions.set([row('race', { pinned: false }), row('other')])
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('race')
  })
})

describe('watchSessionPins write-failure resilience', () => {
  it('retries a failed unpin and does not let the next page re-pin it', async () => {
    // Adopt a server-side pin so the unpin has something to mirror.
    $sessions.set([row('sticky', { pinned: true })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('sticky')
    patch.mockClear()

    // Unpin, but the backend rejects the PATCH (transient failure).
    patch.mockImplementationOnce(() => Promise.reject(new Error('network')))
    $pinnedSessionIds.set([])
    await flush()

    // A page that still says pinned=true must not resurrect the pin while the
    // unpin is unconfirmed — previously the swallowed failure let the pull
    // re-pin it, and the next PATCH(pinned=true) made the unpin stick-forever.
    $sessions.set([row('sticky', { pinned: true })])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('sticky')

    // The unpin is re-asserted rather than dropped.
    await flush()
    expect(patch).toHaveBeenCalledWith('sticky', false, undefined)

    // Once the server page confirms pinned=false the retry retires for good.
    $sessions.set([row('sticky', { pinned: false })])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('sticky')
    expect(patch).not.toHaveBeenCalledWith('sticky', true, undefined)
  })

  it('serializes per-session writes so a quick pin->unpin cannot land out of order', async () => {
    let settlePin: (v: { ok: boolean }) => void = () => {}
    $sessions.set([row('s')])
    patch.mockImplementationOnce(() => new Promise(resolve => (settlePin = resolve)))

    $pinnedSessionIds.set(['s']) // pin
    await flush()
    expect(patch).toHaveBeenCalledWith('s', true, undefined)

    patch.mockClear()
    $pinnedSessionIds.set([]) // immediate unpin while the pin PATCH is in flight
    await flush()

    // The unpin must be queued behind the pin, not fired in parallel — two
    // parallel PATCHes can arrive swapped and leave the pin in place.
    expect(patch).not.toHaveBeenCalled()

    settlePin({ ok: true }) // pin write lands
    await flush()
    await flush()

    // Only now does the unpin PATCH go out — after the pin, so the server
    // ends unpinned regardless of arrival order.
    expect(patch).toHaveBeenCalledWith('s', false, undefined)
  })

  it('keeps the newest pin fenced through a pin->unpin->pin ABA sequence', async () => {
    const held: Array<(v: { ok: boolean }) => void> = []
    $sessions.set([row('aba')])
    // Hold every 'aba' write so the queue stays in flight on demand.
    patch.mockImplementation((id: string) => {
      if (id === 'aba') {return new Promise(resolve => held.push(resolve))}

      return Promise.resolve({ ok: true })
    })

    $pinnedSessionIds.set(['aba']) // pin #1 (held)
    await flush()
    expect(patch).toHaveBeenCalledWith('aba', true, undefined)

    $pinnedSessionIds.set([]) // unpin #2 (queued)
    $pinnedSessionIds.set(['aba']) // pin #3 (queued)
    await flush()
    expect(patch).toHaveBeenCalledTimes(1) // still only the first write issued

    // Pin #1 lands; #2 and #3 are still queued behind it. The ABA hazard: the
    // settling write carries the same value as the newest queued write, so a
    // boolean-only fence would clear the newer write's guard here.
    held.shift()!({ ok: true })
    await flush()

    // A stale page that predates the newest pin must not revert it while any
    // write for the id is still queued.
    $sessions.set([row('aba', { pinned: false })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('aba')

    // Drain the remaining writes; the final durable intent is pinned. Each
    // settled write lets the next queued write issue on a later microtask, so
    // flush between releases.
    held.shift()!({ ok: true }) // #2 settles
    await flush() // #3 issues
    held.shift()!({ ok: true }) // #3 settles
    await flush()
    await flush()

    expect($pinnedSessionIds.get()).toContain('aba')
    const order = patch.mock.calls.map(c => [c[0], c[1]]).filter(c => c[0] === 'aba')
    expect(order).toEqual([
      ['aba', true],
      ['aba', false],
      ['aba', true]
    ])
  })

  it('retains the owning profile for an unpin whose row briefly leaves the list', async () => {
    $sessions.set([row('wander', { pinned: true, profile: 'work' })])
    await flush()
    expect($pinnedSessionIds.get()).toContain('wander')
    patch.mockClear()

    // Unpin, but the backend rejects the PATCH.
    patch.mockImplementationOnce(() => Promise.reject(new Error('network')))
    $pinnedSessionIds.set([])
    await flush()
    patch.mockClear()

    // The row disappears (profile switch / list-scope refresh) before any
    // confirmation. The intent must survive with the captured profile and be
    // re-asserted against the right backend, not dropped as if confirmed.
    $sessions.set([])
    await flush()
    expect(patch).toHaveBeenCalledWith('wander', false, 'work')

    // Back on the original profile, the row confirms pinned=false.
    $sessions.set([row('wander', { pinned: false, profile: 'work' })])
    await flush()
    expect($pinnedSessionIds.get()).not.toContain('wander')
  })

  it('flushSessionPinWrites resolves only once in-flight writes for the id settle', async () => {
    let settlePin: (v: { ok: boolean }) => void = () => {}
    $sessions.set([row('f')])
    patch.mockImplementationOnce(() => new Promise(resolve => (settlePin = resolve)))

    $pinnedSessionIds.set(['f'])
    await flush()

    let flushed = false
    void flushSessionPinWrites('f').then(() => {
      flushed = true
    })
    await flush()
    expect(flushed).toBe(false) // still in flight

    settlePin({ ok: true })
    await flush()
    await flush()
    expect(flushed).toBe(true)
  })
})
