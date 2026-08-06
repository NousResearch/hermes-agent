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

import { watchSessionPins } from './session-pin-sync'

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
  $sessions.set([])
  $pinnedSessionIds.set([])
  patch.mockClear()
})

afterEach(() => {
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
