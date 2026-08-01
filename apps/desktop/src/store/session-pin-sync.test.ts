import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, pinned: boolean, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  setSessionPinnedRemote: (id: string, pinned: boolean, profile?: null | string) => patch(id, pinned, profile)
}))

import { $pinnedSessionIds } from '@/store/layout'
import { $messagingSessions, $sessions } from '@/store/session'

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
  $messagingSessions.set([])
  $pinnedSessionIds.set([])
  patch.mockClear()
})

afterEach(() => {
  $sessions.set([])
  $messagingSessions.set([])
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

  it('flushes a pending pin from a messaging row with its profile', async () => {
    $pinnedSessionIds.set(['pending-message'])
    await flush()
    expect(patch).not.toHaveBeenCalled()

    $messagingSessions.set([row('pending-message', { profile: 'chat-profile', source: 'telegram' })])
    await flush()

    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith('pending-message', true, 'chat-profile')
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

  it('writes once when both stores represent the same durable conversation', async () => {
    $sessions.set([row('local-tip', { _lineage_root_id: 'shared-pin' })])
    $messagingSessions.set([row('message-tip', { _lineage_root_id: 'shared-pin', source: 'telegram' })])
    $pinnedSessionIds.set(['shared-pin'])
    await flush()

    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith('shared-pin', true, undefined)
  })

  it('resolves a pending tip id after its lineage is deduplicated', async () => {
    $pinnedSessionIds.set(['legacy-tip'])
    $sessions.set([row('legacy-root', { profile: 'root-profile' })])
    $messagingSessions.set([
      row('legacy-tip', {
        _lineage_root_id: 'legacy-root',
        profile: 'message-profile',
        source: 'telegram'
      })
    ])
    await flush()

    expect(patch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledWith('legacy-tip', true, 'message-profile')
  })
})

describe('watchSessionPins remote pull', () => {
  it('adopts a messaging pin on its durable root without echoing a PATCH', async () => {
    $messagingSessions.set([row('message-tip', { _lineage_root_id: 'message-root', pinned: true, source: 'telegram' })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual(['message-root'])
    expect(patch).not.toHaveBeenCalled()
  })

  it('deduplicates a split lineage without discarding its authoritative pin', async () => {
    $sessions.set([row('shared-root')])
    $messagingSessions.set([row('shared-tip', { _lineage_root_id: 'shared-root', pinned: true, source: 'telegram' })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual(['shared-root'])
    expect(patch).not.toHaveBeenCalled()
  })

  it('defers remote reconciliation when duplicate rows carry conflicting pin values', async () => {
    $sessions.set([row('conflict-tip', { _lineage_root_id: 'conflict-root', pinned: true })])
    $messagingSessions.set([row('conflict-root', { pinned: false, source: 'telegram' })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual(['conflict-root'])
    expect(patch).not.toHaveBeenCalled()
  })

  it('leaves an unpinned renderer unpinned when conflicting rows arrive in reverse order', async () => {
    $messagingSessions.set([row('reverse-root', { pinned: false, source: 'telegram' })])
    $sessions.set([row('reverse-tip', { _lineage_root_id: 'reverse-root', pinned: true })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual([])
    expect(patch).not.toHaveBeenCalled()
  })

  it('reconciles when either session slice updates', async () => {
    $sessions.set([row('local-remote', { pinned: true })])
    await flush()
    expect($pinnedSessionIds.get()).toEqual(['local-remote'])

    $messagingSessions.set([row('message-remote', { pinned: true, source: 'telegram' })])
    await flush()

    expect($pinnedSessionIds.get()).toEqual(['local-remote', 'message-remote'])
    expect(patch).not.toHaveBeenCalled()
  })

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

  it('guards a messaging pin from a stale page while its write is in flight', async () => {
    let settle: (v: { ok: boolean }) => void = () => {}

    patch.mockImplementationOnce(() => new Promise(resolve => (settle = resolve)))

    $messagingSessions.set([row('message-race-tip', { _lineage_root_id: 'message-race-root', source: 'telegram' })])
    $pinnedSessionIds.set(['message-race-tip'])
    await flush()
    expect(patch).toHaveBeenCalledWith('message-race-tip', true, undefined)

    $messagingSessions.set([
      row('message-race-tip', {
        _lineage_root_id: 'message-race-root',
        pinned: false,
        source: 'telegram'
      })
    ])
    await flush()
    expect($pinnedSessionIds.get()).toContain('message-race-tip')

    settle({ ok: true })
    await flush()
    await flush()

    $messagingSessions.set([
      row('message-race-tip', {
        _lineage_root_id: 'message-race-root',
        pinned: false,
        source: 'telegram'
      }),
      row('another-message', { source: 'telegram' })
    ])
    await flush()

    expect($pinnedSessionIds.get()).not.toContain('message-race-tip')
  })
})
