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
