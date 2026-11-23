import type { WritableAtom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { installDockBadge, needsInputSessionCount } from './dock-badge'
import { $messagingSessions, $sessions } from './session'
import { $attentionSessionIds, $workingSessionIds } from './session-states'
import type * as SessionStates from './session-states'

vi.mock('./session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStates>()
  const { atom } = await import('nanostores')

  return { ...actual, $attentionSessionIds: atom<string[]>([]), $workingSessionIds: atom<string[]>([]) }
})

// The mock swaps these for writable atoms; the real ones are computed.
const attention = $attentionSessionIds as unknown as WritableAtom<string[]>
const working = $workingSessionIds as unknown as WritableAtom<string[]>

const listed = (id: string, extra: Partial<SessionInfo> = {}) => ({ id, ...extra }) as unknown as SessionInfo

afterEach(() => {
  attention.set([])
  working.set([])
  $sessions.set([])
  $messagingSessions.set([])
})

describe('needsInputSessionCount', () => {
  it('counts listed rows waiting on the owner, once each, skipping archived rows', () => {
    const byId = { a: 'needs-input', alias: 'needs-input', b: 'working', c: 'needs-input', d: 'unread' } as const

    expect(
      needsInputSessionCount(byId, [listed('a'), listed('b'), listed('c', { archived: true })], [listed('d')])
    ).toBe(1)
    // A row listed twice (recents and messaging) is still one session.
    expect(needsInputSessionCount(byId, [listed('a')], [listed('a')])).toBe(1)
  })
})

describe('installDockBadge', () => {
  it('sets the Dock badge to the number of sessions waiting on you, and 0 clears it', () => {
    const setCount = vi.fn()
    $sessions.set([listed('s1'), listed('s2'), listed('s3')])
    const stop = installDockBadge(setCount, '')

    expect(setCount).toHaveBeenLastCalledWith(0)

    attention.set(['s1', 's3'])
    expect(setCount).toHaveBeenLastCalledWith(2)

    // A working session is not waiting on anyone.
    working.set(['s2'])
    expect(setCount).toHaveBeenLastCalledWith(2)

    attention.set([])
    expect(setCount).toHaveBeenLastCalledWith(0)

    stop()
    attention.set(['s1'])
    expect(setCount).toHaveBeenLastCalledWith(0)
  })

  it('leaves the badge to the main window: secondary windows and hosts without the bridge do nothing', () => {
    const setCount = vi.fn()
    $sessions.set([listed('s1')])
    attention.set(['s1'])

    installDockBadge(setCount, '?win=secondary')()
    installDockBadge(undefined, '')()

    expect(setCount).not.toHaveBeenCalled()
  })
})
