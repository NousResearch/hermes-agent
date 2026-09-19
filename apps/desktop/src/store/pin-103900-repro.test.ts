import { expect, it, vi } from 'vitest'
const patch = vi.fn(() => Promise.resolve({ ok: true }))
vi.mock('@/hermes', () => ({ setApiRequestProfile: () => {}, setSessionPinnedRemote: (...a: unknown[]) => (patch as (...x: unknown[]) => unknown)(...a) }))
import { $pinnedSessionIds } from '@/store/layout'
import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'

import { resetSessionPinMirror, watchSessionPins } from './session-pin-sync'
;(globalThis as { window?: unknown }).window ??= {}
;(window as unknown as { hermesDesktop: unknown }).hermesDesktop ??= {}
watchSessionPins()
it('pins converge without a loaded row (103900)', async () => {
  $sessions.set([]); $cronSessions.set([]); $messagingSessions.set([])
  $pinnedSessionIds.set([]); resetSessionPinMirror(); patch.mockClear()
  $pinnedSessionIds.set(['orphan-8c3ea2b76624'])
  await Promise.resolve()
  // Rowless pin still PATCHes (profile falls back to the current gateway's DB).
  expect(patch).toHaveBeenCalledWith('orphan-8c3ea2b76624', true, undefined)
})
