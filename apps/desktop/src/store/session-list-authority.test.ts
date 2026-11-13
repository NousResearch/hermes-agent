import { afterEach, expect, it, vi } from 'vitest'

import { $sessionListAuthority, setSessionListAuthority } from './session-list-authority'

afterEach(() => $sessionListAuthority.set(null))

it('skips unchanged authority writes and publishes each changed field', () => {
  const initial = { profile: 'coder', scope: '.local', epoch: 1, status: 'loaded' as const }
  setSessionListAuthority(initial)
  const notify = vi.fn()
  const off = $sessionListAuthority.listen(notify)
  setSessionListAuthority({ ...initial })
  expect($sessionListAuthority.get()).toBe(initial)
  expect(notify).not.toHaveBeenCalled()
  setSessionListAuthority({ ...initial, profile: 'other' })
  setSessionListAuthority({ ...initial, scope: '.remote' })
  setSessionListAuthority({ ...initial, epoch: 2 })
  setSessionListAuthority({ ...initial, status: 'failed' })
  expect(notify).toHaveBeenCalledTimes(4)
  off()
})
