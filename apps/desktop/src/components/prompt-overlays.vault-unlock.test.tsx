import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { PromptOverlays } from '@/components/prompt-overlays'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { clearAllPrompts, sessionVaultUnlockRequest, setVaultUnlockRequest } from '@/store/prompts'
import { hasOpenServerRequest, rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

stubResizeObserver()

beforeEach(() => {
  resetServerRequestsForTests()
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  resetServerRequestsForTests()
  _resetSessionOwnerHintsForTests()
  $gateway.set(null)
  vi.clearAllMocks()
})

// A master password typed into a background profile's unlock card must reach the
// backend that raised the prompt. The window's ambient `$gateway` may be another
// profile's socket entirely; sending the password there is a cross-backend leak.
// The answer is the `vault.unlock` server request's own response frame — it rides
// the socket the request arrived on, so the ambient gateway is never dialled.
it('answers the vault.unlock server request with the master password, never the ambient gateway', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  const respond = vi.fn()
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  rememberServerRequest({ fail: vi.fn(), id: 'req-a', method: 'vault.unlock', params: {}, respond })
  setVaultUnlockRequest({ backend: 'bitwarden', displayName: 'Bitwarden', requestId: 'req-a', sessionId: 'session-a' })

  render(<PromptOverlays sessionId="session-a" />)
  const input = document.querySelector('input[type=password]')!
  fireEvent.change(input, { target: { value: 'fixture-master' } })
  fireEvent.submit(input.closest('form')!)

  await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
  expect(respond).toHaveBeenCalledWith({ value: 'fixture-master' })
  expect(hasOpenServerRequest('req-a')).toBe(false)
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultUnlockRequest('session-a').get()).toBeNull())
})
