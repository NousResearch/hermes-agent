import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { PromptOverlays } from '@/components/prompt-overlays'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { clearAllPrompts, sessionVaultSaveLoginRequest, setVaultSaveLoginRequest } from '@/store/prompts'
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

// The "save this login" card is the zero-setup path: the pair answers the `vault.save_login`
// server request as one JSON value (the frame rides the socket the request arrived on, never
// the ambient gateway), the password field is masked, and Save is disabled until both fields
// are filled.
it('answers the vault.save_login server request with identifier + password as one JSON value', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  const respond = vi.fn()
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  rememberServerRequest({ fail: vi.fn(), id: 'req-s', method: 'vault.save_login', params: {}, respond })
  setVaultSaveLoginRequest({
    origin: 'https://github.com',
    requestId: 'req-s',
    sessionId: 'session-a',
    site: 'github.com'
  })

  render(<PromptOverlays sessionId="session-a" />)
  const identifier = document.querySelector('input[autocomplete=username]') as HTMLInputElement
  const password = document.querySelector('input[type=password]') as HTMLInputElement
  const submit = document.querySelector('button[type=submit]') as HTMLButtonElement
  expect(submit.disabled).toBe(true)
  fireEvent.change(identifier, { target: { value: 'tek@acme.test' } })
  expect(submit.disabled).toBe(true)
  fireEvent.change(password, { target: { value: 'fixture-pw' } })
  expect(submit.disabled).toBe(false)
  fireEvent.submit(password.closest('form')!)

  await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
  const [result] = respond.mock.calls[0] as [{ value: string }]
  expect(JSON.parse(result.value)).toEqual({
    identifier: 'tek@acme.test',
    password: 'fixture-pw'
  })
  expect(hasOpenServerRequest('req-s')).toBe(false)
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultSaveLoginRequest('session-a').get()).toBeNull())
})

it("Don't save answers an empty login and clears the card", async () => {
  $profiles.set([{ name: 'owner' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const respond = vi.fn()
  $gateway.set({ request: vi.fn() } as never)
  rememberServerRequest({ fail: vi.fn(), id: 'req-d', method: 'vault.save_login', params: {}, respond })
  setVaultSaveLoginRequest({
    origin: 'https://github.com',
    requestId: 'req-d',
    sessionId: 'session-a',
    site: 'github.com'
  })

  render(<PromptOverlays sessionId="session-a" />)
  const decline = Array.from(document.querySelectorAll('button')).find(b => b.textContent === "Don't save")!
  fireEvent.click(decline)

  await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
  expect(respond).toHaveBeenCalledWith({ value: '' })
  await waitFor(() => expect(sessionVaultSaveLoginRequest('session-a').get()).toBeNull())
})
