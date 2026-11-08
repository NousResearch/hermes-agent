import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { PromptOverlays } from '@/components/prompt-overlays'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { clearAllPrompts, sessionVaultCodeRequest, setVaultCodeRequest } from '@/store/prompts'
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

// The 2FA code answers the `vault.code` server request (the frame rides the socket the
// request arrived on, never the ambient gateway), whitespace/dashes stripped (users paste
// "246 810" from an SMS); Skip answers "".
it('answers the vault.code server request with the trimmed code, never the ambient gateway', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  const respond = vi.fn()
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  rememberServerRequest({ fail: vi.fn(), id: 'req-c', method: 'vault.code', params: {}, respond })
  setVaultCodeRequest({ hint: '', requestId: 'req-c', sessionId: 'session-a', site: 'github.com' })

  render(<PromptOverlays sessionId="session-a" />)
  const input = document.querySelector('input[autocomplete=one-time-code]') as HTMLInputElement
  const submit = document.querySelector('button[type=submit]') as HTMLButtonElement
  expect(submit.disabled).toBe(true)
  fireEvent.change(input, { target: { value: '246 810' } })
  expect(submit.disabled).toBe(false)
  fireEvent.submit(input.closest('form')!)

  await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
  expect(respond).toHaveBeenCalledWith({ value: '246810' })
  expect(hasOpenServerRequest('req-c')).toBe(false)
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultCodeRequest('session-a').get()).toBeNull())
})

it('Skip answers an empty code and clears the card', async () => {
  $profiles.set([{ name: 'owner' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const respond = vi.fn()
  $gateway.set({ request: vi.fn() } as never)
  rememberServerRequest({ fail: vi.fn(), id: 'req-d', method: 'vault.code', params: {}, respond })
  setVaultCodeRequest({ hint: '', requestId: 'req-d', sessionId: 'session-a', site: 'github.com' })

  render(<PromptOverlays sessionId="session-a" />)
  fireEvent.click(Array.from(document.querySelectorAll('button')).find(b => b.textContent === 'Skip')!)

  await waitFor(() => expect(respond).toHaveBeenCalledTimes(1))
  expect(respond).toHaveBeenCalledWith({ value: '' })
  await waitFor(() => expect(sessionVaultCodeRequest('session-a').get()).toBeNull())
})
