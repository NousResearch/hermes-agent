import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

const gatewayMocks = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn(async () => ({ status: 'ok' }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

import { PromptOverlays } from '@/components/prompt-overlays'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { clearAllPrompts, sessionVaultSaveLoginRequest, setVaultSaveLoginRequest } from '@/store/prompts'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

stubResizeObserver()

afterEach(() => {
  cleanup()
  clearAllPrompts()
  _resetSessionOwnerHintsForTests()
  $gateway.set(null)
  vi.clearAllMocks()
})

// The "save this login" card is the zero-setup path: the pair goes to the OWNING profile's socket as
// one JSON answer, the password field is masked, and Save is disabled until both fields are filled.
it('sends identifier + password as one vault.save_login.respond to the owning profile socket', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  setVaultSaveLoginRequest({
    origin: 'https://github.com',
    requestId: 'req-s',
    sessionId: 'session-a',
    site: 'github.com'
  })

  render(<PromptOverlays sessionId="session-a" />)
  expect(document.body.textContent).toContain('Save your github.com login?')
  const identifier = document.querySelector('input[autocomplete=username]') as HTMLInputElement
  const password = document.querySelector('input[type=password]') as HTMLInputElement
  const submit = document.querySelector('button[type=submit]') as HTMLButtonElement
  expect(submit.disabled).toBe(true)
  fireEvent.change(identifier, { target: { value: 'tek@acme.test' } })
  expect(submit.disabled).toBe(true)
  fireEvent.change(password, { target: { value: 'fixture-pw' } })
  expect(submit.disabled).toBe(false)
  fireEvent.submit(password.closest('form')!)

  await waitFor(() => expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1))
  const [conn, profile, method, params] = gatewayMocks.requestGatewayForAgent.mock.calls[0] as unknown[]
  expect([conn, profile, method]).toEqual(['conn-1', 'owner', 'vault.save_login.respond'])
  expect(JSON.parse((params as { login: string }).login)).toEqual({
    identifier: 'tek@acme.test',
    password: 'fixture-pw'
  })
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultSaveLoginRequest('session-a').get()).toBeNull())
})

it("Don't save answers an empty login and clears the card", async () => {
  $profiles.set([{ name: 'owner' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  $gateway.set({ request: vi.fn() } as never)
  setVaultSaveLoginRequest({
    origin: 'https://github.com',
    requestId: 'req-d',
    sessionId: 'session-a',
    site: 'github.com'
  })

  render(<PromptOverlays sessionId="session-a" />)
  const decline = Array.from(document.querySelectorAll('button')).find(b => b.textContent === "Don't save")!
  fireEvent.click(decline)

  await waitFor(() => expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1))
  expect((gatewayMocks.requestGatewayForAgent.mock.calls[0] as unknown[])[3]).toEqual({
    login: '',
    request_id: 'req-d'
  })
  await waitFor(() => expect(sessionVaultSaveLoginRequest('session-a').get()).toBeNull())
})
