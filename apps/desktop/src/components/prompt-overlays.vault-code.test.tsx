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
import { clearAllPrompts, sessionVaultCodeRequest, setVaultCodeRequest } from '@/store/prompts'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

stubResizeObserver()

afterEach(() => {
  cleanup()
  clearAllPrompts()
  _resetSessionOwnerHintsForTests()
  $gateway.set(null)
  vi.clearAllMocks()
})

// The 2FA code goes to the OWNING profile socket as vault.code.respond, whitespace/dashes stripped
// (users paste "246 810" from an SMS); Skip answers "".
it('sends the trimmed code to the owning profile socket', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  setVaultCodeRequest({ hint: '', requestId: 'req-c', sessionId: 'session-a', site: 'github.com' })

  render(<PromptOverlays sessionId="session-a" />)
  expect(document.body.textContent).toContain('Verification code for github.com')
  const input = document.querySelector('input[autocomplete=one-time-code]') as HTMLInputElement
  const submit = document.querySelector('button[type=submit]') as HTMLButtonElement
  expect(submit.disabled).toBe(true)
  fireEvent.change(input, { target: { value: '246 810' } })
  expect(submit.disabled).toBe(false)
  fireEvent.submit(input.closest('form')!)

  await waitFor(() => expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1))
  expect((gatewayMocks.requestGatewayForAgent.mock.calls[0] as unknown[]).slice(0, 4)).toEqual([
    'conn-1',
    'owner',
    'vault.code.respond',
    { code: '246810', request_id: 'req-c' }
  ])
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultCodeRequest('session-a').get()).toBeNull())
})

it('Skip answers an empty code and clears the card', async () => {
  $profiles.set([{ name: 'owner' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  $gateway.set({ request: vi.fn() } as never)
  setVaultCodeRequest({ hint: '', requestId: 'req-d', sessionId: 'session-a', site: 'github.com' })

  render(<PromptOverlays sessionId="session-a" />)
  fireEvent.click(Array.from(document.querySelectorAll('button')).find(b => b.textContent === 'Skip')!)

  await waitFor(() => expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1))
  expect((gatewayMocks.requestGatewayForAgent.mock.calls[0] as unknown[])[3]).toEqual({ code: '', request_id: 'req-d' })
  await waitFor(() => expect(sessionVaultCodeRequest('session-a').get()).toBeNull())
})
