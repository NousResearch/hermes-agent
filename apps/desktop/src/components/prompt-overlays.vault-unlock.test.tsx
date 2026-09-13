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
import { clearAllPrompts, sessionVaultUnlockRequest, setVaultUnlockRequest } from '@/store/prompts'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

stubResizeObserver()

afterEach(() => {
  cleanup()
  clearAllPrompts()
  _resetSessionOwnerHintsForTests()
  $gateway.set(null)
  vi.clearAllMocks()
})

// A master password typed into a background profile's unlock card must reach the
// backend that raised the prompt. The window's ambient `$gateway` may be another
// profile's socket entirely; sending the password there is a cross-backend leak.
it('routes the master password to the owning profile socket, never the ambient gateway', async () => {
  $profiles.set([{ name: 'owner' }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: 'conn-1', profile: 'owner' })
  const ambient = vi.fn().mockResolvedValue({ status: 'ok' })
  $activeSessionId.set('session-b')
  $gateway.set({ request: ambient } as never)
  setVaultUnlockRequest({ backend: 'bitwarden', displayName: 'Bitwarden', requestId: 'req-a', sessionId: 'session-a' })

  render(<PromptOverlays sessionId="session-a" />)
  const input = document.querySelector('input[type=password]')!
  fireEvent.change(input, { target: { value: 'fixture-master' } })
  fireEvent.submit(input.closest('form')!)

  await waitFor(() => expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1))
  expect(gatewayMocks.requestGatewayForAgent.mock.calls[0].slice(0, 3)).toEqual([
    'conn-1',
    'owner',
    'vault.unlock.respond'
  ])
  expect(ambient).not.toHaveBeenCalled()
  await waitFor(() => expect(sessionVaultUnlockRequest('session-a').get()).toBeNull())
})
