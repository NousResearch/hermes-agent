import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingGate } from '@/store/onboarding-gate'
import { $activeSessionId, $gatewayState, $selectedStoredSessionId, setSessionOwnerHint } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { $freeTierContinuation, continuationKey } from '@/store/free-tier-continuation'
import { closeFreeTierSignIn } from '@/store/free-tier-sign-in'
import { DesktopOnboardingOverlay } from '.'
import { listOAuthProviders } from '@/hermes'

vi.mock('@/hermes', async importOriginal => ({
  ...await importOriginal<typeof import('@/hermes')>(),
  listOAuthProviders: vi.fn(async () => ({ providers: [
    { id: 'openai-codex', name: 'ChatGPT or Codex Subscription', flow: 'device_code', cli_command: 'hermes model', docs_url: '', status: { logged_in: false } }
  ] })),
  getGlobalModelOptions: vi.fn(async () => ({ providers: [] }))
}))

vi.mock('@/lib/onboarding-enabled', () => ({ isOnboardingEnabled: () => true }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...await importOriginal<typeof import('@/store/gateway')>(),
  requestGatewayForAgent: vi.fn(async () => ({ has_guest: true, continuation_required: true, tool_calls_used: 10 }))
}))

const target = { owner: { connectionId: 'remote-a', profile: 'writer' }, sessionId: 'chat-a' }

function seed(running: boolean) {
  localStorage.setItem('hermes-intro-seen-v1', '1')
  Object.assign(window, { hermesDesktop: { skipIntro: true } })
  $gatewayState.set('open')
  $onboardingGate.set({ phase: 'done', guideQueued: false })
  setSessionOwnerHint('chat-a', target.owner)
  $activeSessionId.set('chat-a')
  $selectedStoredSessionId.set('chat-a')
  $sessionStates.set({ 'chat-a': { model: 'nous/welcome', provider: 'nous', busy: running, turnLive: running, awaitingResponse: false, messages: [] } as unknown as ClientSessionState })
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), configured: true, requested: false, manual: false, freeTierReady: false, flow: { status: 'idle' } })
  $freeTierContinuation.set({ [continuationKey(target)]: { target, status: { available: true, enabled: true, has_guest: true, label: 'Nous', model: 'nous/welcome', notice_pending: false, continuation_required: true, capped: true, tool_calls_used: 10 } } })
}

afterEach(() => { cleanup(); closeFreeTierSignIn(); $freeTierContinuation.set({}); $sessionStates.set({}); $activeSessionId.set(null); $selectedStoredSessionId.set(null) })

it('waits for the active turn, keeps the guide protected, and returns after cancel/back without clearing the gate', async () => {
  seed(true)
  const request = vi.fn(async () => ({ ok: true, free_tier: true, provider_configured: true, has_guest: true }))
  render(<DesktopOnboardingOverlay enabled={false} profile="writer" requestGateway={request as never} />)
  expect(screen.queryByRole('button', { name: 'Sign in / create account' })).toBeNull()
  act(() => { $sessionStates.set({ 'chat-a': { ...$sessionStates.get()['chat-a'], busy: false, turnLive: false } }) })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Sign in / create account' })).toBeTruthy())
  fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))
  await waitFor(() => expect(screen.getByRole('button', { name: /ChatGPT or Codex Subscription/ })).toBeTruthy())
  expect(listOAuthProviders).toHaveBeenCalledWith(target.owner)
  expect(screen.queryByRole('button', { name: /choose a provider later/i })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Back' }))
  expect($freeTierContinuation.get()[continuationKey(target)].status.continuation_required).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Use local model' }))
  expect(screen.getByPlaceholderText('http://127.0.0.1:8000/v1')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Back to sign in' }))
  fireEvent.click(screen.getByRole('button', { name: 'Sign in / create account' }))
  expect(screen.queryByRole('button', { name: 'Sign in / create account' })).toBeNull()
  act(() => closeFreeTierSignIn())
  expect(screen.getByRole('button', { name: 'Sign in / create account' })).toBeTruthy()
  act(() => $onboardingGate.set({ phase: 'guided', guideQueued: false }))
  expect(screen.queryByRole('button', { name: 'Sign in / create account' })).toBeNull()
})
