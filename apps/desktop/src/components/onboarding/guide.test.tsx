import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type * as HermesApi from '@/hermes'
import { enFirstRun } from '@/i18n/en_first_run'
import { $composerDraft, $freshDraftKey, mainComposerScope, setComposerDraft } from '@/store/composer'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $desktopOnboarding, closeManualOnboarding, startManualOnboarding } from '@/store/onboarding'
import { $setupGuidePhase } from '@/store/onboarding-phase'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import type * as OnboardingScopeApi from '@/store/onboarding-scope'
import { $newChatProfile, $newChatRoute } from '@/store/profile'
import {
  $guideRecovery,
  $guideScope,
  $guideSuspended,
  decideOnboarding,
  startOnboarding,
  suspendOnboarding
} from '@/store/setup-guide'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { DesktopOnboardingOverlay } from '.'

const transport = vi.hoisted(() => ({ request: vi.fn(), ready: false }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  listOAuthProviders: vi.fn(async () => ({ providers: [makeOAuthProvider('nous', 'Nous Portal')] })),
  getGlobalModelOptions: vi.fn(async () => ({ providers: [] }))
}))
vi.mock('@/store/onboarding-scope', async importOriginal => ({
  ...(await importOriginal<typeof OnboardingScopeApi>()),
  requestOnboardingGateway: (_scope: unknown, method: string, params?: Record<string, unknown>) =>
    transport.request(method, params)
}))

beforeEach(() => {
  transport.ready = false
  transport.request.mockReset()
  transport.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
    switch (method) {
      case 'onboarding.state':
        return { run: true, eligible: false, started: true, step: 'model' }
      case 'onboarding.set_run':
        return { run: params?.run === true, started: true, step: params?.step ?? 'model' }
      case 'shared_metrics.status':
        return { decided: true, enabled: false }
      case 'setup.status':
        return { provider_configured: false }
      case 'setup.runtime_check':
        return { ok: transport.ready, model: 'local-model' }
      case 'free_tier.status':
        return { enabled: false }
      default:
        throw new Error(`Unexpected setup side effect: ${method}`)
    }
  })
  closeManualOnboarding()
  $desktopOnboarding.set({
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: [makeOAuthProvider('nous', 'Nous Portal')],
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  })
  $guideRecovery.set(null)
  $guideSuspended.set(false)
  $setupGuidePhase.set('idle')
  $agentSidebarChoice.set(true)
  $localModelsEnabled.set(true)
  setApiRequestConnection(null)
  setApiRequestProfile('default')
  markOnboardingDecided()
})

afterEach(() => {
  cleanup()
  closeManualOnboarding()
  $setupGuidePhase.set('idle')
  $guideRecovery.set(null)
  $guideSuspended.set(false)
  $agentSidebarChoice.set(null)
  setComposerDraft('')
  mainComposerScope.$attachments.set([])
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $localModelsEnabled.set(false)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  window.location.hash = ''
  window.localStorage.clear()
})

it('releases the real provider picker for local settings and resumes only on request', async () => {
  await decideOnboarding(transport.request as never, true)
  const settingsAction = vi.fn(() => {
    transport.ready = true
  })
  render(
    <>
      <button onClick={settingsAction}>Configure local model</button>
      <DesktopOnboardingOverlay enabled profile="default" requestGateway={transport.request as never} />
    </>
  )

  fireEvent.click(await screen.findByRole('button', { name: enFirstRun.guide.providers }))
  fireEvent.click(await screen.findByRole('button', { name: /Run models locally/ }))

  expect(window.location.hash).toBe('#/settings?tab=providers&pview=local')
  expect($desktopOnboarding.get().manual).toBe(false)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(document.body.style.pointerEvents).not.toBe('none')
  fireEvent.click(screen.getByRole('button', { name: 'Configure local model' }))
  expect(settingsAction).toHaveBeenCalledOnce()
  expect(transport.request.mock.calls.some(([method]) => method === 'onboarding.set_run')).toBe(false)

  fireEvent.click(screen.getByRole('button', { name: enFirstRun.guide.resume }))
  expect(await screen.findByRole('dialog')).toBeTruthy()
  expect(screen.getByText(enFirstRun.guide.model)).toBeTruthy()
  await waitFor(() =>
    expect(screen.getByRole('button', { name: enFirstRun.guide.finish }).hasAttribute('disabled')).toBe(false)
  )
  expect($guideScope.get()).toMatchObject({ connectionId: null, profile: 'default' })
})

it('shows old-runtime upgrade instructions and retries setup without creating a conversation', async () => {
  transport.request.mockRejectedValue(new Error('Invalid params: sessionless [type=extra_forbidden]'))
  await decideOnboarding(transport.request as never, true)
  render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={transport.request as never} />)

  expect(screen.getByRole('alert').textContent).toBe(enFirstRun.guide.upgradeRequired)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(transport.request.mock.calls.some(([method]) => /session\.create|prompt\./.test(method))).toBe(false)

  transport.request.mockResolvedValue({ run: true, eligible: false, started: true, step: 'preferences', decided: true })
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  expect(await screen.findByRole('dialog')).toBeTruthy()
  expect(screen.getByText(enFirstRun.guide.appearance)).toBeTruthy()
  expect(screen.queryByText(enFirstRun.guide.upgradeRequired)).toBeNull()
  expect(
    transport.request.mock.calls
      .filter(([method]) => method === 'onboarding.state')
      .every(([, params]) => params.sessionless === true)
  ).toBe(true)
})

it('keeps the classic manual local-model handoff free of the redesigned guide', async () => {
  $agentSidebarChoice.set(false)
  startManualOnboarding()
  render(<DesktopOnboardingOverlay enabled={false} profile="default" requestGateway={transport.request as never} />)
  fireEvent.click(await screen.findByRole('button', { name: /Run models locally/ }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  expect(window.location.hash).toBe('#/settings?tab=providers&pview=local')
  expect(screen.queryByRole('button', { name: enFirstRun.guide.resume })).toBeNull()
})

it('does not show late recovery after switching back to classic', async () => {
  const reply = Promise.withResolvers<never>()
  transport.request.mockReturnValue(reply.promise)
  const boot = decideOnboarding(transport.request as never, true)
  render(<DesktopOnboardingOverlay enabled={false} profile="default" requestGateway={transport.request as never} />)
  await act(async () => {
    $agentSidebarChoice.set(false)
    reply.reject(new Error('sessionless: extra_forbidden'))
    await boot
  })
  expect(screen.queryByText(enFirstRun.guide.upgradeRequired)).toBeNull()
})

it.each([false, true])(
  'finishes a settings rerun without replacing the foreground draft (suspended: %j)',
  async suspended => {
    transport.ready = true
    await startOnboarding(transport.request as never)
    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={transport.request as never} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Continue' }))

    if (suspended) {
      act(() => suspendOnboarding())
      setApiRequestProfile('research')
      fireEvent.click(await screen.findByRole('button', { name: enFirstRun.guide.resume }))
    }

    const profile = suspended ? 'research' : 'default'
    const owner = { connectionId: 'local', profile }
    const attachment = { id: 'draft-notes', kind: 'file' as const, label: 'notes.txt' }
    act(() => {
      $newChatProfile.set(profile)
      $newChatRoute.set(owner)
      setComposerDraft('Keep this unsent request')
      mainComposerScope.$attachments.set([attachment])
    })
    window.location.hash = '#/settings?tab=onboarding'
    const draftKey = $freshDraftKey.get()
    await waitFor(() =>
      expect(screen.getByRole('button', { name: enFirstRun.guide.finish }).hasAttribute('disabled')).toBe(false)
    )
    fireEvent.click(screen.getByRole('button', { name: enFirstRun.guide.finish }))
    await waitFor(() => expect($setupGuidePhase.get()).toBe('done'))

    expect($freshDraftKey.get()).toBe(draftKey)
    expect($composerDraft.get()).toBe('Keep this unsent request')
    expect(mainComposerScope.$attachments.get()).toEqual([attachment])
    expect($newChatProfile.get()).toBe(profile)
    expect($newChatRoute.get()).toEqual(owner)
    expect(window.location.hash).toBe('#/settings?tab=onboarding')
    expect(transport.request.mock.calls.some(([method]) => /session\.create|prompt\./.test(method))).toBe(false)
  }
)
