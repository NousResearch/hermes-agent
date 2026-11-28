/**
 * The device-code step submits the pasted auth code on Enter. A CJK user
 * confirming a composed word must not submit the truncated code early, so the
 * field has to use the shared IME policy (`@/lib/ime`'s `isSubmitEnter`) rather
 * than an inline `!isComposing` check: that check misses the legacy
 * `keyCode === 229` commit Enter macOS Chinese IMEs fire after `compositionend`.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { OnboardingContext, OnboardingFlow } from '@/store/onboarding'

const submitOnboardingCode = vi.hoisted(() => vi.fn())

vi.mock('@/store/onboarding', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  submitOnboardingCode
}))

const { FlowPanel } = await import('./flow')

afterEach(cleanup)

const provider = {
  cli_command: 'hermes login',
  docs_url: 'https://example.invalid/docs',
  flow: 'pkce' as const,
  id: 'nous',
  name: 'Nous',
  status: { logged_in: false }
}

const ctx: OnboardingContext = { requestGateway: (async () => undefined) as OnboardingContext['requestGateway'] }

const flow: OnboardingFlow = {
  code: 'abcd-1234',
  provider,
  start: { auth_url: 'https://example.invalid/auth', expires_in: 600, flow: 'pkce', session_id: 's1' },
  status: 'awaiting_user'
}

function mount() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <FlowPanel ctx={ctx} flow={flow} leaving={false} onBegin={() => undefined} />
    </I18nProvider>
  )

  return { input: screen.getByRole('textbox') }
}

it('submits the code on a real Enter', () => {
  submitOnboardingCode.mockClear()
  const { input } = mount()

  fireEvent.keyDown(input, { key: 'Enter' })

  expect(submitOnboardingCode).toHaveBeenCalledTimes(1)
})

it('leaves an in-flight IME composition Enter to the IME', () => {
  submitOnboardingCode.mockClear()
  const { input } = mount()

  fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

  expect(submitOnboardingCode).not.toHaveBeenCalled()
})

it('leaves the legacy IME commit Enter (keyCode 229) to the IME', () => {
  submitOnboardingCode.mockClear()
  const { input } = mount()

  fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

  expect(submitOnboardingCode).not.toHaveBeenCalled()
})
