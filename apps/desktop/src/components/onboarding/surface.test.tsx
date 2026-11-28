import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { OverlayView } from '@/app/overlays/overlay-view'
import type * as HermesApi from '@/hermes'
import { en } from '@/i18n/en'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import { $shellChoice } from '@/store/shell-variant'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { DesktopOnboardingOverlay } from '.'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getGlobalModelOptions: async () => ({ providers: [] })
}))

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: async () => ({ providers: [] })
}))

stubResizeObserver()
stubMenuDomApis()

const idle: DesktopOnboardingState = {
  configured: true,
  flow: { status: 'idle' },
  mode: 'oauth',
  providers: [makeOAuthProvider('nous', 'Nous Portal')],
  reason: null,
  requested: false,
  firstRunSkipped: false,
  manual: false,
  localEndpoint: false,
  freeTierReady: false
}

const pendingGateway = (() => new Promise(() => {})) as OnboardingContext['requestGateway']

afterEach(async () => {
  cleanup()
  // Radix restores focus on the next task, before the jsdom realm is disposed.
  const { promise, resolve } = Promise.withResolvers<void>()
  setTimeout(resolve, 0)
  await promise
  $desktopOnboarding.set(idle)
  $shellChoice.set(null)
  localStorage.clear()
})

it('bounds manual V2 setup, preserves key drafts on outside clicks, and returns focus on Escape', async () => {
  $shellChoice.set(true)
  $desktopOnboarding.set(idle)
  const closePage = vi.fn()
  render(
    <OverlayView onClose={closePage}>
      <button onClick={() => $desktopOnboarding.set({ ...idle, manual: true, requested: true, mode: 'apikey' })}>
        Configure provider
      </button>
      <DesktopOnboardingOverlay enabled={false} profile="default" requestGateway={pendingGateway} />
    </OverlayView>
  )
  const opener = screen.getByRole('button', { name: 'Configure provider' })
  opener.focus()
  fireEvent.click(opener)
  expect(screen.getByRole('dialog', { name: en.onboarding.headerTitle })).toBeTruthy()
  const key = screen.getByPlaceholderText('Paste API key')
  fireEvent.change(key, { target: { value: 'unsaved-test-draft' } })
  await act(async () => {
    const { promise, resolve } = Promise.withResolvers<void>()
    setTimeout(resolve, 0)
    await promise
  })
  fireEvent.pointerDown(globalThis.document.querySelector('[data-slot="dialog-overlay"]')!, { button: 0 })
  expect((key as HTMLInputElement).value).toBe('unsaved-test-draft')
  expect(screen.getByRole('dialog')).toBeTruthy()
  fireEvent.keyDown(key, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await waitFor(() => expect(globalThis.document.activeElement).toBe(opener))
  expect(closePage).not.toHaveBeenCalled()
  expect($desktopOnboarding.get().configured).toBe(true)
})

it('keeps classic manual setup and its existing close action', () => {
  $shellChoice.set(false)
  $desktopOnboarding.set({ ...idle, manual: true })
  render(<DesktopOnboardingOverlay enabled={false} profile="default" requestGateway={pendingGateway} />)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(screen.getByRole('heading', { level: 2 })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect($desktopOnboarding.get().manual).toBe(false)
})

it('lets the nested model picker own Escape without closing provider setup', async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  $shellChoice.set(true)
  $desktopOnboarding.set({
    ...idle,
    manual: true,
    flow: { status: 'confirming_model', currentModel: 'test-model', label: 'Test', providerSlug: 'test', saving: false }
  })
  render(
    <QueryClientProvider client={client}>
      <DesktopOnboardingOverlay enabled={false} profile="default" requestGateway={pendingGateway} />
    </QueryClientProvider>
  )
  const change = screen.getByRole('button', { name: 'Change' })
  change.focus()
  fireEvent.click(change)
  const picker = await screen.findByRole('dialog', { name: 'Switch model' })
  fireEvent.keyDown(within(picker).getByPlaceholderText('Filter providers and models...'), { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Switch model' })).toBeNull())
  const setup = screen.getByRole('dialog', { name: en.onboarding.headerTitle })
  await waitFor(() => expect(globalThis.document.activeElement).toBe(change))
  expect($desktopOnboarding.get().flow.status).toBe('confirming_model')
  fireEvent.click(within(setup).getByRole('button', { name: 'Close' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  client.clear()
})

it('keeps V2 first-run setup as a page with the existing choose-later action', () => {
  markOnboardingDecided()
  $shellChoice.set(true)
  $desktopOnboarding.set({ ...idle, configured: false })
  render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: "I'll choose a provider later" }))
  expect($desktopOnboarding.get().firstRunSkipped).toBe(true)
})
