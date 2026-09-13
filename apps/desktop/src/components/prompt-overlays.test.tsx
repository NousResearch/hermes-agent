import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $secretRequest, $sudoRequest, clearAllPrompts, setSecretRequest, setSudoRequest } from '@/store/prompts'
import { hasOpenServerRequest, rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { PromptOverlays } from './prompt-overlays'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

function renderPrompts(sessionId: string | null = 's1') {
  render(
    <I18nProvider configClient={null}>
      <PromptOverlays sessionId={sessionId} />
    </I18nProvider>
  )
}

beforeEach(() => {
  resetServerRequestsForTests()
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
  vi.clearAllMocks()
})

describe('PromptOverlays', () => {
  it('answers the live sudo request with an empty value on Cancel and clears the dialog', async () => {
    const respond = vi.fn()
    const request = vi.fn()

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    rememberServerRequest({ fail: vi.fn(), id: 'sudo-1', method: 'sudo', params: {}, respond })
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('Administrator password')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($sudoRequest.get()).toBeNull())
    expect(respond).toHaveBeenCalledWith({ value: '' })
    expect(hasOpenServerRequest('sudo-1')).toBe(false)
    expect(request).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('dismisses a stale sudo dialog when the gateway no longer has the request open', async () => {
    const request = vi.fn()

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    // No server request registered under this id: it expired / was cancelled.
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('Administrator password')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($sudoRequest.get()).toBeNull())
    expect(request).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('answers the live secret request with an empty value on Cancel and clears the dialog', async () => {
    const respond = vi.fn()
    const request = vi.fn()

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    rememberServerRequest({ fail: vi.fn(), id: 'secret-1', method: 'secret', params: {}, respond })
    setSecretRequest({ envVar: 'TEST_SECRET', prompt: 'Paste a secret', requestId: 'secret-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('TEST_SECRET')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($secretRequest.get()).toBeNull())
    expect(respond).toHaveBeenCalledWith({ value: '' })
    expect(hasOpenServerRequest('secret-1')).toBe(false)
    expect(request).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('dismisses a stale secret dialog when the gateway no longer has the request open', async () => {
    const request = vi.fn()

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    setSecretRequest({ envVar: 'TEST_SECRET', prompt: 'Paste a secret', requestId: 'secret-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('TEST_SECRET')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($secretRequest.get()).toBeNull())
    expect(request).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('shows the command being authorized in the sudo dialog', () => {
    $activeSessionId.set('s1')
    $gateway.set({ request: vi.fn() } as never)
    setSudoRequest({ command: 'sudo pacman -S task', requestId: 'sudo-2', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('sudo pacman -S task')).toBeTruthy()
  })

  it('falls back to the generic description when no command is present', () => {
    $activeSessionId.set('s1')
    $gateway.set({ request: vi.fn() } as never)
    setSudoRequest({ requestId: 'sudo-3', sessionId: 's1' })

    renderPrompts()

    expect(screen.queryByText('sudo pacman -S task')).toBeNull()
    expect(screen.getByText(/Hermes needs your sudo password/)).toBeTruthy()
  })
})
