import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $secretRequest, $sudoRequest, clearAllPrompts, setSecretRequest, setSudoRequest } from '@/store/prompts'
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
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {}
  })
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
  delete window.hermesDesktop.secureCredential
  vi.clearAllMocks()
})

describe('PromptOverlays', () => {
  it('dismisses a stale sudo dialog when the gateway no longer has the password request', async () => {
    const request = vi.fn().mockRejectedValue(new Error('no pending password request'))

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('Administrator password')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($sudoRequest.get()).toBeNull())
    expect(request).toHaveBeenCalledWith('sudo.respond', { password: '', request_id: 'sudo-1' })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('opens native credential capture and sends only a stored receipt to the gateway', async () => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })
    const capture = vi.fn().mockResolvedValue({ status: 'saved' })

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    window.hermesDesktop.secureCredential = { capture }
    setSecretRequest({ envVar: 'TEST_SECRET', prompt: 'Paste a secret', requestId: 'secret-1', sessionId: 's1' })

    renderPrompts()

    await waitFor(() => expect($secretRequest.get()).toBeNull())
    expect(capture).toHaveBeenCalledWith({
      envVar: 'TEST_SECRET',
      locale: 'en',
      profile: 'default',
      prompt: 'Paste a secret',
      requestId: 'secret-1'
    })
    expect(request).toHaveBeenCalledWith('secret.respond', {
      request_id: 'secret-1',
      value: { stored: true }
    })
    expect(screen.queryByLabelText('TEST_SECRET')).toBeNull()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('maps cancellation in the native credential window to an empty response', async () => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })
    const capture = vi.fn().mockResolvedValue({ status: 'cancelled' })

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    window.hermesDesktop.secureCredential = { capture }
    setSecretRequest({ envVar: 'TEST_SECRET', prompt: 'Paste a secret', requestId: 'secret-2', sessionId: 's1' })

    renderPrompts()

    await waitFor(() => expect($secretRequest.get()).toBeNull())
    expect(request).toHaveBeenCalledWith('secret.respond', { request_id: 'secret-2', value: '' })
    expect(notifyError).not.toHaveBeenCalled()
  })
})
