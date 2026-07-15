import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $secretRequest, $sudoRequest, clearAllPrompts, setSecretRequest, setSudoRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'

import { PromptOverlays } from './prompt-overlays'

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

function renderPrompts() {
  render(
    <I18nProvider configClient={null}>
      <PromptOverlays />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
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

  it('dismisses a stale secret dialog when the gateway no longer has the value request', async () => {
    const request = vi.fn().mockRejectedValue(new Error('no pending value request'))

    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    setSecretRequest({ envVar: 'TEST_SECRET', prompt: 'Paste a secret', requestId: 'secret-1', sessionId: 's1' })

    renderPrompts()

    expect(screen.getByText('TEST_SECRET')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    await waitFor(() => expect($secretRequest.get()).toBeNull())
    expect(request).toHaveBeenCalledWith('secret.respond', { request_id: 'secret-1', value: '' })
    expect(notifyError).not.toHaveBeenCalled()
  })
})
