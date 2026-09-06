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

function renderPrompts(sessionId: string | null = 's1') {
  return render(
    <I18nProvider configClient={null}>
      <PromptOverlays sessionId={sessionId} />
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
    expect(request).toHaveBeenCalledWith('sudo.cancel', { request_id: 'sudo-1' })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it.each(['current', 'legacy', 'transient'])('cancels through the %s backend without empty submission', async mode => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })

    if (mode === 'legacy') {request.mockRejectedValueOnce(Object.assign(new Error('method not found'), { code: -32601 }))}

    if (mode === 'transient') {request.mockRejectedValueOnce(new Error('connection reset'))}
    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's1' })
    renderPrompts()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => {
      if (mode === 'transient') {expect(notifyError).toHaveBeenCalled()}
      else {expect($sudoRequest.get()).toBeNull()}
    })
    expect(request).toHaveBeenNthCalledWith(1, 'sudo.cancel', { request_id: 'sudo-1' })

    if (mode === 'legacy') {expect(request).toHaveBeenNthCalledWith(2, 'session.interrupt', { session_id: 's1' })}
    else {expect(request).toHaveBeenCalledTimes(1)}

    if (mode === 'transient') {
      expect($sudoRequest.get()?.requestId).toBe('sudo-1')
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
      await waitFor(() => expect($sudoRequest.get()).toBeNull())
      expect(request).toHaveBeenCalledTimes(2)
    }
  })

  it.each(['', 'test-password'])('marks the %j password as explicit submission', async password => {
    const request = vi.fn().mockResolvedValue({ status: 'ok' })
    $activeSessionId.set('s1')
    $gateway.set({ request } as never)
    setSudoRequest({ requestId: 'sudo-1', sessionId: 's1' })
    const { baseElement } = renderPrompts()
    const input = baseElement.querySelector('input[type="password"]')!
    fireEvent.change(input, { target: { value: password } })
    fireEvent.submit(input.closest('form')!)
    await waitFor(() => expect($sudoRequest.get()).toBeNull())
    expect(request).toHaveBeenCalledWith('sudo.respond', { intent: 'submit', password, request_id: 'sudo-1' })
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
