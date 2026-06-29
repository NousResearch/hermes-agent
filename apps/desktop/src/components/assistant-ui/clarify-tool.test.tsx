import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $notifications, clearNotifications } from '@/store/notifications'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './clarify-tool'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => true
}))

const REQUEST_ID = 'clarify-req-1'
const SESSION_ID = 'session-1'

function renderPendingClarify(request: { request?: ReturnType<typeof vi.fn> } = {}) {
  const requestMock = request.request ?? vi.fn(async () => ({ ok: true }))

  $activeSessionId.set(SESSION_ID)
  setClarifyRequest({
    requestId: REQUEST_ID,
    sessionId: SESSION_ID,
    question: 'Pick one?',
    choices: ['Continue waiting', 'Stop now']
  })
  $gateway.set({ request: requestMock } as never)

  render(
    <I18nProvider configClient={null}>
      <ClarifyTool
        {...({
          args: { question: 'Pick one?', choices: ['Continue waiting', 'Stop now'] },
          result: undefined
        } as unknown as Parameters<typeof ClarifyTool>[0])}
      />
    </I18nProvider>
  )

  return requestMock
}

beforeEach(() => {
  clearNotifications()
  clearClarifyRequest()
  $activeSessionId.set(null)
  $gateway.set(null)
})

afterEach(() => {
  cleanup()
  clearNotifications()
  clearClarifyRequest()
  $activeSessionId.set(null)
  $gateway.set(null)
})

describe('ClarifyTool response timeout handling', () => {
  it('uses a long dedicated timeout for clarify.respond acknowledgements', async () => {
    const requestMock = renderPendingClarify()

    fireEvent.click(screen.getByRole('button', { name: /Continue waiting/i }))
    fireEvent.click(screen.getByRole('button', { name: /^Continue$/i }))

    await waitFor(() => {
      expect(requestMock).toHaveBeenCalledWith(
        'clarify.respond',
        { request_id: REQUEST_ID, answer: 'Continue waiting' },
        120_000
      )
    })
  })

  it('warns that a timed-out clarify response may still be processing', async () => {
    const requestMock = renderPendingClarify({
      request: vi.fn(async () => {
        throw new Error('request timed out: clarify.respond')
      })
    })

    fireEvent.click(screen.getByRole('button', { name: /Stop now/i }))
    fireEvent.click(screen.getByRole('button', { name: /^Continue$/i }))

    await waitFor(() => expect(requestMock).toHaveBeenCalled())

    await waitFor(() => {
      expect($notifications.get()[0]).toMatchObject({
        kind: 'warning',
        title: 'Clarify response may still be processing',
        message: expect.stringContaining('wait a moment before trying again')
      })
    })
  })
})
