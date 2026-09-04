import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { $now, $scheduledRetries } from '@/store/scheduled-retry'
import { $activeSessionId } from '@/store/session'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { ScheduledRetryAction } from './scheduled-retry-action'

stubThreadEnvironment()
stubThreadViewportSize()

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function failedAssistantMessage(): ThreadMessage {
  return {
    id: 'assistant-error-1',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error: 'HTTP 429: usage limit reached' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {
        errorSurface: { layer: 'provider', code: 'rate_limit', retryable: true }
      }
    }
  } as ThreadMessage
}

// The scheduling leaf renders inside the real message runtime (it reads the
// failed message id via assistant-ui state) — mount it through the same
// external-store harness the transcript tests use, wrapped in a session view
// so the sessionId plumbing matches production.
function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [failedAssistantMessage()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <SessionViewProvider value={PRIMARY_SESSION_VIEW}>
      <AssistantRuntimeProvider runtime={runtime}>
        <ScheduledRetryAction messageId="assistant-error-1" sessionId="session-1" />
      </AssistantRuntimeProvider>
    </SessionViewProvider>
  )
}

describe('ScheduledRetryAction', () => {
  beforeEach(() => {
    window.localStorage.removeItem('hermes.desktop.scheduledRetries')
    $scheduledRetries.set({})
    $activeSessionId.set('session-1')
    $now.set(Date.now())
  })

  it('shows the scheduling menu before anything is scheduled', () => {
    render(<Harness />)

    expect(screen.getByRole('button', { name: /retry in/i })).toBeTruthy()
  })

  it('schedules from a preset and cancels again', async () => {
    render(<Harness />)

    await act(async () => {
      fireEvent.pointerDown(screen.getByRole('button', { name: /retry in/i }), { button: 0 })
    })

    await waitFor(() => screen.getByText('In 3 hours'))

    await act(async () => {
      fireEvent.click(screen.getByText('In 3 hours'))
    })

    await waitFor(() => screen.getByTestId('scheduled-retry-notice'))
    expect(screen.getByTestId('scheduled-retry-notice').textContent).toContain('Auto-retry at')

    // Persisted for the restart-survival contract.
    const stored = JSON.parse(
      window.localStorage.getItem('hermes.desktop.scheduledRetries') || 'null'
    )

    expect(stored['session-1'].messageId).toBe('assistant-error-1')

    await act(async () => {
      fireEvent.click(screen.getByTestId('cancel-scheduled-retry'))
    })

    await waitFor(() => screen.getByRole('button', { name: /retry in/i }))
    expect(window.localStorage.getItem('hermes.desktop.scheduledRetries')).toBeNull()
  })

  it('rejects a malformed custom time without scheduling', async () => {
    render(<Harness />)

    await act(async () => {
      fireEvent.pointerDown(screen.getByRole('button', { name: /retry in/i }), { button: 0 })
    })

    await waitFor(() => screen.getByTestId('retry-at-input'))

    await act(async () => {
      fireEvent.change(screen.getByTestId('retry-at-input'), { target: { value: 'nope' } })
    })
    await act(async () => {
      fireEvent.submit(screen.getByTestId('retry-at-form'))
    })

    expect(screen.getByText('Enter a time like 14:30')).toBeTruthy()
    expect(screen.queryByTestId('scheduled-retry-notice')).toBeNull()
  })

  it('schedules from a custom HH:mm entry', async () => {
    render(<Harness />)

    await act(async () => {
      fireEvent.pointerDown(screen.getByRole('button', { name: /retry in/i }), { button: 0 })
    })

    await waitFor(() => screen.getByTestId('retry-at-input'))

    await act(async () => {
      fireEvent.change(screen.getByTestId('retry-at-input'), { target: { value: '23:30' } })
    })
    await act(async () => {
      fireEvent.submit(screen.getByTestId('retry-at-form'))
    })

    await waitFor(() => screen.getByTestId('scheduled-retry-notice'))
  })
})
