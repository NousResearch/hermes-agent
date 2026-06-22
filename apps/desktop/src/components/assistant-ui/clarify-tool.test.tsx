import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { I18nProvider } from '@/i18n'
import { $clarifyRequests, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

function renderClarifyTool(requestMock = vi.fn().mockResolvedValue({ ok: true })) {
  $clarifyRequests.set({})
  $gateway.set({ request: requestMock } as never)
  setClarifyRequest({
    requestId: 'req-1',
    question: 'Which context should Hermes use?',
    choices: ['Past sessions', 'Local reports', 'Web sources'],
    sessionId: null
  })

  const props: ToolCallMessagePartProps = {
    type: 'tool-call',
    toolCallId: 'tool-1',
    toolName: 'clarify',
    args: {
      question: 'Which context should Hermes use?',
      choices: ['Past sessions', 'Local reports', 'Web sources']
    },
    argsText: '',
    result: undefined,
    status: { type: 'running' },
    addResult: vi.fn(),
    resume: vi.fn()
  }

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ClarifyTool {...props} />
    </I18nProvider>
  )

  return requestMock
}

describe('ClarifyTool multi-select UX', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
    $gateway.set(null)
  })

  afterEach(() => {
    cleanup()
    $clarifyRequests.set({})
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  it('keeps regular choice clicks as immediate single-choice submit', async () => {
    const request = renderClarifyTool()

    fireEvent.click(screen.getByRole('button', { name: 'Past sessions' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-1',
        answer: 'Past sessions'
      })
    )
  })

  it('shows the selected single choice while the response is pending', () => {
    const request = renderClarifyTool(vi.fn(() => new Promise(() => {})))

    fireEvent.click(screen.getByRole('button', { name: 'Past sessions' }))

    expect(request).toHaveBeenCalledWith('clarify.respond', {
      request_id: 'req-1',
      answer: 'Past sessions'
    })
    const status = screen.getByRole('status')
    expect(status.textContent).toContain('Selected')
    expect(status.textContent).toContain('Past sessions')
  })

  it('uses the right-side circle to stage multiple choices without submitting until Send selected', async () => {
    const request = renderClarifyTool()

    fireEvent.click(screen.getByRole('button', { name: 'Toggle Past sessions for multi-select' }))
    fireEvent.click(screen.getByRole('button', { name: 'Toggle Web sources for multi-select' }))

    expect(request).not.toHaveBeenCalled()
    expect(screen.getByText('2 selected')).toBeTruthy()
    expect(screen.getByText('Selected')).toBeTruthy()
    expect(screen.getByText('Past sessions, Web sources')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Send selected' }))

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        request_id: 'req-1',
        answer: 'Past sessions, Web sources'
      })
    )
  })

  it('shows the current free-form draft as a custom selection', () => {
    renderClarifyTool()

    fireEvent.click(screen.getByRole('button', { name: 'Other (type your answer)' }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), {
      target: { value: 'Use only local reports' }
    })

    expect(screen.getByText('Selected')).toBeTruthy()
    expect(screen.getByText('Other (type your answer): Use only local reports')).toBeTruthy()
  })
})
