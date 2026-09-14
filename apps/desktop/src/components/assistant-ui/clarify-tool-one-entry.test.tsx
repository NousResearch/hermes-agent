import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './clarify-tool'

// One-entry questions[] (the advertised single-question shape) must render
// choice buttons, not an infinite loading spinner.

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => true
}))

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  $activeSessionId.set(null)
  $gateway.set(null)
  vi.clearAllMocks()
})

function renderClarify(ui: ReactNode) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      {ui}
    </I18nProvider>
  )
}

const REPORTER_QUESTION = 'Approve storing the two Google credential JSONs into BWS secrets?'
const REPORTER_CHOICES = ['Store credentials in BWS now', 'Not yet']

function oneEntryQuestionsArgs() {
  return {
    questions: [{ choices: REPORTER_CHOICES, question: REPORTER_QUESTION }]
  }
}

function liveProps(args: ToolCallMessagePartProps['args']): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: 'clarify-one-entry',
    toolName: 'clarify',
    type: 'tool-call'
  }
}

describe('one-entry questions[] must not spin forever', () => {
  it('renders choice buttons from a one-entry questions[] tool.start before clarify.request', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn().mockResolvedValue({ ok: true }) } as never)

    renderClarify(<ClarifyTool {...liveProps(oneEntryQuestionsArgs())} />)

    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(screen.getByRole('button', { name: /Store credentials in BWS now/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Not yet/ })).toBeTruthy()
    expect(document.querySelector('[data-clarify-batch]')).toBeNull()
    expect(document.querySelector('[data-clarify-choices]')).toBeTruthy()
  })

  it('answers a one-entry questions[] card when the gateway request is still single-shaped', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true })

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest({
      choices: REPORTER_CHOICES,
      multiSelect: false,
      question: REPORTER_QUESTION,
      requestId: 'request-1',
      sessionId: 'session-1'
    })

    renderClarify(<ClarifyTool {...liveProps(oneEntryQuestionsArgs())} />)

    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Not yet/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'Not yet',
        request_id: 'request-1'
      })
    })
  })

  it('keeps a real one-item batch request on the batch card', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn().mockResolvedValue({ ok: true }) } as never)
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: '',
      questions: [
        { choices: REPORTER_CHOICES, multiSelect: false, qid: 'q0', question: REPORTER_QUESTION }
      ],
      requestId: 'request-batch',
      sessionId: 'session-1'
    })

    renderClarify(<ClarifyTool {...liveProps(oneEntryQuestionsArgs())} />)

    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(document.querySelector('[data-clarify-batch]')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Store credentials in BWS now/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Not yet/ })).toBeTruthy()
  })

  it('paints a multi-question card from args instead of spinning when the request is late', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn().mockResolvedValue({ ok: true }) } as never)

    renderClarify(
      <ClarifyTool
        {...liveProps({
          questions: [
            { choices: ['red', 'blue'], question: 'Color?' },
            { question: 'Name?' }
          ]
        })}
      />
    )

    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(screen.getByText('Color?')).toBeTruthy()
    expect(screen.getByText('Name?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /red/ })).toBeTruthy()
  })
})
