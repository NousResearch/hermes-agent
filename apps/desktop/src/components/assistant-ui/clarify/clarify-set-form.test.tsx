import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './index'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@assistant-ui/react', () => ({ useAuiState: () => true }))

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
  vi.clearAllMocks()
})

const QUESTIONS = [
  { choices: ['staging', 'production'], multiSelect: false, qid: 'q0', question: 'Which target?' },
  { choices: ['us', 'eu'], multiSelect: false, qid: 'q1', question: 'Which region?' },
  { choices: null, multiSelect: false, qid: 'q2', question: 'Anything else?' }
]

function renderSet(questions = QUESTIONS) {
  const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
  rememberServerRequest({ fail: vi.fn(), id: 'request-set', method: 'clarify', params: {}, respond: vi.fn() })
  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)
  setClarifyRequest({ questions, requestId: 'request-set', sessionId: 'session-1' })
  const args = { questions: questions.map(({ choices, question }) => ({ choices, question })) }

  const props: ToolCallMessagePartProps = {
    addResult: vi.fn(), args, argsText: JSON.stringify(args), isError: false, respondToApproval: vi.fn(),
    result: undefined, resume: vi.fn(), status: { type: 'running' }, toolCallId: 'clarify-set', toolName: 'clarify',
    type: 'tool-call'
  }

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ClarifyTool {...props} />
    </I18nProvider>
  )

  return request
}

const locks = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls.filter(call => call[0] === 'clarify.lock').map(call => [call[1].question_id, call[1].answer])

describe('a multi-question clarify set is one form', () => {
  it('shows every question at once with one submit', () => {
    renderSet()

    for (const { question } of QUESTIONS) { expect(screen.getByText(question)).toBeTruthy() }

    expect(screen.getAllByRole('button', { name: /Confirm and continue/ })).toHaveLength(1)
  })

  it('plain Enter on an answered earlier question moves on instead of sending the set half answered', async () => {
    const request = renderSet()

    // Picking moves the cursor to the next open question; the user goes back to the answered one.
    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.pointerDown(screen.getByText('Which target?'))
    // Enter on that answered row must not submit while two questions are still open: it moves on.
    fireEvent.keyDown(window, { key: 'Enter' })
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(locks(request)).toEqual([])
    // The cursor is on the next question now: Enter picks its highlighted row.
    fireEvent.keyDown(window, { key: 'Enter' })

    // Answering the rest and pressing Enter on the last question sends the whole set once.
    fireEvent.change(screen.getByPlaceholderText(/Type your answer/), { target: { value: 'after 5pm' } })
    fireEvent.keyDown(screen.getByPlaceholderText(/Type your answer/), { key: 'Enter', ctrlKey: true })
    await waitFor(() => expect(locks(request)).toHaveLength(3))
    expect(locks(request)).toEqual([['q0', 'staging'], ['q1', 'us'], ['q2', 'after 5pm']])
  })

  it('plain Enter on the last question does not send the set while earlier questions are open', async () => {
    const request = renderSet([...QUESTIONS.slice(0, 2), { ...QUESTIONS[2], choices: ['now', 'later'] }])

    // Only the last question is answered; the user goes back to it and presses Enter.
    fireEvent.click(screen.getByRole('button', { name: /later/ }))
    fireEvent.pointerDown(screen.getByText('Anything else?'))
    fireEvent.keyDown(window, { key: 'Enter' })
    // That Enter sent nothing and moved the cursor to the first open question: Enter picks its row.
    fireEvent.keyDown(window, { key: 'Enter' })
    fireEvent.keyDown(screen.getByRole('button', { name: /later/ }), { key: 'Enter', ctrlKey: true })
    await waitFor(() => expect(locks(request)).toHaveLength(3))
    expect(locks(request)).toEqual([['q0', 'staging'], ['q1', null], ['q2', 'later']])
  })

  // An empty multi-select ("pick any") is a valid answer, so it never holds plain Enter back:
  // mashing Enter from any question the keyboard cursor rests on sends the set within a few
  // presses instead of landing on the multi-select, where Enter toggles a choice on and off
  // forever. (On a multi-select row Enter is that toggle by design; a free-text question
  // takes Enter in its own text field.)
  const MIXED = [
    { choices: ['staging', 'production'], multiSelect: false, qid: 'q0', question: 'Which target?' },
    { choices: ['lint', 'tests', 'docs'], multiSelect: true, qid: 'q1', question: 'Which checks?' },
    { choices: null, multiSelect: false, qid: 'q2', question: 'Anything else?' },
    { choices: ['now', 'later'], multiSelect: false, qid: 'q3', question: 'When?' }
  ]

  const ENTER_BUDGET = 3

  for (const answerFirst of [true, false]) {
    for (const start of ['Which target?', 'When?']) {
      it(`plain Enter from "${start}" sends the set within ${ENTER_BUDGET} presses (q0 ${answerFirst ? 'answered' : 'open'})`, async () => {
        const request = renderSet(MIXED)

        if (answerFirst) {
          fireEvent.click(screen.getByRole('button', { name: /staging/ }))
        }

        fireEvent.change(screen.getByPlaceholderText(/Type your answer/), { target: { value: 'after 5pm' } })
        fireEvent.click(screen.getByRole('button', { name: /later/ }))
        // Clicking a question heading takes focus out of the text field a pick moved it into.
        screen.getByPlaceholderText(/Type your answer/).blur()
        fireEvent.pointerDown(screen.getByText(start))

        for (let press = 0; press < ENTER_BUDGET; press += 1) {
          fireEvent.keyDown(window, { key: 'Enter' })
        }

        await waitFor(() => expect(locks(request)).toHaveLength(4))
        expect(locks(request)).toEqual([['q0', 'staging'], ['q1', null], ['q2', 'after 5pm'], ['q3', 'later']])
      })
    }
  }
})
