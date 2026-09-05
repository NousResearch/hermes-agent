import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onComposerInsertRequest } from '@/app/chat/composer/focus'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { hiddenPaneProps } from '@/components/pane-shell/pane-visibility'
import { $activeTreeGroup, $hoveredTreeGroup } from '@/components/pane-shell/tree/store'
import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { ClarifyTool, readClarifyBatchResult, readClarifyResult } from './clarify-tool'

// The OWNER-socket seam (`requestForOwnedSession` → `requestForSessionProfile`
// → here). Mocked so the real owner ladder still runs against real fixtures and
// only the dial is observed; the rest of the gateway store stays actual, so
// `$gateway` remains the genuine ambient atom every other test in this file
// drives.
const gatewayMocks = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

// The live pending card used to require message-running. Tests that exercise
// the pending form force that on; the settle-shift case flips it off.
let messageRunning = true

vi.mock('@assistant-ui/react', () => ({
  useAuiState: () => messageRunning
}))

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  $activeSessionId.set(null)
  $gateway.set(null)
  messageRunning = true
  vi.clearAllMocks()
})

function clarifyTree(ui: ReactNode) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      {ui}
    </I18nProvider>
  )
}

function renderClarify(ui: ReactNode) {
  return render(clarifyTree(ui))
}

function settledClarifyProps(
  args: ToolCallMessagePartProps['args'],
  result: ToolCallMessagePartProps['result'],
  toolCallId: string
): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result,
    resume: vi.fn(),
    status: { type: 'complete' },
    toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
  }
}

function liveClarifyProps(choices = ['staging', 'production']): ToolCallMessagePartProps {
  const args = { choices, question: 'Which deployment target?' }

  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: 'clarify-live',
    toolName: 'clarify',
    type: 'tool-call'
  }
}

function renderLiveClarify({ multiSelect = false }: { multiSelect?: boolean } = {}) {
  const request = vi.fn().mockResolvedValue({ ok: true })

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)
  setClarifyRequest({
    choices: ['staging', 'production'],
    multiSelect,
    question: 'Which deployment target?',
    requestId: 'request-1',
    sessionId: 'session-1'
  })
  const { rerender } = renderClarify(<ClarifyTool {...liveClarifyProps()} />)

  return { request, rerender }
}

describe('ClarifyTool live card stays mounted across settle', () => {
  it('keeps the question card while the gateway request is open and the turn reports not-running', () => {
    messageRunning = false
    renderLiveClarify()

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(document.querySelector('[data-clarify-choices]')).toBeTruthy()
    expect(document.querySelector('[data-clarify-settled]')).toBeNull()
  })

  it('demotes to a tool row when the turn stopped and no request is left to answer', () => {
    messageRunning = false
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn() } as never)
    renderClarify(<ClarifyTool {...liveClarifyProps()} />)

    expect(document.querySelector('[data-clarify-choices]')).toBeNull()
    expect(screen.queryByRole('button', { name: /Continue/ })).toBeNull()
  })

  it('holds the card through the gap between answering and the settled result', async () => {
    const { request, rerender } = renderLiveClarify()

    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalled()
    })

    // tool.complete is what swaps in the settled card; the turn can already
    // read as not-running in that gap.
    messageRunning = false
    rerender(clarifyTree(<ClarifyTool {...liveClarifyProps()} />))

    expect(document.querySelector('[data-clarify-choices]')).toBeTruthy()
  })

  it('demotes when the turn is stopped after the card was live but never answered', () => {
    renderLiveClarify()

    expect(document.querySelector('[data-clarify-choices]')).toBeTruthy()

    messageRunning = false
    act(() => clearClarifyRequest('request-1', 'session-1'))

    expect(document.querySelector('[data-clarify-choices]')).toBeNull()
    expect(screen.queryByRole('button', { name: /Continue/ })).toBeNull()
  })

  it('paints the question from tool args instead of a spinner while request_id is still racing', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn() } as never)
    renderClarify(<ClarifyTool {...liveClarifyProps()} />)

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(screen.getByRole('button', { name: /Continue/ }).hasAttribute('disabled')).toBe(true)
  })
})

describe('ClarifyTool choice selection', () => {
  it('selects independently, deselects and submits multi-select choices as a JSON array', async () => {
    const { request } = renderLiveClarify({ multiSelect: true })
    const staging = screen.getByRole('button', { name: /staging/ })
    const production = screen.getByRole('button', { name: /production/ })

    fireEvent.click(staging)
    fireEvent.click(production)
    expect(staging.getAttribute('aria-pressed')).toBe('true')
    expect(production.getAttribute('aria-pressed')).toBe('true')

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    expect(staging.getAttribute('aria-pressed')).toBe('true')
    expect(production.getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(staging)
    expect(staging.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(staging)

    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: JSON.stringify(['production', 'staging']),
        request_id: 'request-1'
      })
    })
  })

  it('keeps single-select replacement and plain-string submission', async () => {
    const { request } = renderLiveClarify()
    const staging = screen.getByRole('button', { name: /staging/ })
    const production = screen.getByRole('button', { name: /production/ })

    fireEvent.click(staging)
    fireEvent.click(production)

    expect(staging.getAttribute('aria-pressed')).toBe('false')
    expect(production.getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'production',
        request_id: 'request-1'
      })
    })
  })
})

describe('readClarifyResult', () => {
  it('reads question + user_response from the tool JSON payload', () => {
    expect(
      readClarifyResult({
        question: 'Which target?',
        choices_offered: ['staging', 'prod'],
        user_response: 'staging'
      })
    ).toEqual({
      question: 'Which target?',
      answer: 'staging',
      error: undefined
    })
  })

  it('parses a JSON string result the same way as an object', () => {
    expect(
      readClarifyResult(
        JSON.stringify({
          question: 'Ship it?',
          user_response: 'yes'
        })
      )
    ).toEqual({
      question: 'Ship it?',
      answer: 'yes',
      error: undefined
    })
  })

  it('keeps an empty user_response so Skip can render as skipped', () => {
    expect(readClarifyResult({ question: 'Ok?', user_response: '' })).toEqual({
      question: 'Ok?',
      answer: '',
      error: undefined
    })
  })
})

describe('ClarifyTool settled view', () => {
  it('keeps the question and answer visible after the tool completes', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which deployment target?', choices: ['staging', 'prod'] },
          {
            question: 'Which deployment target?',
            choices_offered: ['staging', 'prod'],
            user_response: 'staging'
          },
          'clarify-1'
        )}
      />
    )

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(screen.getByText('staging')).toBeTruthy()
    expect(document.querySelector('[data-clarify-settled]')).toBeTruthy()
    expect(document.querySelector('[data-clarify-answer]')?.textContent).toBe('staging')
  })

  it('labels an empty response as Skipped', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Anything else?' },
          { question: 'Anything else?', user_response: '' },
          'clarify-2'
        )}
      />
    )

    expect(screen.getByText('Anything else?')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
  })

  it('keeps the original choices visible and clickable after a skip', async () => {
    const inserts: string[] = []

    const stop = onComposerInsertRequest(detail => {
      inserts.push(detail.text)
    })

    try {
      renderClarify(
        <ClarifyTool
          {...settledClarifyProps(
            { question: 'Which deployment target?', choices: ['staging', 'prod'] },
            { question: 'Which deployment target?', user_response: '' },
            'clarify-3'
          )}
        />
      )

      // The skip label renders AND the original options are still on screen.
      expect(screen.getByText('Skipped')).toBeTruthy()
      const group = document.querySelector('[data-clarify-late-choices]')
      expect(group).toBeTruthy()
      expect(screen.getByText('staging')).toBeTruthy()
      expect(screen.getByText('prod')).toBeTruthy()

      // Picking one drafts a quoted follow-up into the composer. The insert
      // bus defers dispatch by a macrotask, so flush one tick.
      fireEvent.click(screen.getByText('prod'))
      await new Promise(resolve => window.setTimeout(resolve, 0))

      expect(inserts).toHaveLength(1)
      expect(inserts[0]).toContain('Which deployment target?')
      expect(inserts[0]).toContain('prod')
    } finally {
      stop()
    }
  })

  it('does not render late choices on an answered clarify', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Which deployment target?', choices: ['staging', 'prod'] },
          { question: 'Which deployment target?', user_response: 'staging' },
          'clarify-4'
        )}
      />
    )

    expect(document.querySelector('[data-clarify-late-choices]')).toBeNull()
  })

  it('does not render late choices for a free-text (no-choice) skip', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          { question: 'Anything else?' },
          { question: 'Anything else?', user_response: '' },
          'clarify-5'
        )}
      />
    )

    expect(document.querySelector('[data-clarify-late-choices]')).toBeNull()
  })
})

describe('ClarifyTool keyboard navigation', () => {
  it('cycles through choices and Other with the arrow keys', () => {
    renderLiveClarify()

    const staging = screen.getByRole('button', { name: /staging/ })
    const production = screen.getByRole('button', { name: /production/ })
    const other = screen.getByPlaceholderText(/Other/)

    expect(staging.getAttribute('data-highlighted')).toBe('true')
    expect(staging.getAttribute('aria-current')).toBe('true')
    expect(staging.getAttribute('aria-keyshortcuts')).toBe('A 1')

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    expect(production.getAttribute('data-highlighted')).toBe('true')

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    expect(other.closest('label')?.getAttribute('data-highlighted')).toBe('true')
    expect(other.getAttribute('aria-current')).toBe('true')
    expect(other.getAttribute('aria-keyshortcuts')).toBe('C 3')

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    expect(staging.getAttribute('data-highlighted')).toBe('true')

    fireEvent.keyDown(window, { key: 'ArrowUp' })
    expect(other.closest('label')?.getAttribute('data-highlighted')).toBe('true')
  })

  it('selects by number and confirms the answer with Enter', async () => {
    const { request } = renderLiveClarify()

    fireEvent.keyDown(window, { key: '2' })
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'production',
        request_id: 'request-1'
      })
    })
  })

  it('stages a highlighted multi-select choice with Enter and submits it with Continue', async () => {
    const { request } = renderLiveClarify({ multiSelect: true })
    const production = screen.getByRole('button', { name: /production/ })

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'Enter' })

    expect(production.getAttribute('aria-pressed')).toBe('true')
    expect(request).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: JSON.stringify(['production']),
        request_id: 'request-1'
      })
    })
  })

  it('focuses Other when its number is pressed and leaves typing keys alone', () => {
    renderLiveClarify()

    const other = screen.getByPlaceholderText(/Other/)

    fireEvent.keyDown(window, { key: '3' })
    expect(document.activeElement).toBe(other)

    fireEvent.change(other, { target: { value: 'canary' } })
    fireEvent.keyDown(window, { key: 'ArrowUp' })
    expect(document.activeElement).toBe(other)
    expect((other as HTMLTextAreaElement).value).toBe('canary')
  })

  it('does not intercept keyboard events while an action button has focus', () => {
    const { request } = renderLiveClarify()
    const skip = screen.getByRole('button', { name: 'Skip' })

    skip.focus()

    expect(fireEvent.keyDown(window, { key: 'Enter' })).toBe(true)
    expect(fireEvent.keyDown(window, { key: 'ArrowDown' })).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })
})

describe('ClarifyTool recommended option', () => {
  it('dims the (Recommended) label and answers with the choice the backend sent', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true })

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest({
      choices: ['staging (Recommended)', 'production'],
      multiSelect: false,
      question: 'Which deployment target?',
      requestId: 'request-1',
      sessionId: 'session-1'
    })
    renderClarify(<ClarifyTool {...liveClarifyProps(['staging (Recommended)', 'production'])} />)

    const recommended = screen.getByRole('button', { name: /staging/ })

    // The label rides in its own muted span so the option text still reads first.
    expect(recommended.querySelector('.text-\\(--ui-text-tertiary\\)')?.textContent).toBe('(Recommended)')

    fireEvent.click(recommended)
    fireEvent.keyDown(window, { key: 'Enter' })

    // The decorated string goes back verbatim; the tool strips the label before
    // the agent ever sees the answer.
    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'staging (Recommended)',
        request_id: 'request-1'
      })
    })
  })
})

describe('ClarifyTool pending marker', () => {
  it('marks a live choices card with its row count so type-to-focus yields exactly its keys', () => {
    renderLiveClarify()

    // `clarifyCardOwnsKey` reads the count off this marker to yield only the
    // shortcuts the card renders (A..N + "Other", 1-9, Enter) and let every
    // other printable through to the composer.
    const card = document.querySelector('[data-clarify-choices]')

    expect(card).toBeTruthy()
    expect(Number(card?.getAttribute('data-clarify-choices'))).toBeGreaterThan(0)
  })

  it('does not mark a free-text (no-choice) pending card', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn().mockResolvedValue({ ok: true }) } as never)
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: 'Anything else?',
      requestId: 'request-1',
      sessionId: 'session-1'
    })

    const args = { question: 'Anything else?' }
    renderClarify(
      <ClarifyTool
        addResult={vi.fn()}
        args={args}
        argsText={JSON.stringify(args)}
        isError={false}
        respondToApproval={vi.fn()}
        result={undefined}
        resume={vi.fn()}
        status={{ type: 'running' }}
        toolCallId="clarify-free"
        toolName="clarify"
        type="tool-call"
      />
    )

    // No shortcuts → nothing to protect → composer type-to-focus stays live.
    expect(document.querySelector('[data-clarify-choices]')).toBeNull()
  })
})

// ─── Batch (multi-question) clarify ─────────────────────────────────────────

function batchArgs(): { questions: { question: string; choices?: string[] }[] } {
  return {
    questions: [{ choices: ['red', 'blue'], question: 'Color?' }, { question: 'Name?' }]
  }
}

function liveBatchProps(): ToolCallMessagePartProps {
  const args = batchArgs()

  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: 'clarify-batch',
    toolName: 'clarify',
    type: 'tool-call'
  }
}

function renderLiveBatch(lockedAnswers?: Record<string, string>, multiSelect = false) {
  const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)
  setClarifyRequest({
    choices: null,
    lockedAnswers,
    multiSelect: false,
    question: '',
    questions: [
      { choices: ['red', 'blue'], multiSelect, qid: 'q0', question: 'Color?' },
      { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
    ],
    requestId: 'request-batch',
    sessionId: 'session-1'
  })
  renderClarify(<ClarifyTool {...liveBatchProps()} />)

  return request
}

describe('readClarifyBatchResult', () => {
  it('parses responses with string and list answers plus timed_out', () => {
    const parsed = readClarifyBatchResult(
      JSON.stringify({
        responses: [
          { question: 'Color?', user_response: 'red' },
          { question: 'Tools?', user_response: ['a', 'b'] },
          { question: 'Name?', user_response: '' }
        ],
        timed_out: true
      })
    )

    expect(parsed.timedOut).toBe(true)
    expect(parsed.responses).toHaveLength(3)
    expect(parsed.responses[1]?.answer).toEqual(['a', 'b'])
    expect(parsed.responses[2]?.answer).toBe('')
  })

  it('returns empty responses for single-question payloads', () => {
    expect(readClarifyBatchResult({ question: 'Q?', user_response: 'a' }).responses).toEqual([])
  })
})

describe('ClarifyTool batch card', () => {
  it('renders and answers a one-question array with the single-question card', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
    const args = { questions: [{ choices: ['red', 'blue'], question: 'Color?' }] }

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: '',
      questions: [{ choices: ['red', 'blue'], multiSelect: false, qid: 'q0', question: 'Color?' }],
      requestId: 'request-one-question',
      sessionId: 'session-1'
    })
    renderClarify(
      <ClarifyTool
        {...liveBatchProps()}
        args={args}
        argsText={JSON.stringify(args)}
        toolCallId="clarify-one-question"
      />
    )

    expect(screen.getByText('Color?')).toBeTruthy()
    expect(document.querySelector('[data-clarify-batch]')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'red',
        question_id: 'q0',
        request_id: 'request-one-question'
      })
    })
  })

  it('answers a one-question array when the gateway uses the single-question wire shape', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
    const args = { questions: [{ choices: ['red', 'blue'], question: 'Color?' }] }

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest({
      choices: ['red', 'blue'],
      multiSelect: false,
      question: 'Color?',
      requestId: 'request-single-wire',
      sessionId: 'session-1'
    })
    renderClarify(
      <ClarifyTool {...liveBatchProps()} args={args} argsText={JSON.stringify(args)} toolCallId="clarify-single-wire" />
    )

    expect(screen.getByText('Color?')).toBeTruthy()
    expect(document.querySelector('[data-clarify-batch]')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /blue/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: 'blue',
        request_id: 'request-single-wire'
      })
    })
  })

  it('renders every question at once', () => {
    renderLiveBatch()

    expect(screen.getByText('Color?')).toBeTruthy()
    expect(screen.getByText('Name?')).toBeTruthy()
    expect(screen.getByText('0 of 2 answered')).toBeTruthy()
  })

  it('stages locally and keeps the single confirm disabled until all answered', async () => {
    const request = renderLiveBatch()
    const confirm = screen.getByRole('button', { name: /Confirm and continue/ })

    expect((confirm as HTMLButtonElement).disabled).toBe(true)

    // Staging a pick sends NOTHING to the server.
    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    expect(screen.getByText('1 of 2 answered')).toBeTruthy()
    expect(request).not.toHaveBeenCalled()
    expect((confirm as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    expect(screen.getByText('2 of 2 answered')).toBeTruthy()
    expect(request).not.toHaveBeenCalled()
    expect((confirm as HTMLButtonElement).disabled).toBe(false)
  })

  it('confirm sends every per-question lock in order and completes the batch', async () => {
    const request = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.respond', {
      answer: 'red',
      question_id: 'q0',
      request_id: 'request-batch'
    })
    expect(request).toHaveBeenNthCalledWith(2, 'clarify.respond', {
      answer: 'packet',
      question_id: 'q1',
      request_id: 'request-batch'
    })
  })

  it('a staged answer stays editable before confirm', async () => {
    const request = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.click(screen.getByRole('button', { name: /blue/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    // The re-pick won: blue, not red.
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.respond', {
      answer: 'blue',
      question_id: 'q0',
      request_id: 'request-batch'
    })
  })

  it('pre-stages replayed locked answers from a reconnect', () => {
    renderLiveBatch({ q0: 'red' })

    // The replayed answer counts as staged: one question left to answer.
    expect(screen.getByText('1 of 2 answered')).toBeTruthy()
  })

  it('reselects every choice from a replayed multi-select JSON answer', () => {
    renderLiveBatch({ q0: '["red","blue"]' }, true)

    expect(screen.getByRole('button', { name: /red/ }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: /blue/ }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('1 of 2 answered')).toBeTruthy()
  })

  it('Skip cancels the whole batch without a question_id', async () => {
    const request = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.respond', {
        answer: '',
        request_id: 'request-batch'
      })
    })
  })

  it('renders the settled batch with all questions and answers', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          batchArgs(),
          JSON.stringify({
            responses: [
              { choices_offered: ['red', 'blue'], question: 'Color?', user_response: 'red' },
              { choices_offered: null, question: 'Name?', user_response: '' }
            ]
          }),
          'clarify-batch-settled'
        )}
      />
    )

    expect(screen.getByText('Color?')).toBeTruthy()
    expect(screen.getByText('red')).toBeTruthy()
    expect(screen.getByText('Name?')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
  })
})

// ─── Owner routing (#91684 client half) ─────────────────────────────────────
// The clarify card used to answer on the AMBIENT socket. That socket follows
// foreground focus, so after a profile / Bot Chat switch it can be profile B
// while the blocking clarify belongs to profile A — the response lands on a
// backend that never held the request and the owner stays blocked until the
// tool times out. Every live clarify.respond now routes by request.sessionId.

const OWNER_CONNECTION_ID = 'conn-profile-a'
const OWNER_PROFILE = 'profile-a'

/** Profile A owns the clarify's session; the window has since switched to
 *  profile B, so `$gateway` (ambient) is profile B's socket. */
function armCrossProfileOwner() {
  // Two profiles exist → the ambient gateway is not provably the sole backend,
  // so the legacy single-backend escape hatch stays shut.
  $profiles.set([{ name: OWNER_PROFILE }, { name: 'profile-b' }] as never)
  setSessionOwnerHint('session-a', { connectionId: OWNER_CONNECTION_ID, profile: OWNER_PROFILE })

  const ambient = vi.fn().mockResolvedValue({ ok: true })

  $activeSessionId.set('session-a')
  $gateway.set({ request: ambient } as never)

  return ambient
}

function expectOwnerCall(nth: number, params: Record<string, unknown>) {
  expect(gatewayMocks.requestGatewayForAgent).toHaveBeenNthCalledWith(
    nth,
    OWNER_CONNECTION_ID,
    OWNER_PROFILE,
    'clarify.respond',
    params
  )
}

describe('ClarifyTool owner routing', () => {
  afterEach(() => {
    $profiles.set([])
    _resetSessionOwnerHintsForTests({ storage: true })
    gatewayMocks.requestGatewayForAgent.mockClear()
  })

  it('answers a single clarify on the owner socket, never profile B ambient', async () => {
    const ambient = armCrossProfileOwner()

    setClarifyRequest({
      choices: ['staging', 'production'],
      multiSelect: false,
      question: 'Which deployment target?',
      requestId: 'request-1',
      sessionId: 'session-a'
    })
    renderClarify(<ClarifyTool {...liveClarifyProps()} />)

    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }))

    await waitFor(() => {
      expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1)
    })
    expectOwnerCall(1, { answer: 'staging', request_id: 'request-1' })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('sends both sequential batch locks on the owner socket, in order', async () => {
    const ambient = armCrossProfileOwner()

    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: '',
      questions: [
        { choices: ['red', 'blue'], multiSelect: false, qid: 'q0', question: 'Color?' },
        { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
      ],
      requestId: 'request-batch',
      sessionId: 'session-a'
    })
    renderClarify(<ClarifyTool {...liveBatchProps()} />)

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(2)
    })
    // The LAST lock resolves the blocked tool, so order is load-bearing.
    expectOwnerCall(1, { answer: 'red', question_id: 'q0', request_id: 'request-batch' })
    expectOwnerCall(2, { answer: 'packet', question_id: 'q1', request_id: 'request-batch' })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('sends a batch skip/cancel on the owner socket', async () => {
    const ambient = armCrossProfileOwner()

    setClarifyRequest({
      choices: null,
      multiSelect: false,
      question: '',
      questions: [
        { choices: ['red', 'blue'], multiSelect: false, qid: 'q0', question: 'Color?' },
        { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
      ],
      requestId: 'request-batch',
      sessionId: 'session-a'
    })
    renderClarify(<ClarifyTool {...liveBatchProps()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))

    await waitFor(() => {
      expect(gatewayMocks.requestGatewayForAgent).toHaveBeenCalledTimes(1)
    })
    expectOwnerCall(1, { answer: '', request_id: 'request-batch' })
    expect(ambient).not.toHaveBeenCalled()
  })
})

describe('ClarifyTool visible-card scoping', () => {
  const BACKGROUND_SESSION = 'session-background'
  const FOREGROUND_SESSION = 'session-foreground'
  const BACKGROUND_REQUEST = 'request-background'
  const FOREGROUND_REQUEST = 'request-foreground'
  const ZONE_A_SESSION = 'session-zone-a'
  const ZONE_B_SESSION = 'session-zone-b'
  const ZONE_A_REQUEST = 'request-zone-a'
  const ZONE_B_REQUEST = 'request-zone-b'
  const QUESTION = 'Which deployment target?'

  afterEach(() => {
    $activeTreeGroup.set(null)
    $hoveredTreeGroup.set(null)
  })

  /** Minimal per-session view — the pending card only reads `$runtimeId`. */
  function tileView(sessionId: string): SessionView {
    return { ...({} as SessionView), $runtimeId: atom<null | string>(sessionId), kind: 'tile' }
  }

  function pendingCardProps(toolCallId: string): ToolCallMessagePartProps {
    const args = { choices: ['staging', 'production'], question: QUESTION }

    return { ...liveClarifyProps(), args, argsText: JSON.stringify(args), toolCallId }
  }

  function parkClarify(requestId: string, sessionId: string) {
    setClarifyRequest({
      choices: ['staging', 'production'],
      multiSelect: false,
      question: QUESTION,
      requestId,
      sessionId
    })
  }

  /** A card inside an inactive tab layer — mounted and live, just not on screen. */
  function backgroundCard() {
    return (
      <div {...hiddenPaneProps(true)}>
        <SessionViewProvider value={tileView(BACKGROUND_SESSION)}>
          <ClarifyTool {...pendingCardProps('clarify-background')} />
        </SessionViewProvider>
      </div>
    )
  }

  it('answers the visible card, not a background one that mounted first', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true })

    $gateway.set({ request } as never)
    parkClarify(BACKGROUND_REQUEST, BACKGROUND_SESSION)
    parkClarify(FOREGROUND_REQUEST, FOREGROUND_SESSION)

    // The background card is rendered FIRST, so its window listener registers
    // first. Registration order used to decide the winner, which meant the card
    // the user was looking at lost to one parked in an inactive tab.
    renderClarify(
      <>
        {backgroundCard()}
        <SessionViewProvider value={tileView(FOREGROUND_SESSION)}>
          <ClarifyTool {...pendingCardProps('clarify-foreground')} />
        </SessionViewProvider>
      </>
    )

    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(1)
    })

    // Exactly one answer, carrying the FOREGROUND request id — the background
    // session's turn must not be resumed by a keystroke aimed at this one.
    expect(request).toHaveBeenCalledWith('clarify.respond', {
      answer: 'staging',
      request_id: FOREGROUND_REQUEST
    })
  })

  it('leaves the key alone when the only pending card is hidden', () => {
    const request = vi.fn().mockResolvedValue({ ok: true })

    $gateway.set({ request } as never)
    parkClarify(BACKGROUND_REQUEST, BACKGROUND_SESSION)

    renderClarify(backgroundCard())

    // Untouched (no preventDefault) ⇒ the keystroke stays available to the
    // composer, matching what `clarifyCardOwnsKey` reports with no visible card.
    expect(fireEvent.keyDown(window, { key: 'Enter' })).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  /** A card in its own split zone — unlike `backgroundCard` this one IS on
   *  screen, so a split renders two cards that both clear the hidden-pane
   *  filter and only the zone ladder can tell apart. */
  function zoneCard(zone: string, sessionId: string) {
    return (
      <div data-tree-group={zone}>
        <SessionViewProvider value={tileView(sessionId)}>
          <ClarifyTool {...pendingCardProps(`clarify-${zone}`)} />
        </SessionViewProvider>
      </div>
    )
  }

  /** Both zones visible, zone-a first in document order. */
  function renderSplit() {
    const request = vi.fn().mockResolvedValue({ ok: true })

    $gateway.set({ request } as never)
    parkClarify(ZONE_A_REQUEST, ZONE_A_SESSION)
    parkClarify(ZONE_B_REQUEST, ZONE_B_SESSION)

    renderClarify(
      <>
        {zoneCard('zone-a', ZONE_A_SESSION)}
        {zoneCard('zone-b', ZONE_B_SESSION)}
      </>
    )

    return request
  }

  it('answers the later-in-document card when its zone is the focused one', async () => {
    const request = renderSplit()

    $activeTreeGroup.set('zone-b')
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(1)
    })

    // Both cards are visible and both hold a live window listener, so this is
    // the case document order gets wrong: it would answer zone-a's question.
    expect(request).toHaveBeenCalledWith('clarify.respond', {
      answer: 'staging',
      request_id: ZONE_B_REQUEST
    })
  })

  it('answers the other visible card once the focus moves to its zone', async () => {
    const request = renderSplit()

    $activeTreeGroup.set('zone-a')
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(1)
    })

    // The direct pin for "the other visible card then cannot receive its
    // shortcut": neither zone may be permanently starved of its own keys.
    expect(request).toHaveBeenCalledWith('clarify.respond', {
      answer: 'staging',
      request_id: ZONE_A_REQUEST
    })
  })
})
