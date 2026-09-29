import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { hiddenPaneProps } from '@/components/pane-shell/pane-visibility'
import { $activeTreeGroup, $hoveredTreeGroup } from '@/components/pane-shell/tree/store'
import { I18nProvider } from '@/i18n'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { $profiles } from '@/store/profile'
import { hasOpenServerRequest, rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { readClarifyResult } from './parse'

import { ClarifyTool } from './index'

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

/** Seed a live `clarify` server request; the card answers it synchronously with
 *  `respondToServerRequest`, so the returned spy sees the response frame. */
function liveServerRequest(id: string) {
  const respond = vi.fn()
  rememberServerRequest({ fail: vi.fn(), id, method: 'clarify', params: {}, respond })

  return respond
}

beforeEach(() => {
  resetServerRequestsForTests()
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
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
  const args = { questions: [{ choices, question: 'Which deployment target?' }] }

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

function lock(answer: string, requestId = 'request-1') {
  return ['clarify.lock', { answer, question_id: 'q0', request_id: requestId }]
}

function renderLiveClarify({ multiSelect = false }: { multiSelect?: boolean } = {}) {
  const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
  const respond = liveServerRequest('request-1')

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)
  setClarifyRequest({
    questions: [{ choices: ['staging', 'production'], multiSelect, qid: 'q0', question: 'Which deployment target?' }],
    requestId: 'request-1',
    sessionId: 'session-1'
  })
  const { rerender } = renderClarify(<ClarifyTool {...liveClarifyProps()} />)

  return { request, rerender, respond }
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
    expect(screen.queryByRole('button', { name: /Confirm and continue/ })).toBeNull()
  })

  it('holds the card through the gap between answering and the settled result', async () => {
    const { request, rerender } = renderLiveClarify()

    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalled()
    })

    // tool.complete is what swaps in the settled card; the turn can already
    // read as not-running in that gap.
    messageRunning = false
    rerender(clarifyTree(<ClarifyTool {...liveClarifyProps()} />))

    expect(document.querySelector('form[data-clarify-batch]')).toBeTruthy()
  })

  it('demotes when the turn is stopped after the card was live but never answered', () => {
    renderLiveClarify()

    expect(document.querySelector('[data-clarify-choices]')).toBeTruthy()

    messageRunning = false
    act(() => clearClarifyRequest('request-1', 'session-1'))

    expect(document.querySelector('[data-clarify-choices]')).toBeNull()
    expect(screen.queryByRole('button', { name: /Confirm and continue/ })).toBeNull()
  })

  it('paints the question from tool args instead of a spinner while request_id is still racing', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn() } as never)
    renderClarify(<ClarifyTool {...liveClarifyProps()} />)

    expect(screen.getByText('Which deployment target?')).toBeTruthy()
    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(screen.getByRole('button', { name: /Confirm and continue/ }).hasAttribute('disabled')).toBe(true)
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

    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock(JSON.stringify(['production', 'staging'])))
    })
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
      expect(request).toHaveBeenCalledWith(...lock('production'))
    })
  })

  it('stages a highlighted multi-select choice with Enter and submits it with Confirm', async () => {
    const { request } = renderLiveClarify({ multiSelect: true })
    const production = screen.getByRole('button', { name: /production/ })

    fireEvent.keyDown(window, { key: 'ArrowDown' })
    fireEvent.keyDown(window, { key: 'Enter' })

    expect(production.getAttribute('aria-pressed')).toBe('true')
    expect(request).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock(JSON.stringify(['production'])))
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
    const { request, respond } = renderLiveClarify()
    const skip = screen.getByRole('button', { name: 'Skip' })

    skip.focus()

    expect(fireEvent.keyDown(window, { key: 'Enter' })).toBe(true)
    expect(fireEvent.keyDown(window, { key: 'ArrowDown' })).toBe(true)
    expect(respond).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })

  it('confirms a clicked choice with Enter while the choice button keeps focus', async () => {
    const { request } = renderLiveClarify()
    const production = screen.getByRole('button', { name: /production/ })

    // Click selects the choice; in a real browser the option button keeps
    // focus afterwards. jsdom's fireEvent.click does not move focus, so
    // focus it explicitly to reproduce the reported bug.
    fireEvent.click(production)
    production.focus()
    expect(production.getAttribute('aria-pressed')).toBe('true')
    expect(document.activeElement).toBe(production)

    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock('production'))
    })
  })

  it('picks the highlighted choice with Enter when a choice button is focused but not clicked, then confirms', async () => {
    const { request } = renderLiveClarify()
    const production = screen.getByRole('button', { name: /production/ })

    production.focus()
    expect(production.getAttribute('aria-pressed')).toBe('false')
    expect(document.activeElement).toBe(production)

    fireEvent.keyDown(window, { key: 'Enter' })
    expect(screen.getByRole('button', { name: /staging/ }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock('staging'))
    })
  })

  it('toggles a focused multi-select row with Enter and confirms the set with Confirm', async () => {
    const { request } = renderLiveClarify({ multiSelect: true })
    const staging = screen.getByRole('button', { name: /staging/ })
    const production = screen.getByRole('button', { name: /production/ })

    fireEvent.click(staging)
    fireEvent.click(production)
    production.focus()
    expect(document.activeElement).toBe(production)
    expect(staging.getAttribute('aria-pressed')).toBe('true')
    expect(production.getAttribute('aria-pressed')).toBe('true')

    fireEvent.keyDown(window, { key: 'Enter' })

    expect(production.getAttribute('aria-pressed')).toBe('false')
    expect(staging.getAttribute('aria-pressed')).toBe('true')
    expect(request).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock(JSON.stringify(['staging'])))
    })
  })
})

describe('ClarifyTool recommended option', () => {
  it('dims the (Recommended) label and answers with the bare choice', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
    liveServerRequest('request-1')

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest({
      questions: [
        {
          choices: ['staging (Recommended)', 'production'],
          multiSelect: false,
          qid: 'q0',
          question: 'Which deployment target?'
        }
      ],
      requestId: 'request-1',
      sessionId: 'session-1'
    })
    renderClarify(<ClarifyTool {...liveClarifyProps(['staging (Recommended)', 'production'])} />)

    const recommended = screen.getByRole('button', { name: /staging/ })

    fireEvent.click(recommended)
    fireEvent.keyDown(window, { key: 'Enter' })

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(...lock('staging'))
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
  const respond = liveServerRequest('request-batch')

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)
  setClarifyRequest({
    lockedAnswers,
    questions: [
      { choices: ['red', 'blue'], multiSelect, qid: 'q0', question: 'Color?' },
      { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
    ],
    requestId: 'request-batch',
    sessionId: 'session-1'
  })
  renderClarify(<ClarifyTool {...liveBatchProps()} />)

  return { request, respond }
}

describe('readClarifyResult', () => {
  it('parses responses with string and list answers', () => {
    const parsed = readClarifyResult(
      JSON.stringify({
        responses: [
          { question: 'Color?', user_response: 'red' },
          { question: 'Tools?', user_response: ['a', 'b'] },
          { question: 'Name?', user_response: '' }
        ]
      })
    )

    expect(parsed.responses).toHaveLength(3)
    expect(parsed.responses[1]?.answer).toEqual(['a', 'b'])
    expect(parsed.responses[2]?.answer).toBe('')
  })

  it('returns empty responses for single-question payloads', () => {
    expect(readClarifyResult({ question: 'Q?', user_response: 'a' }).responses).toEqual([])
  })
})

describe('ClarifyTool submit shortcut', () => {
  it('submits selected choices with Cmd/Ctrl+Enter without toggling the focused option', async () => {
    for (const modifier of [{ metaKey: true }, { ctrlKey: true }]) {
      const { request } = renderLiveClarify({ multiSelect: true })
      const choice = screen.getByRole('button', { name: /staging/ })
      fireEvent.click(choice)
      choice.focus()
      fireEvent.keyDown(choice, { key: 'Enter', ...modifier })
      await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
      expect(request).toHaveBeenCalledWith(...lock('["staging"]'))
      cleanup()
    }
  })

  it('submits the batch from its text field while preserving multiline input', async () => {
    for (const modifier of [{ metaKey: true }, { ctrlKey: true }]) {
      const { request } = renderLiveBatch()
      const field = screen.getByPlaceholderText('Type your answer…')
      fireEvent.change(field, { target: { value: 'packet' } })
      field.focus()
      fireEvent.keyDown(field, { key: 'Enter', shiftKey: true })
      fireEvent.keyDown(field, { key: 'Enter', isComposing: true, ...modifier })
      expect(request).not.toHaveBeenCalled()
      fireEvent.keyDown(field, { key: 'Enter', ...modifier })
      await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
      expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
        answer: null,
        question_id: 'q0',
        request_id: 'request-batch'
      })
      expect(request).toHaveBeenNthCalledWith(2, 'clarify.lock', {
        answer: 'packet',
        question_id: 'q1',
        request_id: 'request-batch'
      })
      cleanup()
    }
  })
})

describe('ClarifyTool batch card', () => {
  // #112855: the batch card spun forever while the gateway clarify request
  // raced (or never came). The question text is already in the tool args.
  it('paints batch questions from tool args while the gateway request is still racing', () => {
    $activeSessionId.set('session-1')
    $gateway.set({ request: vi.fn() } as never)
    renderClarify(<ClarifyTool {...liveBatchProps()} />)

    expect(screen.getByText('Color?')).toBeTruthy()
    expect(screen.getByText('Name?')).toBeTruthy()
    // The spinner card is gone; the preview keeps only a screen-reader loading cue.
    expect(screen.queryByRole('status', { name: /loading question/i })).toBeNull()
    expect(screen.getByRole('status').textContent).toMatch(/loading question/i)
    expect(document.querySelector('[data-clarify-batch-preview]')?.getAttribute('aria-busy')).toBe('true')
    // Nothing is answerable yet: no qids to respond with.
    expect((screen.getByRole('button', { name: /red/ }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: /Skip/ }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: /Confirm and continue/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  // #98645: when the request never reaches this window the preview used to
  // sit disabled for the tool's whole 300s timeout with nothing said.
  it('asks the backend to re-deliver a request that never arrived, then says it did not reach the app', async () => {
    vi.useFakeTimers()

    try {
      const request = vi.fn(async (..._args: unknown[]) => ({ events: [], open_requests: [] }))
      $activeSessionId.set('session-1')
      $gateway.set({ request } as never)
      renderClarify(<ClarifyTool {...liveBatchProps()} />)

      expect(screen.queryByText(/didn't reach the app/)).toBeNull()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(4_000)
      })

      expect(request.mock.calls[0]?.slice(0, 2)).toEqual([
        'session.events.since',
        { last_seen: Number.MAX_SAFE_INTEGER, session_id: 'session-1' }
      ])
      expect(screen.getByRole('status').textContent).toMatch(/didn't reach the app.*Press Stop/)
      expect(document.querySelector('[data-clarify-batch-preview]')?.getAttribute('aria-busy')).toBeNull()
      // Nothing on the card can answer, so the actions go rather than sit disabled.
      expect(screen.queryByRole('button', { name: /Confirm and continue/ })).toBeNull()
      expect(screen.queryByRole('button', { name: /Skip/ })).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('goes live when the re-delivery finds the request still open on the backend', async () => {
    vi.useFakeTimers()

    try {
      liveServerRequest('request-batch')

      // The channel re-delivers `open_requests` to the request handlers (which
      // park the clarify) before the call resolves.
      const request = vi.fn(async (method: string) => {
        if (method === 'session.events.since') {
          setClarifyRequest({
            questions: [
              { choices: ['red', 'blue'], multiSelect: false, qid: 'q0', question: 'Color?' },
              { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
            ],
            requestId: 'request-batch',
            sessionId: 'session-1'
          })
        }

        return { events: [], open_requests: [] }
      })

      $activeSessionId.set('session-1')
      $gateway.set({ request } as never)
      renderClarify(<ClarifyTool {...liveBatchProps()} />)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(4_000)
      })

      expect((screen.getByRole('button', { name: /red/ }) as HTMLButtonElement).disabled).toBe(false)
      expect(document.querySelector('[data-clarify-batch-preview]')).toBeNull()
      expect(screen.queryByText(/didn't reach the app/)).toBeNull()
    } finally {
      vi.useRealTimers()
    }
  })

  it('swaps the preview for the live form and answers with the request qids', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    const { rerender } = renderClarify(<ClarifyTool {...liveBatchProps()} />)

    expect(document.querySelector('[data-clarify-batch-preview]')).toBeTruthy()

    act(() => {
      liveServerRequest('request-batch')
      setClarifyRequest({
        questions: [
          { choices: ['red', 'blue'], multiSelect: false, qid: 'q0', question: 'Color?' },
          { choices: null, multiSelect: false, qid: 'q1', question: 'Name?' }
        ],
        requestId: 'request-batch',
        sessionId: 'session-1'
      })
    })
    rerender(clarifyTree(<ClarifyTool {...liveBatchProps()} />))

    expect(document.querySelector('[data-clarify-batch-preview]')).toBeNull()
    expect(screen.getByText('0 of 2 answered')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    // Locks ride the live qids, never the preview's synthetic ones.
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
      answer: 'red',
      question_id: 'q0',
      request_id: 'request-batch'
    })
    expect(request).toHaveBeenNthCalledWith(2, 'clarify.lock', {
      answer: 'packet',
      question_id: 'q1',
      request_id: 'request-batch'
    })
  })

  it('stages locally and enables the single confirm once one question is answered', async () => {
    const { request } = renderLiveBatch()
    const confirm = screen.getByRole('button', { name: /Confirm and continue/ })

    expect((confirm as HTMLButtonElement).disabled).toBe(true)

    // Staging a pick sends NOTHING to the server.
    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    expect(screen.getByText('1 of 2 answered')).toBeTruthy()
    expect(request).not.toHaveBeenCalled()
    expect((confirm as HTMLButtonElement).disabled).toBe(false)

    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    expect(screen.getByText('2 of 2 answered')).toBeTruthy()
    expect(request).not.toHaveBeenCalled()
    expect((confirm as HTMLButtonElement).disabled).toBe(false)
  })

  it('confirm sends every per-question clarify.lock in order and completes the batch', async () => {
    const { request } = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
      answer: 'red',
      question_id: 'q0',
      request_id: 'request-batch'
    })
    expect(request).toHaveBeenNthCalledWith(2, 'clarify.lock', {
      answer: 'packet',
      question_id: 'q1',
      request_id: 'request-batch'
    })
    // The last lock resolves the server request; the card forgets it locally.
    await waitFor(() => expect(hasOpenServerRequest('request-batch')).toBe(false))
  })

  it('a staged answer stays editable before confirm', async () => {
    const { request } = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.click(screen.getByRole('button', { name: /blue/ }))
    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    // The re-pick won: blue, not red.
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
      answer: 'blue',
      question_id: 'q0',
      request_id: 'request-batch'
    })
  })

  it('keeps a picked multi-select batch choice when typing a custom answer alongside it', async () => {
    const { request } = renderLiveBatch(undefined, true)

    fireEvent.click(screen.getByRole('button', { name: /red/ }))
    fireEvent.change(screen.getByPlaceholderText(/Other/), { target: { value: 'green' } })

    // Typing in "Other" must not silently drop the pick already made.
    expect(screen.getByRole('button', { name: /red/ }).getAttribute('aria-pressed')).toBe('true')

    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
      answer: JSON.stringify(['red', 'green']),
      question_id: 'q0',
      request_id: 'request-batch'
    })
  })

  it('keeps a typed multi-select batch custom answer when picking a choice after typing it', async () => {
    const { request } = renderLiveBatch(undefined, true)
    const other = screen.getByPlaceholderText(/Other/)

    fireEvent.change(other, { target: { value: 'green' } })
    fireEvent.click(screen.getByRole('button', { name: /red/ }))

    // Picking a choice must not silently discard the text already typed.
    expect((other as HTMLInputElement).value).toBe('green')
    expect(screen.getByRole('button', { name: /red/ }).getAttribute('aria-pressed')).toBe('true')

    fireEvent.change(screen.getByPlaceholderText('Type your answer…'), { target: { value: 'packet' } })
    fireEvent.submit(document.querySelector('form') as HTMLFormElement)

    await waitFor(() => {
      expect(request).toHaveBeenCalledTimes(2)
    })
    expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
      answer: JSON.stringify(['red', 'green']),
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

  it('Skip cancels the whole batch with an empty response (no answers)', async () => {
    const { request, respond } = renderLiveBatch()

    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))

    await waitFor(() => {
      expect(respond).toHaveBeenCalledWith({})
    })
    expect(request).not.toHaveBeenCalled()
  })

  // ─── Single-entry batch (one-question questions[]) ────────────────────────
  // #95907 made `questions[]` the only advertised shape, so a single question
  // now arrives as a one-entry batch on BOTH sides: tool args carry
  // `questions:[{question, choices}]` and the gateway wire carries
  // `questions:[{qid, question, choices}]` with no top-level question. The
  // batch card must mount for the one-entry case exactly as it does for 2+.

  function singleBatchArgs(): { questions: { question: string; choices: string[] }[] } {
    return {
      questions: [
        {
          choices: ['Local Markdown under .scratch/', 'GitHub Issues', 'Linear', 'GitLab Issues'],
          question: 'Which issue tracker should this repository use?'
        }
      ]
    }
  }

  function liveSingleBatchProps(): ToolCallMessagePartProps {
    const args = singleBatchArgs()

    return {
      addResult: vi.fn(),
      args,
      argsText: JSON.stringify(args),
      isError: false,
      respondToApproval: vi.fn(),
      result: undefined,
      resume: vi.fn(),
      status: { type: 'running' },
      toolCallId: 'clarify-single-batch',
      toolName: 'clarify',
      type: 'tool-call'
    }
  }

  function singleBatchRequest() {
    return {
      questions: [
        {
          choices: ['Local Markdown under .scratch/', 'GitHub Issues', 'Linear', 'GitLab Issues'],
          multiSelect: false,
          qid: 'q0',
          question: 'Which issue tracker should this repository use?'
        }
      ],
      requestId: 'request-single-batch',
      sessionId: 'session-1'
    }
  }

  it('renders a one-entry batch as the batch card, not a blank/spinner single card', () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)
    setClarifyRequest(singleBatchRequest())
    renderClarify(<ClarifyTool {...liveSingleBatchProps()} />)

    expect(screen.getByText('Which issue tracker should this repository use?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /GitHub Issues/ })).toBeTruthy()
    expect(document.querySelector('form[data-clarify-batch]')?.getAttribute('data-clarify-batch')).toBe('1')
  })

  it('mounts the batch card once the wire request lands after the tool row (#98645 timing)', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })

    $activeSessionId.set('session-1')
    $gateway.set({ request } as never)

    // Tool row mounts FIRST with the model's args; the gateway wire (with the
    // qid the renderer needs) arrives a beat later — same ordering as
    // tool.start → clarify.request in a live session.
    const { rerender } = renderClarify(<ClarifyTool {...liveSingleBatchProps()} />)

    await act(async () => {
      setClarifyRequest(singleBatchRequest())
    })
    rerender(clarifyTree(<ClarifyTool {...liveSingleBatchProps()} />))

    await waitFor(() => {
      expect(screen.getByText('Which issue tracker should this repository use?')).toBeTruthy()
    })
    expect(screen.getByRole('button', { name: /GitHub Issues/ })).toBeTruthy()

    // And the single pick answers with the qid-keyed lock.
    fireEvent.click(screen.getByRole('button', { name: /GitHub Issues/ }))
    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith('clarify.lock', {
        answer: 'GitHub Issues',
        question_id: 'q0',
        request_id: 'request-single-batch'
      })
    })
  })

  it('renders the settled batch with all questions and answers', () => {
    renderClarify(
      <ClarifyTool
        {...settledClarifyProps(
          batchArgs(),
          JSON.stringify({
            outcome: 'submitted',
            responses: [
              { choices_offered: ['red', 'blue'], question: 'Color?', status: 'answered', user_response: 'red' },
              { choices_offered: null, question: 'Name?', status: 'skipped', user_response: null }
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
// tool times out. A clarify answer is now the server request's own response
// frame — it rides the socket the request arrived on, so no routing decision is
// left to make. Only `clarify.lock` (a real client→server RPC) still dials a
// socket, and it must route by request.sessionId.

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

function expectOwnerLock(nth: number, params: Record<string, unknown>) {
  expect(gatewayMocks.requestGatewayForAgent).toHaveBeenNthCalledWith(
    nth,
    OWNER_CONNECTION_ID,
    OWNER_PROFILE,
    'clarify.lock',
    params
  )
}

describe('ClarifyTool owner routing', () => {
  afterEach(() => {
    $profiles.set([])
    _resetSessionOwnerHintsForTests({ storage: true })
    gatewayMocks.requestGatewayForAgent.mockClear()
  })

  it('sends both sequential batch locks on the owner socket, in order', async () => {
    const ambient = armCrossProfileOwner()
    liveServerRequest('request-batch')

    setClarifyRequest({
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
    expectOwnerLock(1, { answer: 'red', question_id: 'q0', request_id: 'request-batch' })
    expectOwnerLock(2, { answer: 'packet', question_id: 'q1', request_id: 'request-batch' })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('answers a batch skip/cancel through its server request, never profile B ambient', async () => {
    const ambient = armCrossProfileOwner()
    const respond = liveServerRequest('request-batch')

    setClarifyRequest({
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
      expect(respond).toHaveBeenCalledWith({})
    })
    expect(gatewayMocks.requestGatewayForAgent).not.toHaveBeenCalled()
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
    return { ...liveClarifyProps(), toolCallId }
  }

  function parkClarify(requestId: string, sessionId: string) {
    liveServerRequest(requestId)

    setClarifyRequest({
      questions: [{ choices: ['staging', 'production'], multiSelect: false, qid: 'q0', question: QUESTION }],
      requestId,
      sessionId
    })
  }

  function lockedRequestIds(request: ReturnType<typeof vi.fn>) {
    return request.mock.calls
      .filter(call => call[0] === 'clarify.lock')
      .map(call => (call[1] as { request_id: string }).request_id)
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
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
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
    fireEvent.keyDown(window, { key: 'Enter' })

    // Exactly one answer, on the FOREGROUND request — the background session's
    // turn must not be resumed by a keystroke aimed at this one.
    await waitFor(() => {
      expect(lockedRequestIds(request)).toEqual([FOREGROUND_REQUEST])
    })
  })

  it('leaves the key alone when the only pending card is hidden', () => {
    const request = vi.fn()
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
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
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
    fireEvent.keyDown(window, { key: 'Enter' })

    // Both cards are visible and both hold a live window listener, so this is
    // the case document order gets wrong: it would answer zone-a's question.
    await waitFor(() => {
      expect(lockedRequestIds(request)).toEqual([ZONE_B_REQUEST])
    })
  })

  it('answers the other visible card once the focus moves to its zone', async () => {
    const request = renderSplit()

    $activeTreeGroup.set('zone-a')
    fireEvent.keyDown(window, { key: 'Enter' })
    fireEvent.keyDown(window, { key: 'Enter' })

    // The direct pin for "the other visible card then cannot receive its
    // shortcut": neither zone may be permanently starved of its own keys.
    await waitFor(() => {
      expect(lockedRequestIds(request)).toEqual([ZONE_A_REQUEST])
    })
  })
})
