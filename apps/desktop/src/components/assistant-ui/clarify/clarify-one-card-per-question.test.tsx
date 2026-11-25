import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { chatMessageArraysEquivalent } from '@/app/session/hooks/use-session-actions/message-equivalence'
import { pendingClarifyToolPayload } from '@/app/session/hooks/use-session-actions/restore-pending-clarify'
import { I18nProvider } from '@/i18n'
import { type ChatMessage, restorePendingClarifyToolCall } from '@/lib/chat-messages'
import { $settledClarifyResults, type ClarifyRequest, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { answerParkedCard } from '@/store/parked-card-answers'
import { rememberServerRequest, resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'

import { ClarifyTool } from './index'

// A canonical gateway asks a multi-question clarify one card per question when the
// submitter did not declare `clarify_batch` (gateway/run_turn_runner.py::_clarify_callback_sync).
// Each card is a new request that re-arms the same tool row. These tests drive the seams
// production uses: the transcript re-arm `parkClarify` runs, the composer's
// `answerParkedCard`, and the card's own Skip.

const gatewayMocks = vi.hoisted(() => ({
  requestGatewayForAgent: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

vi.mock('@assistant-ui/react', () => ({ useAuiState: () => true }))

beforeAll(() => {
  // jsdom has no Web Animations API; a demoted row's enter animation would throw.
  Element.prototype.animate ??= (() => ({ cancel: () => {}, finished: Promise.resolve() })) as never
})

afterEach(() => {
  cleanup()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
  $settledClarifyResults.set({})
  vi.clearAllMocks()
})

const TARGET = { choices: ['staging', 'production'], question: 'Which deployment target?' }
const REGION = { choices: ['us', 'eu'], question: 'Which region?' }

const card = (requestId: string, entry: typeof TARGET): ClarifyRequest => ({
  questions: [{ choices: entry.choices, multiSelect: false, qid: `${requestId}-q`, question: entry.question }],
  receivedAt: Date.now() / 1000,
  requestId,
  sessionId: 'session-1'
})

const tree = (ui: ReactNode) => (
  <I18nProvider configClient={null} initialLocale="en">
    {ui}
  </I18nProvider>
)

type ClarifyPart = Extract<ChatMessage['parts'][number], { type: 'tool-call' }>

function rowProps(part: ClarifyPart): ToolCallMessagePartProps & { clarifyRequestId?: string } {
  return {
    addResult: vi.fn(),
    args: part.args,
    argsText: JSON.stringify(part.args),
    clarifyRequestId: part.clarifyRequestId,
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: part.toolCallId,
    toolName: 'clarify',
    type: 'tool-call'
  } as ToolCallMessagePartProps
}

function clarifyCall(toolCallId: string, questions: (typeof TARGET)[]) {
  return { args: { questions }, toolCallId, toolName: 'clarify', type: 'tool-call' } as never
}

const clarifyParts = (messages: ChatMessage[]) =>
  messages.flatMap(message => message.parts.filter((part): part is ClarifyPart => part.type === 'tool-call'))

/** Cards whose choices can be answered right now. */
const liveCards = (container: HTMLElement) =>
  [...container.querySelectorAll('[data-clarify-choices]')].filter(choices =>
    choices.querySelector('button:not([disabled])')
  ).length

/**
 * Assistant messages (one per argument) whose clarify rows re-arm the way `parkClarify` does. Rows
 * render from what the view last published: like `use-session-state-cache`, it takes a new
 * transcript only when `chatMessageArraysEquivalent` sees a change.
 */
function transcript(...rows: ReturnType<typeof clarifyCall>[][]) {
  return transcriptOf(rows, { pending: true })
}

/** A transcript hydrated from REST: its messages are settled until the first re-arm. */
function resumedTranscript(...rows: ReturnType<typeof clarifyCall>[][]) {
  return transcriptOf(rows, { pending: false })
}

function transcriptOf(rows: ReturnType<typeof clarifyCall>[][], { pending }: { pending: boolean }) {
  const request = vi.fn().mockResolvedValue({ ok: true })

  let messages: ChatMessage[] = rows.map((parts, index) => ({
    id: `m${index + 1}`,
    parts,
    pending,
    role: 'assistant'
  }))

  let shown = messages

  const publish = (next: ChatMessage[]) => {
    messages = next

    if (!chatMessageArraysEquivalent(next, shown)) {
      shown = next
    }
  }

  $activeSessionId.set('session-1')
  $gateway.set({ request } as never)

  return {
    add(call: ReturnType<typeof clarifyCall>) {
      const tail = messages.at(-1)!
      publish([...messages.slice(0, -1), { ...tail, parts: [...tail.parts, call] }])
    },
    locks: () =>
      [...request.mock.calls, ...gatewayMocks.requestGatewayForAgent.mock.calls].filter(call =>
        (call as unknown[]).includes('clarify.lock')
      ),
    /** Park the next card. `appends`: the re-arm cannot pick between open rows and adds its own. */
    park(next: ClarifyRequest, { appends = false } = {}) {
      rememberServerRequest({ fail: vi.fn(), id: next.requestId, method: 'clarify', params: {}, respond: vi.fn() })
      setClarifyRequest(next)
      const before = clarifyParts(messages).length
      publish(restorePendingClarifyToolCall(messages, pendingClarifyToolPayload(next)).messages)
      expect(clarifyParts(messages)).toHaveLength(before + (appends ? 1 : 0))
    },
    props: (toolCallId: string) => rowProps(clarifyParts(shown).find(part => part.toolCallId === toolCallId)!),
    /** Every clarify row the view shows, in transcript order. */
    rows: () => (
      <>
        {clarifyParts(shown).map((part, index) => (
          <ClarifyTool key={index} {...rowProps(part)} />
        ))}
      </>
    )
  }
}

describe('a clarify asked one card per question', () => {
  it('shows the next card after the chat bar answered the first, and the chat bar answers that card', async () => {
    const chat = transcript([clarifyCall('call-1', [TARGET, REGION])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(<ClarifyTool {...chat.props('call-1')} />))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })
    expect(view.container.querySelector('[data-clarify-settled]')).not.toBeNull()

    act(() => chat.park(card('prompt-2', REGION)))
    // The request can land before the row's re-armed args; the row stays a card meanwhile,
    // never a bare tool row.
    expect(view.container.querySelector('[data-clarify-settled], [data-clarify-batch]')).not.toBeNull()

    view.rerender(tree(<ClarifyTool {...chat.props('call-1')} />))
    expect(view.container.querySelector('[data-clarify-settled]')).toBeNull()
    expect(screen.getByText('Which region?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)

    await act(async () => {
      expect(answerParkedCard('session-1', 'eu')).toBe(true)
      await Promise.resolve()
    })
    expect(chat.locks().at(-1)).toEqual([
      'clarify.lock',
      { answer: 'eu', question_id: 'prompt-2-q', request_id: 'prompt-2' }
    ])
  })

  it('shows the next card after Skip on the first', () => {
    const chat = transcript([clarifyCall('call-1', [TARGET, REGION])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(<ClarifyTool {...chat.props('call-1')} />))

    fireEvent.click(screen.getByRole('button', { name: 'Skip' }))
    expect(view.container.querySelector('[data-clarify-settled]')).not.toBeNull()

    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(<ClarifyTool {...chat.props('call-1')} />))

    expect(screen.getByText('Which region?')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)
  })

  it("keeps an earlier call's answered row settled when a later call asks", () => {
    // The earlier row never got its tool.complete (a lost frame, or a duplicate row a re-delivered
    // activate left behind), so it is still a live clarify row when the later call's question lands.
    const chat = transcript([clarifyCall('call-1', [TARGET])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })

    chat.add(clarifyCall('call-2', [REGION]))
    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(chat.rows()))

    expect(view.container.querySelectorAll('[data-clarify-settled]')).toHaveLength(1)
    expect(screen.getAllByText('Which region?')).toHaveLength(1)
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)
  })

  it('keeps an earlier answered call settled when a later call asks the same question', () => {
    const chat = transcript([clarifyCall('call-1', [TARGET])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })

    chat.add(clarifyCall('call-2', [TARGET]))
    act(() => chat.park(card('prompt-2', TARGET)))
    view.rerender(tree(chat.rows()))

    expect(view.container.querySelectorAll('[data-clarify-settled]')).toHaveLength(1)
    expect(liveCards(view.container)).toBe(1)
  })

  it("does not show a later call's question on a card answered with its own Confirm", async () => {
    // Confirm latches the card until tool.complete; here that completion has not landed yet.
    const chat = transcript([clarifyCall('call-1', [TARGET])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))
    await waitFor(() => expect(chat.locks()).toHaveLength(1))

    chat.add(clarifyCall('call-2', [REGION]))
    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(chat.rows()))

    expect(liveCards(view.container)).toBe(1)
    expect(screen.getAllByText('Which region?')).toHaveLength(1)
  })

  it('shows each card once when the call is duplicated in a second message', () => {
    // A re-delivered activate can leave a copy of the call; neither copy's args single out a card,
    // so the re-arm adds its own row for each one.
    const chat = transcript([clarifyCall('call-1', [TARGET, REGION])], [clarifyCall('call-1', [TARGET, REGION])])
    chat.park(card('prompt-1', TARGET), { appends: true })
    const view = render(tree(chat.rows()))
    expect(liveCards(view.container)).toBe(1)

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })

    act(() => chat.park(card('prompt-2', REGION), { appends: true }))
    view.rerender(tree(chat.rows()))

    expect(liveCards(view.container)).toBe(1)
    expect(screen.getByRole('button', { name: /^.*eu/ }).hasAttribute('disabled')).toBe(false)
  })

  it('shows one card when a one-question call is duplicated', () => {
    const chat = transcript([clarifyCall('call-1', [TARGET])], [clarifyCall('call-1', [TARGET])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    expect(liveCards(view.container)).toBe(1)
  })

  // A resumed or switched-to session publishes the row with its first card's binding. The next card's
  // re-arm changes only the row's args and binding, and the view has to take that change.
  it('shows the next card after a resume when its own Confirm answered the first', async () => {
    const chat = resumedTranscript([clarifyCall('call-1', [TARGET, REGION])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    fireEvent.click(screen.getByRole('button', { name: /staging/ }))
    fireEvent.click(screen.getByRole('button', { name: /Confirm and continue/ }))
    await waitFor(() => expect(chat.locks()).toHaveLength(1))

    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(chat.rows()))

    expect(liveCards(view.container)).toBe(1)
    expect(screen.getByText('Which region?')).toBeTruthy()
  })

  it('shows the next card after a resume when the chat bar answered the first', () => {
    const chat = resumedTranscript([clarifyCall('call-1', [TARGET, REGION])])
    chat.park(card('prompt-1', TARGET))
    const view = render(tree(chat.rows()))

    act(() => {
      expect(answerParkedCard('session-1', 'staging')).toBe(true)
    })

    act(() => chat.park(card('prompt-2', REGION)))
    view.rerender(tree(chat.rows()))

    expect(liveCards(view.container)).toBe(1)
    expect(screen.getByText('Which region?')).toBeTruthy()
  })
})
