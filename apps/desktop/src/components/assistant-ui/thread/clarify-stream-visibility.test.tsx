import { AssistantRuntimeProvider, type ThreadMessage } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import { createClientSessionState } from '@/lib/chat-runtime'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { resetServerRequestsForTests } from '@/store/server-requests'
import { $activeSessionId } from '@/store/session'
import { $sessionStates, clearAllSessionStates } from '@/store/session-states'
import { holdTranscriptView } from '@/store/session-transcript-view'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('@/components/assistant-ui/wisdom-candidate-card', () => ({ WisdomCandidateCard: () => null }))
vi.mock('@/components/assistant-ui/wisdom-notice-card', () => ({ WisdomNoticeCard: () => null }))
vi.mock('@/components/wisdom-mediation-card', () => ({ WisdomMediationCard: () => null }))

stubThreadEnvironment()
stubThreadViewportSize()

function FullThread() {
  const view = PRIMARY_SESSION_VIEW
  const messages = useStore(view.$messages)
  const busy = useStore(view.$busy)
  const sessionId = useStore(view.$runtimeId)
  const repository = useRuntimeMessageRepository(messages)

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: busy,
    onNew: async () => {}
  })

  return (
    <SessionViewProvider value={view}>
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread sessionId={sessionId} />
      </AssistantRuntimeProvider>
    </SessionViewProvider>
  )
}

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  clearClarifyRequest()
  resetServerRequestsForTests()
  $activeSessionId.set(null)
  $gateway.set(null)
})

it.each(['answer', 'cancel', 'finish', 'batch'] as const)(
  'the real Thread handles %s for a background replay while history remains held',
  async ending => {
    const state = createClientSessionState('stored-A')
    state.messages = [{ id: 'old', role: 'assistant', parts: [{ type: 'text', text: 'unverified history' }] }]
    state.busy = true
    const states = new Map([['A', state]])
    const activeSessionIdRef = { current: 'B' as string | null }
    $activeSessionId.set('B')
    const request = vi.fn().mockResolvedValue({ ok: true, remaining: [] })
    $gateway.set({ request } as never)
    $sessionStates.set({ A: state, B: createClientSessionState('stored-B') })

    const stream = renderMessageStream('B', {
      states,
      activeSessionIdRef,
      updateSessionState: (id, updater) => {
        const next = updater(states.get(id) ?? createClientSessionState())
        states.set(id, next)
        $sessionStates.set({ ...$sessionStates.get(), [id]: next })

        return next
      }
    })

    holdTranscriptView('A', Symbol('test'), state.messages)
    const app = render(<FullThread />)
    let respond!: ReturnType<typeof vi.fn>
    act(() => {
      respond = stream.handleRequest(
        'clarify',
        ending === 'batch'
          ? {
              session_id: 'A',
              questions: [
                { qid: 'q0', question: 'Which path?', choices: ['safe', 'fast'] },
                { qid: 'q1', question: 'Name?' }
              ],
              answers: { q0: 'safe' }
            }
          : { session_id: 'A', question: 'Which path?', choices: ['safe', 'fast'] },
        'req-A'
      )
    })
    expect(app.queryByText('Which path?')).toBeNull()
    act(() => {
      activeSessionIdRef.current = 'A'
      $activeSessionId.set('A')
    })
    await waitFor(() => expect(app.getByText('Which path?')).toBeTruthy())
    expect(app.queryByText('unverified history')).toBeNull()
    const choice = app.getByRole('button', { name: /safe/ })

    for (let node: HTMLElement | null = choice; node; node = node.parentElement) {
      expect(node.hidden).toBe(false)
      expect(getComputedStyle(node).display).not.toBe('none')
      expect(getComputedStyle(node).visibility).not.toBe('hidden')
    }

    if (ending === 'batch') {
      expect(app.getByText('1 of 2 answered')).toBeTruthy()
      fireEvent.change(app.getByPlaceholderText('Type your answer…'), { target: { value: 'test name' } })
      fireEvent.click(app.getByRole('button', { name: /Confirm and continue/ }))
      await waitFor(() => expect(request).toHaveBeenCalledTimes(2))
      expect(request).toHaveBeenNthCalledWith(1, 'clarify.lock', {
        request_id: 'req-A',
        question_id: 'q0',
        answer: 'safe'
      })
      expect(request).toHaveBeenNthCalledWith(2, 'clarify.lock', {
        request_id: 'req-A',
        question_id: 'q1',
        answer: 'test name'
      })
    } else if (ending === 'answer') {
      fireEvent.click(choice)
      fireEvent.click(app.getByRole('button', { name: 'Continue' }))
      await waitFor(() => expect(respond).toHaveBeenCalledExactlyOnceWith({ answer: 'safe' }))
    } else {
      act(() =>
        stream.handleEvent({
          session_id: 'A',
          type: ending === 'cancel' ? 'request.cancel' : 'message.complete',
          payload: ending === 'cancel' ? { id: 'req-A', method: 'clarify', reason: 'timeout' } : {}
        })
      )
    }

    // Settled cards may offer a follow-up choice; the blocking request form must disappear.
    await waitFor(() => expect(app.container.querySelector('[data-clarify-choices]')).toBeNull())
    expect(app.container.querySelector('[data-clarify-batch]')).toBeNull()
    expect($clarifyRequests.get()['A']).toBeUndefined()
    expect(app.queryByText('unverified history')).toBeNull()
  }
)
