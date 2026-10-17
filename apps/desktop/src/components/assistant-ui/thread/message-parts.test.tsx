import type { ThreadMessage, ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { MESSAGE_PARTS_COMPONENTS } from '@/components/assistant-ui/thread/message-parts'
import { I18nProvider } from '@/i18n'
import { _resetToolCardsForTests, learnToolCard } from '@/lib/tool-render-class'
import { $activeSessionId } from '@/store/session'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

const SESSION = 'sess-card-test'
const GOAL = 'Inspect the delegate card'

function props(): ToolCallMessagePartProps {
  const args = { tasks: [{ goal: GOAL }] }

  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId: 'call-1',
    toolName: 'agents_run',
    type: 'tool-call'
  }
}

function view(): SessionView {
  return {
    $awaitingResponse: atom(false),
    $busy: atom(false),
    $cwd: atom(''),
    $fast: atom(false),
    $lastVisibleIsUser: atom(false),
    $messages: atom([]),
    $messagesEmpty: atom(false),
    $model: atom(''),
    $provider: atom(''),
    $reasoningEffort: atom(''),
    $reasoningEffortPending: atom(false),
    $reasoningEffortWire: atom(''),
    $runtimeId: atom(SESSION),
    $storedId: atom(null),
    $turnStartedAt: atom(null),
    kind: 'primary'
  }
}

function renderFallback() {
  const Fallback = MESSAGE_PARTS_COMPONENTS.tools.Fallback

  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <SessionViewProvider value={view()}>
        <Fallback {...props()} />
      </SessionViewProvider>
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  _resetToolCardsForTests()
  $activeSessionId.set(null)
})

describe('declared tool cards in the transcript', () => {
  it('renders a tool that declared the delegate card as the delegate card', () => {
    learnToolCard('agents_run', 'delegate')

    const { container } = renderFallback()

    expect(container.querySelector('[data-delegate-card]')).not.toBeNull()
    expect(container.textContent).toContain(GOAL)
  })

  // Without a declared card the call is a plain tool row; it needs the full
  // thread runtime the generic row reads message state from.
  it('does not render the delegate card for a tool that never declared one', async () => {
    const createdAt = new Date('2026-09-11T12:00:00Z')
    const args = { tasks: [{ goal: GOAL }] }

    const message = {
      id: 'assistant-agents-run',
      role: 'assistant',
      content: [
        { type: 'tool-call', toolCallId: 'call-1', toolName: 'agents_run', args, argsText: JSON.stringify(args) }
      ],
      status: { type: 'running' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as unknown as ThreadMessage

    const { container } = render(
      <ThreadRuntime messages={[message]}>
        <Thread />
      </ThreadRuntime>
    )

    await screen.findByText(/agents run/i)
    expect(container.querySelector('[data-delegate-card]')).toBeNull()
  })
})
