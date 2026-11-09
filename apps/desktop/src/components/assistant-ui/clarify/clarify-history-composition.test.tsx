import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { I18nProvider } from '@/i18n'
import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { clearClarifyRequest } from '@/store/clarify'
import { $activeSessionId } from '@/store/session'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

const sessionId = 'clarify-history-session'

const history: SessionMessage[] = [
  {
    role: 'assistant',
    content: 'I was checking the deployment target.',
    timestamp: 10,
    display_metadata: '{"interrupted":true}',
    tool_calls: [
      {
        id: 'clarify-history-call',
        type: 'function',
        function: {
          name: 'clarify',
          arguments: JSON.stringify({ questions: [{ choices: ['staging', 'production'], question: 'Which target?' }] })
        }
      }
    ]
  }
]

afterEach(() => {
  cleanup()
  clearClarifyRequest(undefined, sessionId)
  $activeSessionId.set(null)
})

describe('rehydrated Clarify history composition', () => {
  it('demotes an interrupted historical call through the real converter and renderer', async () => {
    $activeSessionId.set(sessionId)
    const [historical] = toChatMessages(history)
    const clarifyPart = historical.parts.find(part => part.type === 'tool-call')

    expect(historical.interrupted).toBe(true)
    expect(clarifyPart).toMatchObject({ args: { questions: [{ question: 'Which target?' }] }, toolName: 'clarify' })
    expect(clarifyPart?.completedAt).toBeUndefined()

    const runtimeMessage = toRuntimeMessage(historical)

    expect(runtimeMessage.status?.type).toBe('complete')
    expect(runtimeMessage.metadata?.custom?.interrupted).toBe(true)

    const { container } = render(
      <I18nProvider configClient={null} initialLocale="en">
        <SessionViewProvider value={PRIMARY_SESSION_VIEW}>
          <ThreadRuntime messages={[runtimeMessage]}>
            <Thread />
          </ThreadRuntime>
        </SessionViewProvider>
      </I18nProvider>
    )

    expect(await screen.findByText('I was checking the deployment target.')).toBeTruthy()
    expect(container.querySelector('[data-clarify-choices]')).toBeNull()
    expect(screen.queryByText('Which target?')).toBeNull()
    expect(screen.queryByRole('button', { name: /Confirm and continue/ })).toBeNull()
  })
})
