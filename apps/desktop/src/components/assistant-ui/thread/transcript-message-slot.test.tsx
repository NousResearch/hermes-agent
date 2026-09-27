import { type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { StrictMode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { registry } from '@/contrib/registry'
import { TRANSCRIPT_MESSAGE_AREA, type TranscriptMessageContribution } from '@/lib/transcript-message'

import { assistantMessage, createdAt, stubThreadEnvironment, ThreadRuntime, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
const disposers: Array<() => void> = []
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

function slash(id: string, command: string, output: string): ThreadMessage {
  return {
    id,
    role: 'system',
    content: [{ type: 'text', text: `slash:${command}\n${output}` }],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function contribute(id: string, data: TranscriptMessageContribution) {
  const dispose = registry.register({ id, area: TRANSCRIPT_MESSAGE_AREA, data })
  disposers.push(dispose)

  return dispose
}

function Harness({ messages, session = 'session-a' }: { messages: ThreadMessage[]; session?: string }) {
  return (
    <SessionViewProvider value={{ ...PRIMARY_SESSION_VIEW, $runtimeId: atom(session), kind: 'tile' }}>
      <ThreadRuntime messages={messages}>
        <Thread sessionId={session} />
      </ThreadRuntime>
    </SessionViewProvider>
  )
}

describe('transcript-message contributions in the real thread', () => {
  it('keeps native slash text and assistant actions without a registration or match', () => {
    const { container } = render(
      <Harness messages={[userMessage(), slash('slash-1', '/example', 'ordinary output'), assistantMessage()]} />
    )

    expect(container.querySelectorAll('[data-role="system"]')).toHaveLength(1)
    expect(screen.getByText('ordinary output')).toBeTruthy()
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(1)

    act(() => {
      contribute('declines', { match: () => false, render: () => <button>never</button> })
    })
    expect(screen.getByText('ordinary output')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'never' })).toBeNull()
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(1)
  })

  it('scopes matching and actionable inline render to the actual session and row, never a fabricated turn', () => {
    const action = vi.fn()
    const seen: unknown[] = []
    contribute('inline', {
      match: props => {
        seen.push(props)

        return props.kind === 'slash-result' && props.command === '/example'
      },
      render: props => <button onClick={() => action(props)}>Review in chat</button>
    })
    const messages = [userMessage(), slash('slash-1', '/example', 'card output'), assistantMessage()]

    const { container, rerender } = render(
      <StrictMode>
        <Harness messages={messages} session="session-a" />
      </StrictMode>
    )

    expect(screen.getByRole('button', { name: 'Review in chat' }).closest('[data-role="system"]')).toBeTruthy()
    expect(container.querySelectorAll('[data-role="system"]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(1)
    expect(screen.queryByText('card output')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Review in chat' }))
    expect(action).toHaveBeenCalledWith({
      kind: 'slash-result',
      sessionId: 'session-a',
      messageId: 'slash-1',
      isLast: false,
      command: '/example',
      output: 'card output'
    })
    expect(seen).toContainEqual(expect.objectContaining({ sessionId: 'session-a', messageId: 'slash-1' }))
    rerender(
      <StrictMode>
        <Harness messages={messages} session="session-b" />
      </StrictMode>
    )
    fireEvent.click(screen.getByRole('button', { name: 'Review in chat' }))
    expect(action).toHaveBeenLastCalledWith(expect.objectContaining({ sessionId: 'session-b', messageId: 'slash-1' }))
  })

  it('mounts an assistant-footer only when completed and preserves the ordinary assistant row', () => {
    const calls = vi.fn()
    contribute('footer', {
      match: props => props.kind === 'assistant-footer' && props.isLast,
      render: props => <button onClick={() => calls(props)}>Inspect result</button>
    })
    const reply = assistantMessage()
    const { container, unmount } = render(<Harness messages={[userMessage(), reply]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Inspect result' }))
    expect(calls).toHaveBeenCalledWith({
      kind: 'assistant-footer',
      sessionId: 'session-a',
      messageId: reply.id,
      isLast: true
    })
    expect(screen.getByText('done')).toBeTruthy()
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(1)
    unmount()
    render(<Harness messages={[userMessage(), { ...reply, status: { type: 'running' } } as ThreadMessage]} />)
    expect(screen.queryByRole('button', { name: 'Inspect result' })).toBeNull()
  })

  it('leaves a matched-looking slash row as text when every matcher fails', () => {
    const failed = vi.spyOn(console, 'error').mockImplementation(() => {})

    try {
      contribute('bad-predicate', {
        match: () => {
          throw new Error('bad match')
        },
        render: () => <button>never</button>
      })
      const { container } = render(<Harness messages={[slash('slash-1', '/example', 'plain result')]} />)
      expect(screen.getByText('plain result')).toBeTruthy()
      expect(screen.queryByRole('button', { name: 'never' })).toBeNull()
      expect(container.querySelectorAll('[data-role="system"]')).toHaveLength(1)
    } finally {
      failed.mockRestore()
    }
  })

  it('falls back to the original slash text on renderer failure, and recovers after replacement', () => {
    const failed = vi.spyOn(console, 'error').mockImplementation(() => {})

    try {
      contribute('broken-match', {
        match: () => {
          throw new Error('bad match')
        },
        render: () => null
      })

      const dispose = contribute('replacement', {
        match: () => true,
        render: () => {
          throw new Error('bad render')
        }
      })

      const messages = [slash('slash-1', '/example', 'plain result')]
      const { container } = render(<Harness messages={messages} />)
      expect(screen.getByText('plain result')).toBeTruthy()
      expect(container.querySelectorAll('[data-role="system"]')).toHaveLength(1)
      act(() => {
        dispose()
        contribute('replacement', { match: () => true, render: () => <button>Recovered action</button> })
      })
      expect(screen.getByRole('button', { name: 'Recovered action' })).toBeTruthy()
      expect(screen.queryByText('plain result')).toBeNull()
    } finally {
      failed.mockRestore()
    }
  })
})
