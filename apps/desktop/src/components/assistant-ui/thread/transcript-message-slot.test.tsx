import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { type ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import {
  TRANSCRIPT_MESSAGE_AREA,
  type TranscriptMessageContribution,
  type TranscriptMessageProps
} from '@/lib/transcript-message'
import { $activeSessionId } from '@/store/session'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const metadata = { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
const disposers: Array<() => void> = []

stubThreadEnvironment()

function user(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistant(
  id: string,
  text: string,
  status: 'complete' | 'incomplete' | 'running' = 'complete',
  interim = false
): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: [{ type: 'text', text }],
    status:
      status === 'complete'
        ? { type: 'complete', reason: 'stop' }
        : status === 'running'
          ? { type: 'running' }
          : { type: 'incomplete', reason: 'error', error: 'failed' },
    createdAt,
    metadata: {
      ...metadata,
      custom: interim ? { interim: true } : {}
    }
  } as ThreadMessage
}

function Harness({ messages, sessionId = null }: { messages: ThreadMessage[]; sessionId?: string | null }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: messages.at(-1)?.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionId={sessionId} />
    </AssistantRuntimeProvider>
  )
}

function addFooter(
  id: string,
  order: number,
  renderFooter: (props: TranscriptMessageProps) => ReactNode,
  match: (props: TranscriptMessageProps) => boolean = props => props.kind === 'assistant-footer'
) {
  const dispose = registry.register({
    area: TRANSCRIPT_MESSAGE_AREA,
    data: { match },
    id,
    order,
    render: renderFooter,
    source: `plugin:${id}`
  } satisfies TranscriptMessageContribution)

  disposers.push(dispose)

  return dispose
}

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $activeSessionId.set(null)
  vi.restoreAllMocks()
})

describe('transcript-message assistant-footer contributions', () => {
  it('renders ordered contributions inside completed rows with containing-thread identity and latest-row state', async () => {
    $activeSessionId.set('focused-global-session')
    const seen: Array<{ contribution: string; props: TranscriptMessageProps }> = []

    addFooter('later', 20, props => {
      seen.push({ contribution: 'later', props })

      return <span data-footer="later">later</span>
    })
    addFooter('earlier', 10, props => {
      seen.push({ contribution: 'earlier', props })

      return <span data-footer="earlier">earlier</span>
    })

    const { container } = render(
      <Harness
        messages={[
          user('user-1', 'first question'),
          assistant('assistant-1', 'first answer'),
          user('user-2', 'second question'),
          assistant('assistant-2', 'second answer')
        ]}
        sessionId="containing-thread-session"
      />
    )

    await screen.findByText('second answer')
    const rows = Array.from(container.querySelectorAll('[data-slot="aui_assistant-message-root"]'))
    expect(rows).toHaveLength(2)

    for (const row of rows) {
      expect(Array.from(row.querySelectorAll('[data-footer]')).map(node => node.getAttribute('data-footer'))).toEqual([
        'earlier',
        'later'
      ])
    }

    expect(seen.filter(entry => entry.contribution === 'earlier').map(entry => entry.props)).toEqual([
      {
        isLast: false,
        kind: 'assistant-footer',
        messageId: 'assistant-1',
        sessionId: 'containing-thread-session'
      },
      {
        isLast: true,
        kind: 'assistant-footer',
        messageId: 'assistant-2',
        sessionId: 'containing-thread-session'
      }
    ])
  })

  it('does not render for running, interim, incomplete, or collapsed inter-agent rows', async () => {
    addFooter('probe', 0, props => <span data-testid="footer-probe">{props.messageId}</span>)

    const cases: Array<{ messages: ThreadMessage[]; text: string }> = [
      {
        messages: [user('user-running', 'question'), assistant('assistant-running', 'working', 'running')],
        text: 'working'
      },
      {
        messages: [user('user-interim', 'question'), assistant('assistant-interim', 'aside', 'complete', true)],
        text: 'aside'
      },
      {
        messages: [user('user-error', 'question'), assistant('assistant-error', 'partial', 'incomplete')],
        text: 'partial'
      },
      {
        messages: [
          user('user-agent', 'Message from 🤖 Hermes (@hermes): please check the build'),
          assistant('assistant-agent', 'build is green')
        ],
        text: 'build is green'
      }
    ]

    for (const { messages, text } of cases) {
      const view = render(<Harness messages={messages} sessionId="runtime-session" />)
      await screen.findByText(text)
      expect(screen.queryByTestId('footer-probe')).toBeNull()
      view.unmount()
    }
  })

  it('isolates renderer errors per contribution and row, disposes registrations, and adds no chrome for null output', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)

    addFooter('broken-on-first-row', 0, props => {
      if (props.messageId === 'assistant-1') {
        throw new Error('plugin exploded')
      }

      return <span data-testid="recovered-on-next-row">healthy next row</span>
    })
    const disposeTemporary = addFooter('temporary', 10, () => <span data-testid="temporary-footer">temporary</span>)
    addFooter(
      'wrong-placement',
      20,
      () => <span data-testid="wrong-placement">wrong placement</span>,
      props => props.kind === 'slash-result'
    )
    addFooter(
      'broken-match',
      30,
      () => <span data-testid="broken-match">broken match</span>,
      () => {
        throw new Error('matcher exploded')
      }
    )

    const messages = [
      user('user-1', 'first question'),
      assistant('assistant-1', 'first answer'),
      user('user-2', 'second question'),
      assistant('assistant-2', 'second answer')
    ]

    const { container } = render(<Harness messages={messages} sessionId="runtime-session" />)

    await screen.findByText('second answer')
    expect(screen.getByText('first answer')).toBeTruthy()
    expect(screen.getByTestId('recovered-on-next-row')).toBeTruthy()
    expect(screen.getAllByTestId('temporary-footer')).toHaveLength(2)
    expect(screen.queryByTestId('wrong-placement')).toBeNull()
    expect(screen.queryByTestId('broken-match')).toBeNull()
    expect(container.textContent).not.toContain('plugin exploded')

    const secondRow = screen.getByText('second answer').closest('[data-slot="aui_assistant-message-root"]')
    const childCountBeforeNullContribution = secondRow?.children.length

    addFooter('null-output', 20, () => null)
    expect(secondRow?.children.length).toBe(childCountBeforeNullContribution)

    disposeTemporary()
    await waitFor(() => expect(screen.queryByTestId('temporary-footer')).toBeNull())
    expect(screen.getByText('second answer')).toBeTruthy()
  })
})
