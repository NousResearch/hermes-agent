import type { ThreadMessage } from '@assistant-ui/react'
import { fireEvent, render, waitFor, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { createdAt, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime, userMessage } from '../test-utils'

import { Thread } from '.'

// The transcript is the app's whole reading surface: every answer — streamed or
// settled — has to come back as rich text, never as the markdown source. These
// lock the two states down, because the "cheap streaming path" is exactly where
// literal `**`, `|---|` and backticks leak back onto the screen.

stubThreadEnvironment()
stubThreadViewportSize()

const SETTLED_ANSWER = [
  'Mam **pamięć trwałą** w `hermes-home/memories/`, więc odpowiem wprost:',
  '',
  '| Warstwa | Gdzie siedzi | Stan u Ciebie |',
  '|---|---|---|',
  '| Pamięć trwała | `hermes-home/memories/` | **pusta** — zero faktów |',
  '',
  '```ts',
  'const answer = 42',
  '```'
].join('\n')

// Mid-stream: the table is already complete, the list under it is not.
const STREAMING_ANSWER = [
  'Mam **pamięć trwałą** w `hermes-home/memories/`:',
  '',
  '| Warstwa | Stan u Ciebie |',
  '|---|---|',
  '| Pamięć trwała | **pusta** |',
  '',
  '- Skille (procedury) siedzą w `hermes-home/skills/`',
  '- Historia sesji w `state.db`'
].join('\n')

const REASONING = 'The user asked about **memory**; let me check `hermes-home/memories/`.'

function assistant(text: string, running: boolean, extra: Partial<ThreadMessage> = {}): ThreadMessage {
  return {
    id: `assistant-${running ? 'running' : 'done'}`,
    role: 'assistant',
    content: [{ type: 'text', text }],
    createdAt,
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} },
    ...extra
  } as ThreadMessage
}

function reasoningMessage(running: boolean): ThreadMessage {
  return {
    id: `assistant-reasoning-${running ? 'running' : 'done'}`,
    role: 'assistant',
    content: [{ type: 'reasoning', text: REASONING }],
    createdAt,
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

function renderTranscript(messages: ThreadMessage[]) {
  return render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )
}

function expectRichAnswer(container: HTMLElement) {
  const text = container.textContent ?? ''

  expect(text).not.toContain('**pamięć trwałą**')
  expect(text).not.toContain('**pusta**')
  expect(text).not.toContain('|---|')
  expect(text).not.toContain('```ts')
}

describe('assistant answers in the transcript', () => {
  it('renders a settled answer as rich markdown (bold, table, code)', async () => {
    const { container } = renderTranscript([userMessage(), assistant(SETTLED_ANSWER, false)])

    await waitFor(() => {
      expect(container.querySelector('table')).toBeTruthy()
    })

    expectRichAnswer(container)
    expect(container.querySelector('th')?.textContent).toContain('Warstwa')
    expect(container.querySelectorAll('tbody tr').length).toBe(1)
    expect(container.querySelector('strong, [data-streamdown="strong"], .font-semibold')).toBeTruthy()
    expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy()
    expect(container.textContent).toContain('const answer = 42')
    expect(container.textContent).toContain('Mam pamięć trwałą')
  })

  it('renders a streaming answer as rich markdown too', async () => {
    const { container } = renderTranscript([userMessage(), assistant(STREAMING_ANSWER, true)])

    await waitFor(() => {
      expect(container.querySelector('table')).toBeTruthy()
    })

    expectRichAnswer(container)
    expect(container.querySelector('th')?.textContent).toContain('Warstwa')
    expect(container.querySelector('li')?.textContent).toContain('Skille (procedury)')
  })

  it('renders a streaming answer without collapsing it into one raw text block', async () => {
    const { container } = renderTranscript([userMessage(), assistant(SETTLED_ANSWER, true)])

    await waitFor(() => {
      expect(container.querySelector('table')).toBeTruthy()
    })

    expectRichAnswer(container)
  })
})

describe('reasoning drafts in the transcript', () => {
  it('stays folded to its header — no second full-size answer', async () => {
    const { container } = renderTranscript([userMessage(), assistant('Odpowiedź gotowa.', false), reasoningMessage(false)])

    const toggle = within(container).getByRole('button', { name: /Przemyślał|Myśli/i })

    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(container.querySelector('[data-slot="aui_thinking-body"]')).toBeNull()
    expect(container.textContent).not.toContain('let me check')
  })

  it('opens on demand and then renders the draft as markdown, not raw markers', async () => {
    const { container } = renderTranscript([userMessage(), reasoningMessage(false)])

    fireEvent.click(within(container).getByRole('button', { name: /Przemyślał|Myśli/i }))

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_reasoning-text"]')).toBeTruthy()
    })

    const draft = container.querySelector('[data-slot="aui_reasoning-text"]') as HTMLElement

    expect(draft.textContent).toContain('let me check')
    expect(draft.textContent).not.toContain('**memory**')
    expect(draft.textContent).not.toContain('`hermes-home/memories/`')
    expect(draft.querySelector('strong, [data-streamdown="strong"], .font-semibold, code')).toBeTruthy()
    // A draft is a note, not the reply: it never carries the reading column's
    // full-size typography.
    expect(draft.className).toContain('text-[0.6875rem]')
  })
})
