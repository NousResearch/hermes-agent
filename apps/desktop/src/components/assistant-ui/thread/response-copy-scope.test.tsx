// Bug #118864: the footer's text accessor (getMessageText) joined EVERY
// message in the response group, so the tail reply's Copy/Read-aloud included
// sealed interim narration and mid-turn commentary. The default Copy/Read
// aloud must read only the current (tail) reply, with a separate
// full-response copy that keeps the whole-tool-turn semantic the grouping
// change (3bdd4cc5fd) delivered for background continuations.
import { cleanup, fireEvent, render, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import { $touchPointer } from '@/lib/touch-interaction'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

beforeEach(stubThreadEnvironment)
afterEach(() => {
  cleanup()
  $touchPointer.set(false)
  vi.unstubAllGlobals()
})

const stored: SessionMessage[] = [
  { role: 'user', content: 'Verify it.', timestamp: 1 },
  { role: 'assistant', content: 'Checking the deployment.', timestamp: 2 },
  {
    role: 'user',
    content: '[IMPORTANT: Background process proc_example completed normally (exit code 0).\nOutput:\nVerified.]',
    timestamp: 3
  },
  { role: 'assistant', content: 'The deployment is verified.', timestamp: 4 }
]

const messages = () => toChatMessages(stored).map(toRuntimeMessage)

function renderThread() {
  const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
  vi.stubGlobal('navigator', { ...navigator, clipboard })

  const rendered = render(
    <ThreadRuntime messages={messages()}>
      <Thread />
    </ThreadRuntime>
  )

  return { clipboard, ...rendered }
}

it.each(['desktop', 'touch'])(
  'default %s Copy reads only the tail reply, not the whole response group',
  async surface => {
    $touchPointer.set(surface === 'touch')
    const { clipboard, container } = renderThread()

    await waitFor(() => expect(container.textContent).toContain('The deployment is verified.'))
    const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
    const surfaceActions = actions.querySelector(`.aui-message-actions-${surface}`) as HTMLElement
    fireEvent.click(within(surfaceActions).getByRole('button', { name: /^copy$/i }))

    await waitFor(() => expect(clipboard.writeText).toHaveBeenLastCalledWith('The deployment is verified.'))
  }
)

it('Copy full response reads every assistant text segment in the group', async () => {
  const { clipboard, container } = renderThread()

  await waitFor(() => expect(container.textContent).toContain('The deployment is verified.'))
  const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
  fireEvent.click(within(actions).getByRole('button', { name: /copy full response/i }))

  await waitFor(() =>
    expect(clipboard.writeText).toHaveBeenLastCalledWith('Checking the deployment.\n\nThe deployment is verified.')
  )
})

it.each(['desktop', 'touch'])(
  'a solo reply copies its own text through %s actions and offers no full-response button',
  async surface => {
    $touchPointer.set(surface === 'touch')

    const storedSolo: SessionMessage[] = [
      { role: 'user', content: 'Hi', timestamp: 1 },
      { role: 'assistant', content: 'Hello there.', timestamp: 2 }
    ]

    const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
    vi.stubGlobal('navigator', { ...navigator, clipboard })

    const { container } = render(
      <ThreadRuntime messages={toChatMessages(storedSolo).map(toRuntimeMessage)}>
        <Thread />
      </ThreadRuntime>
    )

    await waitFor(() => expect(container.textContent).toContain('Hello there.'))
    const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
    const surfaceActions = actions.querySelector(`.aui-message-actions-${surface}`) as HTMLElement
    fireEvent.click(within(surfaceActions).getByRole('button', { name: /^copy$/i }))
    await waitFor(() => expect(clipboard.writeText).toHaveBeenLastCalledWith('Hello there.'))
    // The two scopes are identical on a solo reply, so only the default Copy exists.
    expect(within(actions).queryByRole('button', { name: /copy full response/i })).toBeNull()
  }
)

// The disappearing-final-response defect ended a completed, text-bearing turn as
// a reasoning-only row ("Thought for…") with NO footer — no copy control. Pin
// that the hydrated tool-heavy turn (assistant tool row + tool rows + the final
// prose row, each carrying reasoning) renders its answer AND keeps the footer.
it('a hydrated tool-heavy turn renders its final answer with the copy control', async () => {
  const storedToolTurn = [
    { role: 'user', content: 'Run the tools.', timestamp: 1 },
    {
      role: 'assistant',
      content: '',
      reasoning: 'Thinking Process: plan',
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'terminal', arguments: '{}' } }],
      timestamp: 2
    },
    { role: 'tool', content: 'ok', tool_call_id: 'call-1', timestamp: 3 },
    { role: 'assistant', content: 'The result is 71.', reasoning: 'Thinking Process: final', timestamp: 4 }
  ] as never

  const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
  vi.stubGlobal('navigator', { ...navigator, clipboard })

  const { container } = render(
    <ThreadRuntime messages={toChatMessages(storedToolTurn).map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(container.textContent).toContain('The result is 71.'))

  const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
  expect(actions).toBeTruthy()
  fireEvent.click(within(actions).getByRole('button', { name: /^copy$/i }))
  await waitFor(() => expect(clipboard.writeText).toHaveBeenLastCalledWith('The result is 71.'))
})
