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
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

beforeEach(stubThreadEnvironment)
afterEach(() => {
  cleanup()
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

it('default Copy reads only the tail reply, not the whole response group', async () => {
  const { clipboard, container } = renderThread()

  await waitFor(() => expect(container.textContent).toContain('The deployment is verified.'))
  const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
  fireEvent.click(within(actions).getByRole('button', { name: /^copy$/i }))

  await waitFor(() => expect(clipboard.writeText).toHaveBeenLastCalledWith('The deployment is verified.'))
})

it('Copy full response reads every assistant text segment in the group', async () => {
  const { clipboard, container } = renderThread()

  await waitFor(() => expect(container.textContent).toContain('The deployment is verified.'))
  const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
  fireEvent.click(within(actions).getByRole('button', { name: /copy full response/i }))

  await waitFor(() =>
    expect(clipboard.writeText).toHaveBeenLastCalledWith('Checking the deployment.\n\nThe deployment is verified.')
  )
})

it('a solo reply copies its own text and offers no separate full-response button', async () => {
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
  fireEvent.click(within(actions).getByRole('button', { name: /^copy$/i }))
  await waitFor(() => expect(clipboard.writeText).toHaveBeenLastCalledWith('Hello there.'))
  // The two scopes are identical on a solo reply, so only the default Copy exists.
  expect(within(actions).queryByRole('button', { name: /copy full response/i })).toBeNull()
})
