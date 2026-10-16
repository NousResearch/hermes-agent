import { cleanup, render, waitFor } from '@testing-library/react'
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

const completionText = (id: string) =>
  [
    `[ASYNC DELEGATION COMPLETE — ${id}]`,
    'A background subagent you dispatched earlier has finished.',
    'Status: completed   API calls: 1   Duration: 1s',
    '--- RESULT ---',
    'Worker raw result.'
  ].join('\n')

const completion = {
  role: 'user',
  content: completionText('deleg_review'),
  display_kind: 'async_delegation_complete',
  display_metadata: { display_text: 'Background agent finished' }
} as SessionMessage

const processCompletion = {
  role: 'user',
  content: '[IMPORTANT: Background process proc_review completed normally (exit code 0).\nOutput:\nok]',
  display_kind: 'process_complete',
  display_metadata: { display_text: 'Background Process Finished: review' }
} as SessionMessage

function renderRows(rows: SessionMessage[]) {
  const messages = toChatMessages(rows.map((row, index) => ({ timestamp: index + 1, ...row }))).map(toRuntimeMessage)

  return render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )
}

function finalReply(container: HTMLElement) {
  return [...container.querySelectorAll('[data-role="assistant"]')].find(row =>
    row.textContent?.includes('The review found one blocker.')
  )
}

it.each([
  ['delegation completion', completion],
  ['process completion', processCompletion]
])('keeps the reply to a %s visible after an inter-agent delivery', async (_, notification) => {
  const { container } = renderRows([
    { role: 'user', content: 'Message from bob: please review the change' },
    notification,
    { role: 'assistant', content: 'The review found one blocker.' }
  ] as SessionMessage[])

  await waitFor(() => expect(finalReply(container)).toBeTruthy())
  expect(finalReply(container)!.querySelector('details')).toBeNull()
  expect(container.textContent).not.toContain('Replied to bob')
})

it('still collapses a direct reply to an inter-agent delivery', async () => {
  const { container } = renderRows([
    { role: 'user', content: 'Message from bob: please review the change' },
    { role: 'assistant', content: 'Acknowledged, bob.' }
  ] as SessionMessage[])

  await waitFor(() => expect(container.textContent).toContain('Replied to bob'))
})

it('keeps every assistant reply around consecutive completions in the main transcript', async () => {
  const { container } = renderRows([
    { role: 'user', content: 'Review it.' },
    { role: 'assistant', content: 'Started two reviewers.' },
    completion,
    { ...completion, content: completionText('deleg_second') },
    { role: 'assistant', content: 'The review found one blocker.' }
  ] as SessionMessage[])

  await waitFor(() => expect(finalReply(container)).toBeTruthy())
  expect(container.querySelectorAll('[data-slot="aui_background-result"]')).toHaveLength(2)
  expect([...container.querySelectorAll('[data-role="assistant"]')].map(row => row.textContent)).toEqual([
    'Started two reviewers.',
    'The review found one blocker.'
  ])

  for (const row of container.querySelectorAll('[data-role="assistant"]')) {
    expect(row.closest('[data-slot="aui_background-result"]')).toBeNull()
    expect(row.querySelector('details')).toBeNull()
  }

  expect(container.textContent).not.toContain('Worker raw result.')
})
