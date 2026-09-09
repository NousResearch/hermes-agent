import type { ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { setCollapseSettledDiffs } from '@/store/diff-disclosure'
import { $toolDisclosureStates } from '@/store/tool-view'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

function transcript(running: boolean) {
  const message: ThreadMessage = {
    ...assistantMessage(),
    content: [
      {
        type: 'tool-call',
        args: { path: 'example.txt' },
        argsText: '{"path":"example.txt"}',
        result: { diff: '--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-old\n+new', success: true },
        toolCallId: 'diff-call',
        toolName: 'patch'
      }
    ],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' }
  }

  return (
    <ThreadRuntime messages={[message]}>
      <Thread />
    </ThreadRuntime>
  )
}

const header = (container: HTMLElement) =>
  container.querySelector<HTMLButtonElement>('[data-slot="tool-block"] button[aria-expanded]')!

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  setCollapseSettledDiffs(true)
})

it('keeps a live diff open, collapses it when settled, and lets the reader reopen it', async () => {
  const view = render(transcript(true))

  expect(header(view.container).getAttribute('aria-expanded')).toBe('true')
  view.rerender(transcript(false))
  await waitFor(() => expect(header(view.container).getAttribute('aria-expanded')).toBe('false'))
  fireEvent.click(header(view.container))
  expect(header(view.container).getAttribute('aria-expanded')).toBe('true')

  view.unmount()
  const restored = render(transcript(false))
  expect(header(restored.container).getAttribute('aria-expanded')).toBe('true')
})

it('honors opt-out and manual disclosure choices across preference and running-state changes', () => {
  setCollapseSettledDiffs(false)
  const view = render(transcript(false))

  expect(header(view.container).getAttribute('aria-expanded')).toBe('true')
  act(() => setCollapseSettledDiffs(true))
  expect(header(view.container).getAttribute('aria-expanded')).toBe('false')
  fireEvent.click(header(view.container))
  fireEvent.click(header(view.container))
  act(() => setCollapseSettledDiffs(false))
  view.rerender(transcript(true))
  expect(header(view.container).getAttribute('aria-expanded')).toBe('false')
  expect(localStorage.getItem('hermes.desktop.diffs.collapseWhenSettled')).toBe('false')
})
