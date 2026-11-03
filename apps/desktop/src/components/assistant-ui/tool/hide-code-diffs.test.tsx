import type { ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $hideCodeDiffs, $toolDisclosureStates, setHideCodeDiffs, setToolViewMode } from '@/store/tool-view'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

stubThreadEnvironment()
stubThreadViewportSize()

const diff = '--- a/demo.ts\n+++ b/demo.ts\n@@ -1 +1,2 @@\n-beforeEdit\n+afterEdit\n+addedLine'

function editMessage(toolName: string, failed = false): ThreadMessage {
  return {
    ...assistantMessage(),
    content: [
      {
        type: 'tool-call',
        toolCallId: `edit-${toolName}`,
        toolName,
        args: { path: '/repo/demo.ts', content: 'afterEdit\naddedLine' },
        argsText: '{}',
        result: failed
          ? { success: false, error: 'File is read-only' }
          : { success: true, path: '/repo/demo.ts', inline_diff: diff }
      }
    ]
  } as ThreadMessage
}

beforeEach(() => {
  $toolDisclosureStates.set({})
  setHideCodeDiffs(false)
  setToolViewMode('product')
})

afterEach(() => {
  cleanup()
  setHideCodeDiffs(false)
  setToolViewMode('product')
  $toolDisclosureStates.set({})
})

it('keeps edit counts without code in either display mode and restores disclosure when disabled', async () => {
  for (const toolName of ['patch', 'edit_file', 'write_file']) {
    const { container } = render(
      <ThreadRuntime messages={[editMessage(toolName)]}>
        <Thread />
      </ThreadRuntime>
    )

    await waitFor(() => expect(container.querySelector('[data-tool-row][data-file-edit]')).not.toBeNull())
    const row = container.querySelector('[data-tool-row]')!
    // An explicitly open historical row must not override the preference.
    const toggle = row.querySelector('button[aria-expanded]')!
    fireEvent.click(toggle)
    fireEvent.click(toggle)

    for (const mode of ['product', 'technical'] as const) {
      act(() => {
        setToolViewMode(mode)
        setHideCodeDiffs(true)
      })
      await waitFor(() => expect(row.hasAttribute('data-tool-open')).toBe(false))
      expect(row.textContent).toContain('+2')
      expect(row.textContent).toContain('−1')
      expect(row.textContent).not.toContain('afterEdit')
      expect(row.textContent).not.toContain('beforeEdit')
      expect(row.querySelector('pre, code, button[aria-expanded]')).toBeNull()
      expect($hideCodeDiffs.get()).toBe(true)
      expect(localStorage.getItem('hermes.desktop.toolView.hideCodeDiffs')).toBe('true')
    }

    act(() => setHideCodeDiffs(false))
    await waitFor(() => expect(row.hasAttribute('data-tool-open')).toBe(true))
    cleanup()
  }
})

it('still discloses failed edits when code diffs are hidden', async () => {
  setHideCodeDiffs(true)
  setToolViewMode('technical')

  const { container } = render(
    <ThreadRuntime messages={[editMessage('patch', true)]}>
      <Thread />
    </ThreadRuntime>
  )

  const toggle = container.querySelector('[data-tool-row] button[aria-expanded]')!
  expect(toggle).not.toBeNull()
  fireEvent.click(toggle)
  expect(await screen.findByText('File is read-only')).toBeTruthy()
  expect(container.textContent).not.toContain('afterEdit')
})
