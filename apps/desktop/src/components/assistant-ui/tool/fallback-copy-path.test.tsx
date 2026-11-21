// @vitest-environment jsdom
// Regression for #89156: a diff's content-copy action is not a path-copy action.
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $hideCodeDiffs, $toolDisclosureStates } from '@/store/tool-view'

import { fileEditPath, toolPreviewOutcome } from './fallback-model'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'copy-path-message', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')
const diff = '--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-before\n+after'
const writeClipboard = vi.fn(async (_text: string) => {})
let originalDesktop: typeof window.hermesDesktop

function show(toolName: string, args: Record<string, unknown> | string, result: Record<string, unknown> | string) {
  render(
    <ToolFallback
      {...({ args, result, toolCallId: 'copy-path-call', toolName } as ComponentProps<typeof ToolFallback>)}
    />
  )
}

beforeEach(() => {
  originalDesktop = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { writeClipboard } })
  writeClipboard.mockReset().mockResolvedValue(undefined)
})

afterEach(() => {
  cleanup()
  clearDismissedToolRows()
  $toolDisclosureStates.set({})
  $hideCodeDiffs.set(false)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
})

it('copies serialized arguments through the browser fallback and keeps dismissal independent', async () => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
  const originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard')
  const writeText = vi.fn(async (_text: string) => {})
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })

  try {
    const path = '/remote/project/full path.txt'
    show('patch', JSON.stringify({ path }), { success: true, diff })
    const toggle = screen.getByRole('button', { expanded: true })
    fireEvent.click(toggle)
    fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(path))
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(writeClipboard).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }))
    expect(screen.queryByRole('button', { name: 'Copied' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Dismiss' })).toBeNull()
  } finally {
    if (originalClipboard) {Object.defineProperty(navigator, 'clipboard', originalClipboard)}
    else {Reflect.deleteProperty(navigator, 'clipboard')}
  }
})

it('copies the supplied edit path without changing disclosure or replacing diff copy', async () => {
  for (const [toolName, path] of [
    ['patch', '/srv/project/文档 with spaces/file.txt'],
    ['write_file', 'C:\\work\\文档 with spaces\\file.txt'],
    ['edit_file', '../project/file.txt']
  ]) {
    show(toolName, { path }, { success: true, diff })
    const toggle = screen.getByRole('button', { expanded: true })
    fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith(path))
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'Copy file' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith(diff))
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByRole('button', { name: 'Copied' })).toBeTruthy()
    cleanup()
    $toolDisclosureStates.set({})
  }
})

it('keeps path copy available with hidden diffs and failures, but never copies placeholder or error text', async () => {
  $hideCodeDiffs.set(true)
  show('patch', {}, { success: true, resolved_path: '~/project/complete.txt', diff })
  fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
  await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith('~/project/complete.txt'))
  cleanup()

  show('patch', { path: '/remote/project/file.txt' }, { error: 'Permission denied' })
  writeClipboard.mockRejectedValueOnce(new Error('clipboard unavailable'))
  fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
  expect(await screen.findByRole('button', { name: 'Copy failed' })).toBeTruthy()
  expect(writeClipboard).toHaveBeenLastCalledWith('/remote/project/file.txt')
  cleanup()

  for (const toolName of ['patch', 'terminal']) {
    show(toolName, {}, { error: 'Permission denied', diff })
    expect(screen.queryByRole('button', { name: 'Copy path' })).toBeNull()
    cleanup()
  }
})

it('never offers a diff-content path as Copy path while preserving preview inference and diff copy', async () => {
  const commentDiff = diff.replace('+after', '+// See docs/index.html for details')

  for (const field of ['diff', 'inline_diff']) {
    const args = { path: '', file: null, filepath: 42 }
    const result = { success: true, path: [], file: false, filepath: ' ', resolved_path: {}, [field]: commentDiff }

    // Preview/display inference is an existing, separate contract, not a clipboard source.
    expect(fileEditPath(args, result)).toBe('docs/index.html')
    const preview = toolPreviewOutcome({ args, result, toolCallId: 'preview-call', toolName: 'patch' })
    expect(preview.previewTarget).toBe('docs/index.html')
    show('patch', args, result)
    expect(screen.queryByRole('button', { name: 'Copy path' })).toBeNull()
    expect(writeClipboard).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Copy file' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith(commentDiff))
    cleanup()
    writeClipboard.mockClear()
  }
})

it('copies only explicit string path fields with argument precedence and result fallback', async () => {
  const argumentPath = '../source/文档 with spaces.txt'
  const resultPath = 'C:\\work\\resolved file.txt'
  const commentDiff = diff.replace('+after', '+// See docs/index.html for details')

  for (const field of ['path', 'file', 'filepath']) {
    show('patch', { [field]: argumentPath }, { path: resultPath, diff: commentDiff })
    fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith(argumentPath))
    cleanup()
  }

  for (const field of ['path', 'file', 'filepath', 'resolved_path']) {
    show('patch', { path: null, file: ' ', filepath: 42 }, JSON.stringify({ [field]: resultPath, diff: commentDiff }))
    fireEvent.click(screen.getByRole('button', { name: 'Copy path' }))
    await waitFor(() => expect(writeClipboard).toHaveBeenLastCalledWith(resultPath))
    cleanup()
  }

  show('terminal', { path: argumentPath }, { resolved_path: resultPath, diff: commentDiff })
  expect(screen.queryByRole('button', { name: 'Copy path' })).toBeNull()
})
