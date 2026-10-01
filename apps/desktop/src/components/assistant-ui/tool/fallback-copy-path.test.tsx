// @vitest-environment jsdom
// Regression for #89156: a diff's content-copy action is not a path-copy action.
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $hideCodeDiffs, $toolDisclosureStates } from '@/store/tool-view'

vi.mock('@assistant-ui/react', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuiState: (select: (state: unknown) => unknown) =>
    select({ message: { id: 'copy-path-message', status: { type: 'complete' } }, thread: { isRunning: false } })
}))

const { ToolFallback } = await import('./fallback')
const diff = '--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-before\n+after'
const writeClipboard = vi.fn(async (_text: string) => {})
let originalDesktop: typeof window.hermesDesktop

function show(toolName: string, args: Record<string, unknown>, result: Record<string, unknown>) {
  render(<ToolFallback {...({ args, result, toolCallId: 'copy-path-call', toolName } as ComponentProps<typeof ToolFallback>)} />)
}

beforeEach(() => {
  originalDesktop = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { writeClipboard } })
  writeClipboard.mockReset().mockResolvedValue(undefined)
})

afterEach(() => {
  cleanup()
  $toolDisclosureStates.set({})
  $hideCodeDiffs.set(false)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
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
