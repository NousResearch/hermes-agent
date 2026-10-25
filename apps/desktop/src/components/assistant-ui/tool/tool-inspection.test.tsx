import type { ThreadMessage } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $toolDisclosureStates, $toolViewMode } from '@/store/tool-view'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

import { buildToolView, toolCopyPayload } from './fallback-model'

stubThreadEnvironment()
stubThreadViewportSize()

const writeText = vi.fn(async (_text: string) => {})
const originalClipboard = navigator.clipboard

function message(result: unknown, args: unknown = { command: 'printf ok' }): ThreadMessage {
  return {
    ...assistantMessage(),
    content: [{ type: 'tool-call', toolCallId: 'inspect-1', toolName: 'terminal', args, argsText: '{}', result }]
  } as ThreadMessage
}

function Harness({ value, sessionId = 'session-a' }: { value: ThreadMessage; sessionId?: string }) {
  return (
    <ThreadRuntime messages={[value]}>
      <Thread sessionId={sessionId} />
    </ThreadRuntime>
  )
}

beforeEach(() => {
  writeText.mockClear()
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } })
  $toolViewMode.set('product')
  $toolDisclosureStates.set({})
})

afterEach(() => {
  cleanup()
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: originalClipboard })
})

describe('tool inspection in the normal transcript', () => {
  it('searches and copies past the inline limit independently of long arguments, with bounded painting', async () => {
    const stdout = 'x'.repeat(25_000) + '\nneedle after the inline limit\n' + 'y'.repeat(100) + '\nneedle again\n'
    render(<Harness value={message({ stdout, stderr: 'diagnostic' }, { command: 'c'.repeat(30_000) })} />)
    const trigger = await screen.findByRole('button', { name: 'Open tool details' })
    fireEvent.click(trigger)
    const dialog = await screen.findByRole('dialog')
    const ui = within(dialog)
    fireEvent.click(ui.getByRole('button', { name: 'Arguments' }))
    expect(ui.getByRole('region', { name: 'Arguments' }).textContent?.length).toBeLessThanOrEqual(16_002)
    fireEvent.click(ui.getByRole('button', { name: 'stdout' }))
    fireEvent.change(ui.getByRole('textbox', { name: 'Find in section (case-sensitive)' }), {
      target: { value: 'needle' }
    })
    expect(ui.getByRole('region', { name: 'stdout' }).textContent).toContain('needle after the inline limit')
    expect(ui.getByText('1/2')).toBeTruthy()
    fireEvent.click(ui.getByRole('button', { name: 'Next match' }))
    expect(ui.getByText('2/2')).toBeTruthy()
    expect(ui.getByRole('region', { name: 'stdout' }).textContent).toContain('needle again')
    fireEvent.click(ui.getByRole('button', { name: 'Next match' }))
    expect(ui.getByText('1/2')).toBeTruthy()
    fireEvent.click(ui.getByRole('button', { name: 'Previous match' }))
    expect(ui.getByText('2/2')).toBeTruthy()
    expect(ui.getByRole('region', { name: 'stdout' }).textContent?.length).toBeLessThanOrEqual(16_002)
    fireEvent.click(ui.getByRole('button', { name: 'Wrap lines' }))
    expect(ui.getByRole('button', { name: 'Wrap lines' }).getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(ui.getByRole('button', { name: 'Copy section' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(stdout))
    fireEvent.keyDown(dialog, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(window.document.activeElement).toBe(trigger)
  })

  it('keeps the opened snapshot after the row disappears but closes it on a session switch', async () => {
    const { rerender } = render(<Harness value={message({ stdout: 'original' })} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Open tool details' }))
    await screen.findByRole('dialog')
    rerender(<Harness value={assistantMessage()} />)
    expect(within(screen.getByRole('dialog')).getByRole('region', { name: 'Result' }).textContent).toContain('original')
    rerender(<Harness sessionId="session-b" value={message({ stdout: 'different session' })} />)
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('copies short and whitespace-bearing output as output, never as a command or path', () => {
    for (const text of ['ok', '', ' \n ']) {
      for (const part of [
        { type: 'tool-call' as const, toolName: 'terminal', args: { command: 'echo ok' }, result: { stdout: text } },
        { type: 'tool-call' as const, toolName: 'read_file', args: { path: '/tmp/result' }, result: { content: text } }
      ]) {
        expect(toolCopyPayload(part, buildToolView(part, '')).text).toBe(text)
      }
    }
  })
})
