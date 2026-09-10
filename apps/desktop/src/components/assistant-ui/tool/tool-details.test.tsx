import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { buildToolDetailSections, findTextMatches, ToolDetailsDialog } from './tool-details'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }

afterEach(() => {
  cleanup()
  delete desktopWindow.hermesDesktop
})

describe('buildToolDetailSections', () => {
  it('keeps full arguments, results, streams and diff independently selectable', () => {
    const longResult = `before-${'x'.repeat(25_000)}-needle-after`

    const sections = buildToolDetailSections(
      {
        args: { command: 'printf ok', empty: '' },
        completedAt: 20,
        result: { diff: '+new', output: longResult, stderr: '' },
        timestamp: 10,
        toolCallId: 'call_1',
        toolName: 'terminal',
        type: 'tool-call'
      },
      ''
    )

    expect(sections.find(item => item.id === 'arguments')?.copyText).toContain('printf ok')
    expect(sections.find(item => item.id === 'command')?.copyText).toBe('printf ok')
    expect(sections.find(item => item.id === 'stdout')?.copyText).toBe(longResult)
    expect(sections.find(item => item.id === 'stderr')?.state).toBe('empty')
    expect(sections.find(item => item.id === 'diff')?.copyText).toBe('+new')
    expect(sections.find(item => item.id === 'result')?.copyText).toContain('needle-after')
    expect(sections.find(item => item.id === 'metadata')?.copyText).toContain('call_1')
  })

  it('distinguishes missing values from values supplied as empty strings', () => {
    const missing = buildToolDetailSections({ toolName: 'terminal', type: 'tool-call' }, '')
    const empty = buildToolDetailSections({ args: '', result: '', toolName: 'terminal', type: 'tool-call' }, '')

    expect(missing.find(item => item.id === 'arguments')?.state).toBe('unavailable')
    expect(missing.find(item => item.id === 'result')?.state).toBe('unavailable')
    expect(empty.find(item => item.id === 'arguments')?.state).toBe('empty')
    expect(empty.find(item => item.id === 'result')?.state).toBe('empty')
  })
})

describe('findTextMatches', () => {
  it('searches the full selected text beyond the inline render limit', () => {
    const text = `${'x'.repeat(25_000)}Needle${'y'.repeat(25_000)}`

    expect(findTextMatches(text, 'needle')).toEqual([25_000])
  })
})

describe('ToolDetailsDialog', () => {
  it('selects and copies exact short output instead of the command', async () => {
    const writeClipboard = vi.fn().mockResolvedValue(undefined)
    desktopWindow.hermesDesktop = { writeClipboard } as unknown as Window['hermesDesktop']

    render(
      <ToolDetailsDialog
        inlineDiff=""
        onOpenChange={() => undefined}
        open
        part={{
          args: { command: 'echo ok' },
          result: { exit_code: 0, output: 'ok' },
          toolName: 'terminal',
          type: 'tool-call'
        }}
      />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    fireEvent.click(screen.getByRole('button', { name: 'Copy stdout' }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('ok'))
  })

  it('searches beyond the first render window, toggles wrapping and dismisses', () => {
    const onOpenChange = vi.fn()
    const output = `${'a'.repeat(25_000)}far-away-match${'b'.repeat(25_000)}`

    render(
      <ToolDetailsDialog
        inlineDiff=""
        onOpenChange={onOpenChange}
        open
        part={{ result: { output }, toolName: 'terminal', type: 'tool-call' }}
      />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Search selected section' }), {
      target: { value: 'far-away-match' }
    })

    expect(screen.getByText('1/1')).toBeTruthy()
    expect(screen.getByRole('tabpanel').textContent).toContain('far-away-match')

    fireEvent.click(screen.getByRole('button', { name: 'No wrap' }))
    expect(screen.getByRole('button', { name: 'Wrap' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
