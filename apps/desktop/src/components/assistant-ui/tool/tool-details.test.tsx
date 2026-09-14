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

    expect(findTextMatches(text, 'needle')).toEqual([{ end: 25_006, start: 25_000 }])
  })

  it('searches past 1,000 matches without silently wrapping early', () => {
    const matches = findTextMatches('x'.repeat(1_500), 'x')

    expect(matches).toHaveLength(1_500)
    expect(matches[1_000]).toEqual({ end: 1_001, start: 1_000 })
  })

  it('returns offsets into the original string when Unicode case folding expands', () => {
    expect(findTextMatches('İx', 'x')).toEqual([{ end: 2, start: 1 }])
  })
})

describe('ToolDetailsDialog', () => {
  it('snapshots full serialization for each dialog mount', () => {
    const firstPart = { result: { output: 'first' }, toolName: 'terminal', type: 'tool-call' as const }
    const secondPart = { result: { output: 'second' }, toolName: 'terminal', type: 'tool-call' as const }

    const { rerender, unmount } = render(
      <ToolDetailsDialog inlineDiff="" onOpenChange={() => undefined} open part={firstPart} />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    expect(screen.getByRole('tabpanel').textContent).toBe('first')

    rerender(<ToolDetailsDialog inlineDiff="" onOpenChange={() => undefined} open part={secondPart} />)
    expect(screen.getByRole('tabpanel').textContent).toBe('first')

    unmount()
    render(<ToolDetailsDialog inlineDiff="" onOpenChange={() => undefined} open part={secondPart} />)
    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    expect(screen.getByRole('tabpanel').textContent).toBe('second')
  })

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

  it('scrolls each active full-text match into view, toggles wrapping and dismisses', async () => {
    const onOpenChange = vi.fn()
    const scrollIntoView = vi.fn()
    Element.prototype.scrollIntoView = scrollIntoView
    const output = `${'a'.repeat(25_000)}far-away-match${'b'.repeat(25_000)}far-away-match`

    render(
      <ToolDetailsDialog
        inlineDiff=""
        onOpenChange={onOpenChange}
        open
        part={{ result: { output }, toolName: 'terminal', type: 'tool-call' }}
      />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    scrollIntoView.mockClear()
    fireEvent.change(screen.getByRole('textbox', { name: 'Search selected section' }), {
      target: { value: 'far-away-match' }
    })

    expect(screen.getByText('1/2')).toBeTruthy()
    const firstMatch = screen.getByText('far-away-match')
    expect(firstMatch.getAttribute('data-match-offset')).toBe('25000')
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Next match' }))
    expect(screen.getByText('2/2')).toBeTruthy()
    const secondMatch = screen.getByText('far-away-match')
    expect(secondMatch.getAttribute('data-match-offset')).toBe('50014')
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalledTimes(2))

    fireEvent.click(screen.getByRole('button', { name: 'Previous match' }))
    expect(screen.getByText('1/2')).toBeTruthy()
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalledTimes(3))

    fireEvent.click(screen.getByRole('button', { name: 'No wrap' }))
    expect(screen.getByRole('button', { name: 'Wrap' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('marks the original character at a Unicode-safe search range', () => {
    render(
      <ToolDetailsDialog
        inlineDiff=""
        onOpenChange={() => undefined}
        open
        part={{ result: { output: 'İx' }, toolName: 'terminal', type: 'tool-call' }}
      />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Search selected section' }), { target: { value: 'x' } })

    const match = screen.getByText('x')
    expect(match.getAttribute('data-match-offset')).toBe('1')
    expect(match.textContent).toBe('x')
  })

  it('advances through a long single line without growing the mounted payload', () => {
    const output = `${'a'.repeat(20_000)}${'b'.repeat(20_000)}tail`

    render(
      <ToolDetailsDialog
        inlineDiff=""
        onOpenChange={() => undefined}
        open
        part={{ result: { output }, toolName: 'terminal', type: 'tool-call' }}
      />
    )

    fireEvent.click(screen.getByRole('tab', { name: 'stdout' }))
    expect(screen.getByRole('tabpanel').textContent).toBe('a'.repeat(20_000))

    fireEvent.click(screen.getByRole('button', { name: 'Next chunk' }))
    expect(screen.getByRole('tabpanel').textContent).toBe('b'.repeat(20_000))

    fireEvent.click(screen.getByRole('button', { name: 'Next chunk' }))
    expect(screen.getByRole('tabpanel').textContent).toBe('tail')
    expect(screen.getByRole('tabpanel').textContent?.length).toBeLessThanOrEqual(20_000)
  })
})
