import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CompletionSummaryDialog, needsCompletionSummary } from './completion-dialog'

afterEach(cleanup)

describe('needsCompletionSummary', () => {
  it('requires a summary for done from a non-review status', () => {
    expect(needsCompletionSummary('done', 'ready')).toBe(true)
    expect(needsCompletionSummary('done', 'todo')).toBe(true)
    expect(needsCompletionSummary('done', 'blocked')).toBe(true)
  })

  it('exempts review -> done (human approval) and non-done moves', () => {
    expect(needsCompletionSummary('done', 'review')).toBe(false)
    expect(needsCompletionSummary('ready', 'todo')).toBe(false)
    expect(needsCompletionSummary('blocked', 'running')).toBe(false)
  })
})

describe('CompletionSummaryDialog', () => {
  it('keeps Complete disabled until a non-empty summary is typed', () => {
    const onConfirm = vi.fn()
    render(<CompletionSummaryDialog count={1} onCancel={vi.fn()} onConfirm={onConfirm} />)

    const complete = screen.getByRole('button', { name: 'Complete' }) as HTMLButtonElement
    expect(complete.disabled).toBe(true)

    // Whitespace is not evidence.
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '   ' } })
    expect(complete.disabled).toBe(true)
    fireEvent.click(complete)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('forwards the trimmed summary on confirm', () => {
    const onConfirm = vi.fn()
    render(<CompletionSummaryDialog count={1} onCancel={vi.fn()} onConfirm={onConfirm} />)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '  shipped the fix  ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Complete' }))
    expect(onConfirm).toHaveBeenCalledWith('shipped the fix')
  })

  it('confirms on Cmd/Ctrl+Enter', () => {
    const onConfirm = vi.fn()
    render(<CompletionSummaryDialog count={1} onCancel={vi.fn()} onConfirm={onConfirm} />)

    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'done via keyboard' } })
    fireEvent.keyDown(box, { ctrlKey: true, key: 'Enter' })
    expect(onConfirm).toHaveBeenCalledWith('done via keyboard')
  })

  it('cancels without confirming', () => {
    const onCancel = vi.fn()
    const onConfirm = vi.fn()
    render(<CompletionSummaryDialog count={1} onCancel={onCancel} onConfirm={onConfirm} />)

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onConfirm).not.toHaveBeenCalled()
  })

  it('reflects a bulk count in its copy', () => {
    render(<CompletionSummaryDialog count={3} onCancel={vi.fn()} onConfirm={vi.fn()} />)
    expect(screen.getByText(/3 tasks/)).toBeTruthy()
  })
})
