import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

afterEach(() => {
  cleanup()
})

describe('Markdown task-list checkboxes', () => {
  it('preserves the Markdown state and lets the user toggle it locally', () => {
    const text = '- [ ] Open item\n- [x] Completed item'
    const { rerender, unmount } = render(<MarkdownTextContent isRunning={false} text={text} />)

    const [openItem, completedItem] = screen.getAllByRole<HTMLInputElement>('checkbox')

    expect(openItem.disabled).toBe(false)
    expect(openItem.checked).toBe(false)
    expect(completedItem.disabled).toBe(false)
    expect(completedItem.checked).toBe(true)

    fireEvent.click(openItem)
    fireEvent.click(completedItem)
    rerender(<MarkdownTextContent isRunning={false} text={`${text}\n\nMore text`} />)

    const [updatedOpenItem, updatedCompletedItem] = screen.getAllByRole<HTMLInputElement>('checkbox')

    expect(updatedOpenItem.checked).toBe(true)
    expect(updatedCompletedItem.checked).toBe(false)

    unmount()
    render(<MarkdownTextContent isRunning={false} text={text} />)

    const [resetOpenItem, resetCompletedItem] = screen.getAllByRole<HTMLInputElement>('checkbox')

    expect(resetOpenItem.checked).toBe(false)
    expect(resetCompletedItem.checked).toBe(true)
  })

  it('keeps checkboxes outside GFM task-list items disabled', () => {
    render(<MarkdownTextContent isRunning={false} text={'<input type="checkbox" checked disabled>'} />)

    const checkbox = screen.getByRole<HTMLInputElement>('checkbox')

    expect(checkbox.disabled).toBe(true)
    expect(checkbox.checked).toBe(true)

    fireEvent.click(checkbox)

    expect(checkbox.checked).toBe(true)
  })
})
