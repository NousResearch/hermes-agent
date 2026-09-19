import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { codiconIcon } from './codicon'
import { SegmentedControl } from './segmented-control'

const options = [
  { id: 'list', label: 'List view', icon: codiconIcon('list-unordered') },
  { id: 'cards', label: 'Card view', icon: codiconIcon('extensions') }
]

afterEach(cleanup)

describe('SegmentedControl', () => {
  it('preserves visible labels by default', () => {
    render(<SegmentedControl onChange={vi.fn()} options={options} value="cards" />)

    expect(screen.getByRole('button', { name: 'Card view', pressed: true }).textContent).toBe('Card view')
    expect(screen.getByRole('button', { name: 'List view', pressed: false }).textContent).toBe('List view')
  })

  it('keeps accessible labels and selection behavior for icon-only controls', () => {
    const onChange = vi.fn()
    render(<SegmentedControl iconOnly onChange={onChange} options={options} value="cards" />)
    const list = screen.getByRole('button', { name: 'List view', pressed: false })
    const cards = screen.getByRole('button', { name: 'Card view', pressed: true })

    expect(list.textContent).toBe('')
    expect(cards.textContent).toBe('')
    expect(list.querySelector('.codicon-list-unordered')).not.toBeNull()
    expect(cards.querySelector('.codicon-extensions')).not.toBeNull()
    fireEvent.click(list)
    expect(onChange).toHaveBeenCalledExactlyOnceWith('list')
  })

  it('retains a visible label when an icon-only option has no icon', () => {
    render(<SegmentedControl iconOnly onChange={vi.fn()} options={[{ id: 'list', label: 'List view' }]} value="list" />)

    expect(screen.getByRole('button', { name: 'List view' }).textContent).toBe('List view')
  })

  it('does not change a disabled icon-only control', () => {
    const onChange = vi.fn()
    render(<SegmentedControl disabled iconOnly onChange={onChange} options={options} value="cards" />)
    const list = screen.getByRole<HTMLButtonElement>('button', { name: 'List view' })

    expect(list.disabled).toBe(true)
    fireEvent.click(list)
    expect(onChange).not.toHaveBeenCalled()
  })
})
