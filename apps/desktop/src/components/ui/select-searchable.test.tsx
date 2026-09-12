import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './select'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(cleanup)

describe('SelectContent searchable', () => {
  it('filters items as you type', () => {
    render(
      <Select defaultOpen value="openai/gpt-5.6-luna-pro">
        <SelectTrigger aria-label="model">
          <SelectValue />
        </SelectTrigger>
        <SelectContent searchable>
          <SelectItem value="openai/gpt-5.6-luna-pro">openai/gpt-5.6-luna-pro</SelectItem>
          <SelectItem value="anthropic/claude-sonnet-5">anthropic/claude-sonnet-5</SelectItem>
        </SelectContent>
      </Select>
    )

    const input = screen.getByRole('textbox', { name: 'Search…' })
    fireEvent.change(input, { target: { value: 'luna' } })

    expect(screen.getByRole('option', { name: 'openai/gpt-5.6-luna-pro' })).not.toBeNull()
    expect(screen.queryByRole('option', { name: 'anthropic/claude-sonnet-5' })).toBeNull()
  })

  it('shows an empty state when nothing matches', () => {
    render(
      <Select defaultOpen>
        <SelectTrigger aria-label="model">
          <SelectValue placeholder="pick" />
        </SelectTrigger>
        <SelectContent searchable>
          <SelectItem value="openai/gpt-5.6-luna-pro">openai/gpt-5.6-luna-pro</SelectItem>
        </SelectContent>
      </Select>
    )

    fireEvent.change(screen.getByRole('textbox', { name: 'Search…' }), { target: { value: 'zzz-none' } })

    expect(screen.getByText('No results')).not.toBeNull()
    expect(screen.queryByText('openai/gpt-5.6-luna-pro')).toBeNull()
  })
})
