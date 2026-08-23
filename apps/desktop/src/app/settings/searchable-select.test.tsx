import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'
import type { ConfigFieldSchema } from '@/types/hermes'

import { ConfigField } from './config-field'
import { rankSearchOption, SearchableSelect } from './searchable-select'

beforeAll(() => {
  stubResizeObserver()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('rankSearchOption', () => {
  it('ranks a final-segment match above a mid-path match', () => {
    // "york" hits the city segment of America/New_York (score 2) but only a
    // mid-path segment of America/New_York/Special (score 1).
    expect(rankSearchOption('America/New_York', 'york')).toBe(2)
    expect(rankSearchOption('America/New_York/Special', 'york')).toBe(1)
    expect(rankSearchOption('America/New_York', 'york')).toBeGreaterThan(
      rankSearchOption('America/New_York/Special', 'york')
    )
  })

  it('is case-insensitive', () => {
    expect(rankSearchOption('Asia/Kolkata', 'KOLKATA')).toBe(2)
    expect(rankSearchOption('ASIA/KOLKATA', 'kolkata')).toBe(2)
  })

  it('scores a substring match anywhere as 1', () => {
    expect(rankSearchOption('America/New_York', 'amer')).toBe(1)
  })

  it('scores a slashless option by plain substring', () => {
    expect(rankSearchOption('UTC', 'ut')).toBe(1)
    expect(rankSearchOption('UTC', 'xyz')).toBe(0)
  })

  it('scores a non-match as 0', () => {
    expect(rankSearchOption('Europe/Berlin', 'tokyo')).toBe(0)
  })
})

describe('SearchableSelect', () => {
  const options = ['America/New_York', 'Asia/Kolkata', 'Europe/Berlin', 'UTC']

  it('opens, filters, and selects an option', () => {
    const onChange = vi.fn()

    render(<SearchableSelect onChange={onChange} options={options} placeholder="Search…" value="" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'kolkata' } })
    fireEvent.click(screen.getByText('Asia/Kolkata'))

    expect(onChange).toHaveBeenCalledWith('Asia/Kolkata')
  })

  it('renders the clear item when clearLabel is set and selecting it resets to blank', () => {
    const onChange = vi.fn()

    render(<SearchableSelect clearLabel="System default" onChange={onChange} options={options} value="Asia/Kolkata" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })

  it('omits the clear item without clearLabel', () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} value="" />)

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.queryByText('System default')).toBeNull()
  })

  it('lets the option list size to its content instead of the shrink-wrapped trigger', () => {
    // The trigger shrink-wraps to its current value inside the settings grid
    // (~120px for "Europe/Berlin"); pinning the popover width to it clipped
    // every IANA row after "Africa/A…". The popover may grow to cover a wider
    // trigger, but must never be capped at the trigger's width.
    render(<SearchableSelect onChange={vi.fn()} options={options} value="Europe/Berlin" />)

    fireEvent.click(screen.getByRole('combobox'))

    const content = screen.getByRole('listbox', { hidden: true }).closest('[data-slot="popover-content"]')
    const classes = content?.className.split(/\s+/) ?? []

    expect(classes.some(c => c.startsWith('min-w-') && c.includes('radix-popover-trigger-width'))).toBe(true)
    expect(classes.some(c => /^w-[[(].*radix-popover-trigger-width/.test(c))).toBe(false)
  })

  it('shows the placeholder when the value is blank', () => {
    render(<SearchableSelect onChange={vi.fn()} options={options} placeholder="Search…" value="" />)

    expect(screen.getByRole('combobox').textContent).toContain('Search…')
  })
})

describe('ConfigField searchable routing', () => {
  const searchableSchema: ConfigFieldSchema = {
    type: 'select',
    searchable: true,
    clearable: true,
    options: ['America/New_York', 'UTC']
  }

  it('routes searchable select schemas to SearchableSelect, not a free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={searchableSchema} schemaKey="timezone" value="UTC" />
    )

    // The searchable trigger renders; the generic free-text <Input> does not.
    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).not.toBeNull()
    expect(container.querySelector('input[type="text"]')).toBeNull()
  })

  it('keeps plain string schemas on the free-text input', () => {
    const { container } = render(
      <ConfigField onChange={vi.fn()} schema={{ type: 'string' }} schemaKey="some.other.key" value="hello" />
    )

    expect(container.querySelector('[data-slot="searchable-select-trigger"]')).toBeNull()
    expect(screen.getByDisplayValue('hello')).not.toBeNull()
  })

  it('surfaces the clear item via schema.clearable and resets to blank', () => {
    const onChange = vi.fn()

    render(<ConfigField onChange={onChange} schema={searchableSchema} schemaKey="timezone" value="UTC" />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('System default'))

    expect(onChange).toHaveBeenCalledWith('')
  })
})

describe('SearchableSelect rich options', () => {
  it('shows the selected option label on the trigger', () => {
    render(
      <SearchableSelect onChange={vi.fn()} options={[{ value: 'openrouter', label: 'OpenRouter' }]} value="openrouter" />
    )

    expect(screen.getByRole('combobox').textContent).toContain('OpenRouter')
  })

  it('falls back to the raw value when the selection is not in the options', () => {
    // Mirrors withActive / missing-saved-provider: an out-of-catalog model
    // still renders its id instead of a blank trigger.
    render(<SearchableSelect onChange={vi.fn()} options={['hermes-4']} value="hermes-4-mini" />)

    expect(screen.getByRole('combobox').textContent).toContain('hermes-4-mini')
  })

  it('round-trips a cased, slashed value through selection', () => {
    const onChange = vi.fn()

    render(
      <SearchableSelect
        onChange={onChange}
        options={[{ value: 'anthropic/claude-opus-4.8', label: 'Claude Opus 4.8' }]}
        placeholder="Search…"
        value=""
      />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByText('Claude Opus 4.8'))

    expect(onChange).toHaveBeenCalledWith('anthropic/claude-opus-4.8')
  })

  it('does not re-emit when the current value is re-picked (Radix no-op parity)', () => {
    const onChange = vi.fn()

    render(
      <SearchableSelect onChange={onChange} options={[{ value: 'nous', label: 'Nous' }]} placeholder="Search…" value="nous" />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByRole('option', { name: 'Nous' }))

    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('SearchableSelect keywords', () => {
  it('folds item keywords into the filter score', () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={[{ value: 'openrouter', label: 'OpenRouter', keywords: ['OpenRouter'] }]}
        placeholder="Search…"
        value=""
      />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'openrouter' } })

    expect(screen.getByText('OpenRouter')).toBeTruthy()
  })

  it('keeps an opaque id findable via its alias haystack', () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={[{ value: 'k3', keywords: ['k3 kimi-k3 kimi'] }]}
        placeholder="Search…"
        value=""
      />
    )

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'kimi' } })

    // 'k3' alone scores 0 against 'kimi'; only the folded keywords keep it listed.
    expect(screen.getByText('k3')).toBeTruthy()
  })
})

describe('SearchableSelect className passthrough', () => {
  it('merges the extra class onto the trigger', () => {
    const { container } = render(
      <SearchableSelect className="min-w-60" onChange={vi.fn()} options={['hermes-4']} value="" />
    )

    expect(container.querySelector('[data-slot="searchable-select-trigger"]')?.className).toContain('min-w-60')
  })
})
