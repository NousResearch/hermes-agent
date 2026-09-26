import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { SearchableSelect } from './searchable-select'

beforeAll(() => {
  stubResizeObserver()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn(() => false)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

/** Open the palette and return its list container, so text queries hit rows
 *  and never the trigger's own (identical) selected label. */
async function openPalette() {
  fireEvent.click(screen.getByRole('combobox'))

  const list = await waitFor(() => {
    const el = document.querySelector('[data-slot="command-list"]')

    if (!el) {
      throw new Error('palette not open')
    }

    return el
  })

  return within(list as HTMLElement)
}

describe('SearchableSelect', () => {
  it('renders a combobox trigger showing the selected option label', () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={[
          { label: 'Nous Portal (nous)', value: 'nous' },
          { label: 'OpenRouter', value: 'openrouter' }
        ]}
        value="nous"
      />
    )

    expect(screen.getByRole('combobox').textContent).toContain('Nous Portal (nous)')
  })

  it('falls back to the raw value when the selection is out of catalog', () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={[{ label: 'OpenRouter', value: 'openrouter' }]}
        value="custom-gateway"
      />
    )

    expect(screen.getByRole('combobox').textContent).toContain('custom-gateway')
  })

  it('shows the placeholder when nothing is selected', () => {
    render(<SearchableSelect onChange={vi.fn()} options={['a']} placeholder="Pick one…" value="" />)

    expect(screen.getByRole('combobox').textContent).toContain('Pick one…')
  })

  it('emits the option value (not the label) on select', async () => {
    const onChange = vi.fn()

    render(
      <SearchableSelect
        onChange={onChange}
        options={[
          { label: 'Nous Portal (nous)', value: 'nous' },
          { label: 'OpenRouter', value: 'openrouter' }
        ]}
        value="nous"
      />
    )

    const rows = await openPalette()

    fireEvent.click(rows.getByText('OpenRouter'))
    expect(onChange).toHaveBeenCalledWith('openrouter')
  })

  it('accepts plain string options', async () => {
    const onChange = vi.fn()

    render(<SearchableSelect onChange={onChange} options={['gpt-4o', 'o3-mini']} value="gpt-4o" />)

    const rows = await openPalette()

    fireEvent.click(rows.getByText('o3-mini'))
    expect(onChange).toHaveBeenCalledWith('o3-mini')
  })

  it('filters by substring while typing', async () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={['claude-sonnet-4', 'gpt-4o', 'gpt-5']}
        value="gpt-4o"
      />
    )

    const rows = await openPalette()

    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'gpt-5' } })

    await waitFor(() => {
      expect(rows.queryByText('gpt-5')).toBeTruthy()
      expect(rows.queryByText('claude-sonnet-4')).toBeNull()
    })
  })

  it('matches alias keywords beyond the visible label', async () => {
    render(
      <SearchableSelect
        onChange={vi.fn()}
        options={[
          { keywords: ['kimi', 'kimi-k3'], label: 'k3', value: 'k3' },
          { label: 'gpt-4o', value: 'gpt-4o' }
        ]}
        value="k3"
      />
    )

    const rows = await openPalette()

    // "kimi" is not in the label — only the keyword haystack carries it.
    fireEvent.change(screen.getByPlaceholderText('Search…'), { target: { value: 'kimi' } })

    await waitFor(() => {
      expect(rows.queryByText('k3')).toBeTruthy()
      expect(rows.queryByText('gpt-4o')).toBeNull()
    })
  })

  it('renders grouped options under headings', async () => {
    render(
      <SearchableSelect
        groups={[
          { options: [{ label: 'Default (global model)', value: '__default__' }] },
          {
            label: 'OpenRouter',
            options: [
              { label: 'gpt-4o', value: 'openrouter:gpt-4o' },
              { label: 'o3-mini', value: 'openrouter:o3-mini' }
            ]
          }
        ]}
        onChange={vi.fn()}
        value="__default__"
      />
    )

    const rows = await openPalette()

    expect(rows.getByText('Default (global model)')).toBeTruthy()
    expect(rows.getByText('OpenRouter')).toBeTruthy()
    expect(rows.getByText('gpt-4o')).toBeTruthy()
  })

  it('is a same-value no-op (Radix parity for autosaving callers)', async () => {
    const onChange = vi.fn()

    render(<SearchableSelect onChange={onChange} options={['a', 'b']} value="a" />)

    const rows = await openPalette()

    fireEvent.click(rows.getByText('a'))

    expect(onChange).not.toHaveBeenCalled()
    // The palette still dismisses on pick, even when nothing changes.
    await waitFor(() => expect(document.querySelector('[data-slot="command-list"]')).toBeNull())
  })

  it('clears the value via the optional clear item', async () => {
    const onChange = vi.fn()

    render(<SearchableSelect clearLabel="None" onChange={onChange} options={['a']} value="a" />)

    const rows = await openPalette()

    fireEvent.click(rows.getByText('None'))
    expect(onChange).toHaveBeenCalledWith('')
  })
})
