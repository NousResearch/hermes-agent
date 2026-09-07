import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { ComboboxInput } from './combobox-input'

// jsdom doesn't implement these; Radix's Popover (PopoverContent/Arrow) and
// cmdk's Command list use them once the popover actually mounts.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
})

afterEach(cleanup)

const options = ['eleven_multilingual_v2', 'eleven_flash_v2_5', 'voxtral-mini-latest', 'gpt-4o-mini-tts']

function ControlledCombobox({ initialValue = '' }: { initialValue?: string }) {
  const [value, setValue] = useState(initialValue)

  return <ComboboxInput onChange={setValue} options={options} value={value} />
}

function renderCombobox(initialValue = '') {
  render(<ControlledCombobox initialValue={initialValue} />)
}

it('matches a separator-equivalent typed query, same as every other search-fold picker', async () => {
  renderCombobox()

  fireEvent.focus(screen.getByRole('textbox'))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'eleven multilingual' } })

  await screen.findByText('eleven_multilingual_v2')
  expect(screen.queryByText('eleven_flash_v2_5')).toBeNull()
  expect(screen.queryByText('voxtral-mini-latest')).toBeNull()
})

it('still matches a raw substring query (regression guard)', async () => {
  renderCombobox()

  fireEvent.focus(screen.getByRole('textbox'))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'voxtral' } })

  await screen.findByText('voxtral-mini-latest')
  expect(screen.queryByText('eleven_multilingual_v2')).toBeNull()
})

it('shows every option once the value is an exact match', async () => {
  renderCombobox('gpt-4o-mini-tts')

  fireEvent.focus(screen.getByRole('textbox'))

  await screen.findByText('gpt-4o-mini-tts')
  expect(screen.getByText('eleven_multilingual_v2')).not.toBeNull()
  expect(screen.getByText('voxtral-mini-latest')).not.toBeNull()
})
