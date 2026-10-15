// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { ComboboxInput } from './combobox-input'

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(cleanup)

function Field({ searchable = true }: { searchable?: boolean }) {
  const [value, setValue] = useState('my-custom-deployment')
  return (
    <ComboboxInput
      aria-label="Deployment"
      value={value}
      onChange={setValue}
      options={['gpt-model', 'claude-model']}
      search={searchable ? { placeholder: 'Search models', emptyMessage: 'No matching models' } : undefined}
    />
  )
}

it('searches independently of the selected value and keeps an empty result searchable', () => {
  render(<Field />)
  fireEvent.click(screen.getByRole('button', { name: 'Show options' }))
  const search = screen.getByPlaceholderText('Search models')
  fireEvent.change(search, { target: { value: 'GPT' } })
  expect(screen.getByText('gpt-model')).toBeTruthy()
  expect(screen.queryByText('claude-model')).toBeNull()
  expect(screen.getByLabelText('Deployment')).toHaveProperty('value', 'my-custom-deployment')
  fireEvent.change(search, { target: { value: 'missing' } })
  expect(screen.getByText('No matching models')).toBeTruthy()
  fireEvent.change(search, { target: { value: 'claude' } })
  fireEvent.click(screen.getByText('claude-model'))
  expect(screen.getByLabelText('Deployment')).toHaveProperty('value', 'claude-model')
  fireEvent.click(screen.getByRole('button', { name: 'Show options' }))
  expect(screen.getByPlaceholderText('Search models')).toHaveProperty('value', '')
  expect(screen.getByText('gpt-model')).toBeTruthy()
})

it('preserves manual entry and the existing autocomplete behavior without menu search', () => {
  render(<Field searchable={false} />)
  fireEvent.change(screen.getByLabelText('Deployment'), { target: { value: 'claude' } })
  expect(screen.getByText('claude-model')).toBeTruthy()
  expect(screen.queryByText('gpt-model')).toBeNull()
  fireEvent.click(screen.getByText('claude-model'))
  expect(screen.getByLabelText('Deployment')).toHaveProperty('value', 'claude-model')
  fireEvent.change(screen.getByLabelText('Deployment'), { target: { value: 'unlisted-deployment' } })
  expect(screen.getByLabelText('Deployment')).toHaveProperty('value', 'unlisted-deployment')
})
