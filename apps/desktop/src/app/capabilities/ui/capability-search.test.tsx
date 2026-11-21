import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { CapabilitySearch } from './capability-search'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it.each([false, true])('keeps search, clear, and catalog actions independent (V2=%s)', enabled => {
  $shellChoice.set(enabled)
  const install = vi.fn()

  function Search() {
    const [value, setValue] = useState('')
    return (
      <CapabilitySearch
        actions={<button onClick={install}>Install from Git</button>}
        onChange={setValue}
        placeholder="Search plugins"
        scope="Plugins"
        value={value}
      />
    )
  }

  render(<Search />)
  const input = screen.getByRole<HTMLInputElement>('textbox', { name: 'Search plugins' })
  fireEvent.change(input, { target: { value: 'memory' } })
  fireEvent.click(screen.getByRole('button', { name: 'Install from Git' }))
  expect(install).toHaveBeenCalledOnce()
  expect(input.value).toBe('memory')
  fireEvent.click(screen.getByRole('button', { name: 'Clear search' }))
  expect(input.value).toBe('')
})
