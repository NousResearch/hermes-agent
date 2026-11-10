import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Switch } from './switch'

afterEach(cleanup)

it.each([
  ['ltr', 'default'],
  ['ltr', 'xs'],
  ['rtl', 'default'],
  ['rtl', 'xs']
] as const)('keeps the %s %s switch interactive inside an oppositely directed surface', (dir, size) => {
  const onCheckedChange = vi.fn()
  render(
    <div dir={dir === 'rtl' ? 'ltr' : 'rtl'}>
      <div dir={dir}>
        <Switch aria-label="Thinking" onCheckedChange={onCheckedChange} size={size} />
      </div>
    </div>
  )
  const control = screen.getByRole('switch', { name: 'Thinking' })
  expect(control.getAttribute('aria-checked')).toBe('false')
  fireEvent.click(control)
  expect(control.getAttribute('aria-checked')).toBe('true')
  expect(onCheckedChange).toHaveBeenLastCalledWith(true)
  fireEvent.click(control)
  expect(control.getAttribute('aria-checked')).toBe('false')
  expect(onCheckedChange).toHaveBeenLastCalledWith(false)
})
