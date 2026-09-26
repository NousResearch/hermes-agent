import { cleanup, fireEvent, render, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Settings2 } from '@/lib/icons'

import { OverlayNav, type OverlayNavGroup } from './overlay-split-layout'

afterEach(cleanup)

it('expands independently of navigation and reveals a newly selected child', () => {
  const select = vi.fn()

  const group = (active: boolean, child: string): OverlayNavGroup => ({
    active,
    id: 'appearance',
    label: 'Appearance',
    icon: Settings2,
    onSelect: select,
    children: ['General', 'Theme'].map(label => ({
      active: active && child === label,
      id: label,
      label,
      icon: Settings2,
      onSelect: select
    }))
  })

  const { container, rerender } = render(<OverlayNav groups={[group(false, '')]} />)
  const rail = within(container.querySelector('[data-tour="overlay-nav"]') as HTMLElement)
  fireEvent.click(rail.getByRole('button', { name: 'Expand: Appearance' }))
  expect(select).not.toHaveBeenCalled()
  expect(rail.getByRole('button', { name: 'Theme' })).toBeTruthy()
  fireEvent.click(rail.getByRole('button', { name: 'Theme' }))
  expect(select).toHaveBeenCalledOnce()
  rerender(<OverlayNav groups={[group(true, 'Theme')]} />)
  fireEvent.click(rail.getByRole('button', { name: 'Collapse: Appearance' }))
  expect(select).toHaveBeenCalledOnce()
  expect(rail.queryByRole('button', { name: 'Theme' })).toBeNull()
  rerender(<OverlayNav groups={[group(true, 'General')]} />)
  expect(rail.getByRole('button', { name: 'General' }).getAttribute('aria-current')).toBe('page')
  expect(rail.getByRole('button', { name: 'Collapse: Appearance' }).getAttribute('aria-expanded')).toBe('true')
})
