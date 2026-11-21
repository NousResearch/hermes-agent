import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Settings2 } from '@/lib/icons'
import { setSidebarOpen } from '@/store/layout'

import { OverlayNav, type OverlayNavGroup } from './overlay-split-layout'
import { PageToolbar } from './page-toolbar'

afterEach(() => {
  cleanup()
  act(() => setSidebarOpen(true))
})

it('only a titled nav is the page panel the band offers to show again once folded', () => {
  const groups: OverlayNavGroup[] = [
    { active: true, id: 'overview', label: 'Overview', icon: Settings2, onSelect: vi.fn() }
  ]

  act(() => setSidebarOpen(false))

  const { rerender } = render(
    <>
      <PageToolbar onBack={vi.fn()} />
      <OverlayNav groups={groups} />
    </>
  )

  // An untitled nav (Command Center) has no panel to fold, so a "Show sidebar" there would do nothing.
  expect(screen.queryByRole('button', { name: 'Show sidebar' })).toBeNull()

  rerender(
    <>
      <PageToolbar onBack={vi.fn()} />
      <OverlayNav groups={groups} title="Settings" />
    </>
  )
  expect(screen.getByRole('button', { name: 'Show sidebar' })).toBeTruthy()
})

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

it('fills only the current page when a third-level sub-page is open; its parent is named, not filled', () => {
  const noop = vi.fn()

  const groups: OverlayNavGroup[] = [
    {
      active: true,
      id: 'plugins',
      label: 'Plugins',
      icon: Settings2,
      onSelect: noop,
      children: [
        {
          active: true,
          id: 'hello',
          label: 'Hello Runtime',
          icon: Settings2,
          onSelect: noop,
          children: [{ active: true, id: 'about', label: 'About', icon: Settings2, onSelect: noop }]
        }
      ]
    }
  ]

  const { container } = render(<OverlayNav groups={groups} />)
  const rail = within(container.querySelector('[data-tour="overlay-nav"]') as HTMLElement)
  const parent = rail.getByRole('button', { name: 'Hello Runtime' })
  const page = rail.getByRole('button', { name: 'About' })

  expect(page.getAttribute('aria-current')).toBe('page')
  expect(page.className).toContain('bg-(--chrome-action-hover)')
  expect(parent.getAttribute('aria-current')).toBeNull()
  expect(parent.className).toContain('bg-transparent')
  expect(parent.className).toContain('font-medium')
})
