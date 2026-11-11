import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $shellChoice } from '@/store/shell-variant'

import { Panel, PanelHeader, PanelListRow, PanelRowMenu } from './panel'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.setPointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

describe('PanelRowMenu', () => {
  it('opens its actions menu from the kebab without a tooltip', async () => {
    const onSelect = vi.fn()

    render(<PanelRowMenu items={[{ label: 'Rename', onSelect }]} />)

    const trigger = screen.getByRole('button', { name: 'Actions' })

    expect(trigger.closest('[data-slot="tooltip-trigger"]')).toBeNull()

    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename' }))

    expect(onSelect).toHaveBeenCalledOnce()
  })
})

it('uses a default Portal page heading and preserves row selection when returning to classic', () => {
  $shellChoice.set(null)
  const select = vi.fn()
  render(
    <Panel onClose={vi.fn()}>
      <PanelHeader title={en.cron.title} />
      <PanelListRow active onSelect={select} title="Status report" />
    </Panel>
  )
  expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
  const row = screen.getByRole('button', { name: 'Status report' })
  expect(row.getAttribute('aria-current')).toBe('true')
  fireEvent.click(row)
  expect(select).toHaveBeenCalledOnce()

  act(() => $shellChoice.set(false))
  expect(screen.getByRole('heading', { level: 2 })).toBeTruthy()
  expect(screen.queryByRole('heading', { level: 1 })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Status report' }))
  expect(select).toHaveBeenCalledTimes(2)
})

it('keeps the default Portal heading for an empty page without changing classic empty states', () => {
  $shellChoice.set(false)
  const { rerender } = render(<PanelHeader hideInClassic title="Agents" />)
  expect(screen.queryByRole('heading')).toBeNull()
  act(() => $shellChoice.set(null))
  expect(screen.getByRole('heading', { level: 1, name: 'Agents' })).toBeTruthy()
  rerender(<PanelHeader hideInClassic={false} title="Agents" />)
  expect(screen.getAllByRole('heading')).toHaveLength(1)
})
