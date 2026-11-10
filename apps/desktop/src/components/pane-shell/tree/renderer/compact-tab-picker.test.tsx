import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { $layoutEditMode } from '../../edit-mode'
import { group } from '../model'
import { $layoutTree, $narrowViewport, $newSessionTabAction } from '../store'

import { TreeGroup } from './tree-group'

const ids = ['workspace', 'session-tile:second', 'session-tile:third']
const disposers: (() => void)[] = []

function Host({ topEdge = false }: { topEdge?: boolean }) {
  const node = useStore($layoutTree)

  return <MemoryRouter>{node?.type === 'group' && <TreeGroup node={node} topEdge={topEdge} />}</MemoryRouter>
}

beforeEach(() => {
  stubResizeObserver()
  stubMenuDomApis()
  vi.stubGlobal('CSS', { escape: (value: string) => value })
  globalThis.document.documentElement.dataset.hermesDesktopHost = 'browser'
  vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
  ids.forEach((id, i) => disposers.push(registry.register({
    area: 'panes', id, title: `Session ${i + 1}`, data: { placement: 'main' },
    render: () => <input aria-label={`Draft ${i + 1}`} defaultValue="" />
  })))
  $layoutTree.set({ ...group(ids), active: ids[0], tabStrip: 'always' })
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  $narrowViewport.set(false)
  $newSessionTabAction.set(null)
  $layoutEditMode.set(false)
  delete globalThis.document.documentElement.dataset.hermesDesktopHost
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('exposes every phone tab and changes the active pane without discarding its draft or siblings', () => {
  render(<Host />)
  fireEvent.change(screen.getByRole('textbox', { name: 'Draft 1' }), { target: { value: 'keep this draft' } })
  fireEvent.pointerDown(screen.getByRole('button', { name: '3 tabs Session 1' }), { button: 0, pointerType: 'mouse', ctrlKey: false })
  expect(screen.getAllByRole('menuitemradio')).toHaveLength(ids.length)
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Session 3' }))
  expect($layoutTree.get()).toMatchObject({ active: ids[2], panes: ids })

  fireEvent.pointerDown(screen.getByRole('button', { name: '3 tabs Session 3' }), { button: 0, pointerType: 'mouse', ctrlKey: false })
  fireEvent.click(screen.getByRole('menuitemradio', { name: 'Session 1' }))
  expect((screen.getByRole('textbox', { name: 'Draft 1' }) as HTMLInputElement).value).toBe('keep this draft')
  expect($layoutTree.get()).toMatchObject({ active: ids[0], panes: ids })
})

it('keeps the normal strip on wider layouts', () => {
  act(() => $narrowViewport.set(false))
  const { container } = render(<Host />)
  expect(container.querySelector('[data-slot="compact-tab-picker"]')).toBeNull()
  expect(container.querySelectorAll('[data-tree-tab]')).toHaveLength(ids.length)
})

it('reserves a separate touch row for phone tabs even when the titlebar reports spare space', () => {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.titlebarCluster === 'left') {
      return new DOMRect(14, 0, 44, 44)
    }

    if (this.dataset.titlebarCluster === 'right') {
      return new DOMRect(242, 0, 176, 44)
    }

    return new DOMRect(0, 0, 430, 844)
  })
  $layoutTree.set({ ...group([ids[0]]), active: ids[0], tabStrip: 'always' })
  const newTab = vi.fn()
  $newSessionTabAction.set(newTab)
  const { container } = render(<><div data-titlebar-cluster="left" /><div data-titlebar-cluster="right" /><Host topEdge /></>)
  const header = container.querySelector<HTMLElement>('[data-panel-header]')!
  const height = () => parseFloat(header.style.height)

  // The chrome leaves 148px, but the drag handle and new-tab target consume
  // 92px of it. The picker needs its own row to keep its label and glyphs apart.
  const touchHeader = height()
  expect(screen.getByRole('button', { name: '1 tab Session 1' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'New session tab' }))
  expect(newTab).toHaveBeenCalledOnce()

  // The edit veil starts below the taller header instead of covering the picker.
  const zone = header.parentElement!
  const before = new Set(zone.children)
  act(() => $layoutEditMode.set(true))
  const veils = [...zone.children].filter(child => !before.has(child)) as HTMLElement[]
  expect(veils).toHaveLength(1)
  expect(veils[0]!.style.top).toBe(header.style.height)
  act(() => $layoutEditMode.set(false))

  act(() => $narrowViewport.set(false))
  expect(height()).toBeLessThan(touchHeader)
  expect(container.querySelector('[data-slot="compact-tab-picker"]')).toBeNull()

  delete globalThis.document.documentElement.dataset.hermesDesktopHost
  act(() => $narrowViewport.set(true))
  expect(height()).toBeLessThan(touchHeader)
  expect(container.querySelector('[data-slot="compact-tab-picker"]')).toBeNull()
})
