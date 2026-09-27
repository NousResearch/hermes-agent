// Exercise the real Sessions metadata and the shell’s connection-count subscription.
import '@/app/contrib/controller'

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $layoutEditMode } from '@/components/pane-shell/edit-mode'
import type { GroupNode } from '@/components/pane-shell/tree/model'
import { startPaneDrag } from '@/components/pane-shell/tree/renderer/drag-session'
import { TreeGroup } from '@/components/pane-shell/tree/renderer/tree-group'
import { $hiddenTreePanes, $narrowViewport } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { $connectionsRegistry, selectConnection } from '@/store/connections'
import { stubMenuDomApis } from '@/test/jsdom'

vi.mock('@/components/pane-shell/tree/renderer/drag-session', () => ({ startPaneDrag: vi.fn() }))

vi.mock(import('@/store/connections'), async importOriginal => ({
  ...(await importOriginal()),
  selectConnection: vi.fn(async () => undefined)
}))

const local = { id: 'local', kind: 'local' as const, label: 'This device', tokenPreview: null, tokenSet: false }
const remote = { ...local, id: 'remote', kind: 'remote' as const, label: 'Homelab' }

const node: GroupNode = {
  active: 'sessions',
  id: 'sidebar-zone',
  panes: ['sessions', 'hermes-bots:pane'],
  type: 'group'
}

const setConnections = (multiple: boolean) =>
  $connectionsRegistry.set({
    connections: multiple ? [local, remote] : [local],
    primary: 'local',
    secureTokenStorage: true,
    version: 2
  })

const view = (group = node) => (
  <MemoryRouter>
    <SidebarProvider>
      <TreeGroup node={group} parentAxis="row" topEdge />
    </SidebarProvider>
  </MemoryRouter>
)

let disposeBots: () => void

beforeEach(() => {
  stubMenuDomApis()
  // Pane bodies are unrelated to zone chrome; retain the production metadata.
  const sessions = registry.getArea('panes').find(pane => pane.id === 'sessions')!
  registry.register({ ...sessions, render: () => null })
  disposeBots = registry.register({
    area: 'panes',
    id: 'hermes-bots:pane',
    title: 'Bots',
    data: { collapsible: true, connectionScoped: true },
    render: () => null
  })
})

afterEach(() => {
  cleanup()
  disposeBots()
  $connectionsRegistry.set(null)
  $hiddenTreePanes.set(new Set())
  $narrowViewport.set(false)
  $layoutEditMode.set(false)
  vi.clearAllMocks()
})

describe('the sidebar connection header', () => {
  it('belongs above both tabs and uses the existing connection switch path', () => {
    setConnections(true)
    const { container, rerender } = render(view())
    const switcher = screen.getByRole('group', { name: 'Registered gateways' })
    const tabs = screen.getByRole('tablist')

    expect(switcher.closest('[data-tree-group]')).toBe(tabs.closest('[data-tree-group]'))
    expect(switcher.compareDocumentPosition(tabs) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(switcher.querySelector('[data-sidebar="menu-button"]')?.getAttribute('data-size')).toBe('nav')

    rerender(view({ ...node, active: 'hermes-bots:pane' }))
    expect(screen.getByRole('tab', { name: 'Bots' }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('group', { name: 'Registered gateways' })).toBe(switcher)

    fireEvent.pointerDown(screen.getByRole('button', { name: /Registered gateways/ }), {
      button: 0,
      pointerType: 'mouse'
    })
    fireEvent.click(screen.getByRole('menuitemradio', { name: 'Homelab' }))
    expect(selectConnection).toHaveBeenCalledExactlyOnceWith('remote')

    // Bots owns machine scope even in its own zone.
    rerender(view({ ...node, active: 'hermes-bots:pane', panes: ['hermes-bots:pane'] }))
    expect(screen.getByRole('group', { name: 'Registered gateways' })).toBeTruthy()
    rerender(view({ ...node, minimized: true }))
    expect(container.querySelector('[data-slot="connection-switcher"]')).toBeNull()
  })

  it('scopes only visible navigation tabs outside the main zone', () => {
    setConnections(true)
    const { container, rerender } = render(view())
    const hasHeader = () => Boolean(container.querySelector('[data-slot="connection-switcher"]'))

    for (const hidden of node.panes) {
      act(() => $hiddenTreePanes.set(new Set([hidden])))
      expect(screen.queryByRole('tablist')).toBeNull()
      expect(hasHeader()).toBe(true)
    }

    act(() => $hiddenTreePanes.set(new Set(node.panes)))
    expect(hasHeader()).toBe(false)
    act(() => $hiddenTreePanes.set(new Set()))
    act(() => $narrowViewport.set(true))
    expect(hasHeader()).toBe(false)
    act(() => $narrowViewport.set(false))
    // Hide tabs (⌘⌥T) writes this preference; it hides only the strip.
    rerender(view({ ...node, tabStrip: 'never' }))
    expect(screen.queryByRole('tablist')).toBeNull()
    expect(hasHeader()).toBe(true)
    rerender(view({ ...node, panes: ['workspace', ...node.panes] }))
    expect(hasHeader()).toBe(false)
    const mainHeader = container.querySelector('[data-panel-header]')!.outerHTML

    act(() => setConnections(false))
    expect(container.querySelector('[data-panel-header]')!.outerHTML).toBe(mainHeader)
  })

  it('contains the edit veil in the body and routes tab presses to their own drag handlers', () => {
    setConnections(true)
    $layoutEditMode.set(true)
    const { container } = render(view())
    const veil = container.querySelector('[data-zone-edit-overlay]')!
    const body = veil.parentElement!
    const strip = screen.getByRole('tablist')
    const switcher = screen.getByRole('group', { name: 'Registered gateways' })

    // The containing block flows after ALL header content; inset-0 covers only
    // the body even as the header's height changes.
    expect(body.classList.contains('relative')).toBe(true)
    expect(veil.classList.contains('inset-0')).toBe(true)
    expect(body.contains(strip)).toBe(false)
    expect(body.contains(switcher)).toBe(false)
    expect(strip.compareDocumentPosition(body) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    for (const [name, id] of [
      ['Sessions', 'sessions'],
      ['Bots', 'hermes-bots:pane']
    ]) {
      fireEvent.pointerDown(screen.getByRole('tab', { name }), { button: 0, pointerType: 'mouse' })
      expect(vi.mocked(startPaneDrag).mock.calls.at(-1)?.[0]).toBe(id)
    }
  })

  it('leaves single-connection zone chrome unchanged when gateways are added and removed', () => {
    $connectionsRegistry.set(null)
    const panes = registry.getArea('panes')
    const { container } = render(view())
    const header = container.querySelector('[data-panel-header]')!
    const originalHeader = header.outerHTML
    const originalChildren = [...header.parentElement!.children]

    expect(container.querySelector('[data-slot="connection-switcher"]')).toBeNull()
    act(() => setConnections(false))
    expect(container.querySelector('[data-panel-header]')!.outerHTML).toBe(originalHeader)
    expect([...header.parentElement!.children]).toEqual(originalChildren)
    act(() => setConnections(true))
    expect(screen.getByRole('group', { name: 'Registered gateways' })).toBeTruthy()
    act(() => setConnections(false))

    expect(container.querySelector('[data-slot="connection-switcher"]')).toBeNull()
    expect(container.querySelector('[data-panel-header]')).toBe(header)
    expect(header.outerHTML).toBe(originalHeader)
    expect([...header.parentElement!.children]).toEqual(originalChildren)
    expect(registry.getArea('panes')).toBe(panes)
  })
})
