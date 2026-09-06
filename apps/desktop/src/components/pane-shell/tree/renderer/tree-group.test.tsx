import { fireEvent, screen } from '@testing-library/react'
import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { $paneStates, getPaneStateSnapshot } from '@/store/panes'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { $layoutEditMode } from '../../edit-mode'
import type { GroupNode } from '../model'
import { group, split } from '../model'
import { $layoutTree, $treeDragging, declareDefaultTree, NEW_SESSION_DRAG, SESSION_TILE_DRAG } from '../store'

import { TreeGroup } from './tree-group'

let root: null | Root = null
let container: HTMLDivElement | null = null
let disposePane: (() => void) | null = null
const disposers: (() => void)[] = []

beforeAll(() => {
  stubResizeObserver()
  stubMenuDomApis()
})

function render(ui: ReactNode) {
  if (!container) {
    container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    root = createRoot(container)
  }

  act(() => {
    root!.render(ui)
  })
}

function terminalGroup(minimized: boolean): GroupNode {
  return {
    active: 'terminal',
    id: 'terminal-zone',
    minimized,
    panes: ['terminal'],
    // The chevron lives in the strip, so this zone has to be showing one. A
    // lone unregistered pane is on auto and would render none.
    tabStrip: 'always',
    type: 'group'
  }
}

const toggle = (label: string) =>
  globalThis.document.querySelector<HTMLButtonElement>(
    `[data-tree-group="terminal-zone"] button[aria-label="${label}"]`
  )!

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()
  disposePane?.()
  disposers.splice(0).forEach(d => d())
  $paneStates.set({})
  $layoutTree.set(null)
  $layoutEditMode.set(false)
  root = null
  container = null
  disposePane = null
  vi.unstubAllGlobals()
})

describe('TreeGroup', () => {
  describe('ZoneMenu axis lock actions', () => {
    const rowGroup = (): GroupNode => ({
      active: 'sidebar',
      id: 'sidebar-zone',
      minimized: false,
      panes: ['sidebar'],
      tabStrip: 'always',
      type: 'group'
    })

    const columnGroup = (): GroupNode => ({
      active: 'terminal',
      id: 'terminal-zone',
      minimized: false,
      panes: ['terminal'],
      tabStrip: 'always',
      type: 'group'
    })

    /** Radix opens a ContextMenu on contextmenu after a pointerdown positions it. */
    function openContextMenu(target: HTMLElement) {
      fireEvent.pointerDown(target, { button: 2, pointerType: 'mouse' })
      fireEvent.contextMenu(target, { button: 2 })
    }

    it('shows "Lock width" on a row child', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(split('row', [group(['sidebar'], { id: 'sidebar-zone' })]))
      render(<TreeGroup lockAxisRow node={rowGroup()} parentAxis="row" />)

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="sidebar"]')!
      openContextMenu(tab)

      expect(await screen.findByRole('menuitem', { name: /lock width/i })).toBeTruthy()
    })

    it('shows "Lock height" on a column child', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { height: '38vh' },
          id: 'terminal',
          render: () => <div>Terminal</div>,
          title: 'Terminal'
        })
      )
      declareDefaultTree(split('column', [group(['terminal'], { id: 'terminal-zone' })]))
      render(<TreeGroup lockAxisColumn node={columnGroup()} parentAxis="column" />)

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')!
      openContextMenu(tab)

      expect(await screen.findByRole('menuitem', { name: /lock height/i })).toBeTruthy()
    })

    it('shows "Unlock width" when width is already locked', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(split('row', [group(['sidebar'], { id: 'sidebar-zone' })]))
      $paneStates.set({ sidebar: { lockWidth: true, lockedWidth: 237, open: true, widthOverride: 237 } })

      render(<TreeGroup lockAxisRow node={rowGroup()} parentAxis="row" />)

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="sidebar"]')!
      openContextMenu(tab)

      expect(await screen.findByRole('menuitem', { name: /unlock width/i })).toBeTruthy()
    })

    it('shows both "Lock column width" and "Lock height" for a zone in column-within-row (shared-column)', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { height: '38vh', width: '237px' },
          id: 'terminal',
          render: () => <div>Terminal</div>,
          title: 'Terminal'
        })
      )
      declareDefaultTree(split('row', [split('column', [group(['terminal'], { id: 'terminal-zone' })])]))
      // lockAxisRow=true (row ancestor) + lockAxisColumn=true (column parent)
      render(<TreeGroup lockAxisColumn lockAxisRow node={columnGroup()} parentAxis="column" />)

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')!
      openContextMenu(tab)

      expect(await screen.findByRole('menuitem', { name: /lock column width/i })).toBeTruthy()
      expect(await screen.findByRole('menuitem', { name: /lock height/i })).toBeTruthy()
    })

    it('shows "Lock column width" label for a zone in column-within-row', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { height: '38vh', width: '237px' },
          id: 'terminal',
          render: () => <div>Terminal</div>,
          title: 'Terminal'
        })
      )
      const tree = split('row', [split('column', [group(['terminal'], { id: 'terminal-zone' })])])
      declareDefaultTree(tree)
      $layoutTree.set(tree)
      render(<TreeGroup lockAxisColumn lockAxisRow node={columnGroup()} parentAxis="column" />)

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')!
      openContextMenu(tab)

      expect(await screen.findByRole('menuitem', { name: /lock column width/i })).toBeTruthy()
    })

    it('locking column width from one zone locks all zones in the column', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { height: '38vh', width: '237px' },
          id: 'terminal',
          render: () => <div>Terminal</div>,
          title: 'Terminal'
        }),
        registry.register({
          area: 'panes',
          data: { height: '200px', width: '200px' },
          id: 'files',
          render: () => <div>Files</div>,
          title: 'Files'
        })
      )

      const tree = split('row', [
        split('column', [group(['terminal'], { id: 'terminal-zone' }), group(['files'], { id: 'files-zone' })])
      ])

      declareDefaultTree(tree)
      $layoutTree.set(tree)
      render(<TreeGroup lockAxisColumn lockAxisRow node={columnGroup()} parentAxis="column" />)

      // Mock the zone element's bounding rect to 500px (column width)
      const zoneEl = document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')!
      Object.defineProperty(zoneEl, 'getBoundingClientRect', {
        configurable: true,
        value: () => ({ height: 600, width: 500, ...{ toJSON: () => ({}) } })
      })

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="terminal"]')!
      openContextMenu(tab)

      const lockItem = await screen.findByRole('menuitem', { name: /lock column width/i })
      fireEvent.click(lockItem)

      // Both zone panes should be locked
      const termSnap = getPaneStateSnapshot('terminal')
      const filesSnap = getPaneStateSnapshot('files')
      expect(termSnap?.lockWidth).toBe(true)
      expect(termSnap?.lockedWidth).toBe(500)
      expect(filesSnap?.lockWidth).toBe(true)
      expect(filesSnap?.lockedWidth).toBe(500)
    })

    it('lock action captures measured DOM bounding rect width', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(split('row', [group(['sidebar'], { id: 'sidebar-zone' })]))
      render(<TreeGroup lockAxisRow node={rowGroup()} parentAxis="row" />)

      // Mock the zone element's bounding rect to 420px
      const zoneEl = document.querySelector<HTMLElement>('[data-tree-group="sidebar-zone"]')!
      Object.defineProperty(zoneEl, 'getBoundingClientRect', {
        configurable: true,
        value: () => ({ height: 600, width: 420, ...{ toJSON: () => ({}) } })
      })

      const tab = document.querySelector<HTMLElement>('[data-tree-tab="sidebar"]')!
      openContextMenu(tab)

      const lockItem = await screen.findByRole('menuitem', { name: /lock width/i })
      fireEvent.click(lockItem)

      const snap = getPaneStateSnapshot('sidebar')
      expect(snap?.lockWidth).toBe(true)
      expect(snap?.lockedWidth).toBe(420)
    })
  })

  describe('edit-mode veil lock toggle', () => {
    const sidebarGroup = (): GroupNode => ({
      active: 'sidebar',
      id: 'sidebar-zone',
      minimized: false,
      panes: ['sidebar'],
      tabStrip: 'always',
      type: 'group'
    })

    it('renders a lock button on a lockable pane in edit mode', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(split('row', [group(['sidebar'], { id: 'sidebar-zone' })]))
      $layoutEditMode.set(true)
      render(<TreeGroup lockAxisRow node={sidebarGroup()} parentAxis="row" />)

      expect(await screen.findByRole('button', { name: /lock pane/i })).toBeTruthy()
    })

    it('does not render a lock button on a pane with no lockable axis', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(group(['sidebar'], { id: 'sidebar-zone' }))
      $layoutEditMode.set(true)
      render(<TreeGroup node={sidebarGroup()} />)

      expect(screen.queryByRole('button', { name: /lock pane/i })).toBeNull()
    })

    it('toggles the width lock when the corner button is clicked', async () => {
      vi.stubGlobal('CSS', { escape: (value: string) => value })
      disposers.push(
        registry.register({
          area: 'panes',
          data: { width: '237px' },
          id: 'sidebar',
          render: () => <div>Sidebar</div>,
          title: 'Sidebar'
        })
      )
      declareDefaultTree(split('row', [group(['sidebar'], { id: 'sidebar-zone' })]))
      $layoutEditMode.set(true)
      render(<TreeGroup lockAxisRow node={sidebarGroup()} parentAxis="row" />)

      const zoneEl = document.querySelector<HTMLElement>('[data-tree-group="sidebar-zone"]')!
      Object.defineProperty(zoneEl, 'getBoundingClientRect', {
        configurable: true,
        value: () => ({ height: 600, width: 420, ...{ toJSON: () => ({}) } })
      })

      fireEvent.click(await screen.findByRole('button', { name: /lock pane/i }))

      expect(getPaneStateSnapshot('sidebar')?.lockWidth).toBe(true)
      expect(getPaneStateSnapshot('sidebar')?.lockedWidth).toBe(420)

      // The button flips to the unlock action.
      fireEvent.click(await screen.findByRole('button', { name: /unlock pane/i }))

      expect(getPaneStateSnapshot('sidebar')?.lockWidth).toBeUndefined()
    })
  })

  it('points the docked-zone chevron in the collapse or restore action direction', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    // jsdom does not implement CSS.escape, which the real tab-strip effect uses.
    vi.stubGlobal('CSS', { escape: (value: string) => value })

    render(<TreeGroup node={terminalGroup(false)} parentAxis="column" />)

    expect(toggle('Minimize').querySelector('i')!.className).toContain('codicon-chevron-down')

    render(<TreeGroup node={terminalGroup(true)} parentAxis="column" />)

    expect(toggle('Restore').querySelector('i')!.className).toContain('codicon-chevron-up')
  })

  // The invariant behind the shared eligibility predicate
  // (hostsSessionDropTarget): a session or new-session drag must paint the
  // SAME zones either drag resolver would accept, and stay dark everywhere
  // else. Standing chrome (terminal) is always dark; a zone hosting a chat
  // strip (workspace) paints for both sentinels; with no session-drag active
  // nothing paints even over an eligible zone.
  describe('session-drop overlay eligibility (one truth with the resolvers)', () => {
    const groupFor = (panes: string[], id = 'zone-a'): GroupNode => ({
      active: panes[0]!,
      id,
      minimized: false,
      panes,
      type: 'group'
    })

    const sheet = () => globalThis.document.querySelector('[data-tree-group="zone-a"] .pointer-events-none.absolute')

    async function withDragging(dragging: null | string, run: () => void) {
      await act(async () => {
        $treeDragging.set(dragging)
      })

      try {
        run()
      } finally {
        await act(async () => {
          $treeDragging.set(null)
        })
      }
    }

    it('stays dark over standing chrome (terminal) during a new-session drag', async () => {
      disposePane = registry.register({
        area: 'panes',
        data: { height: '12rem' },
        id: 'terminal',
        render: () => <div>Terminal</div>,
        title: 'Terminal'
      })
      vi.stubGlobal('CSS', { escape: (value: string) => value })

      render(<TreeGroup node={terminalGroup(false)} parentAxis="column" />)

      await withDragging(NEW_SESSION_DRAG, () => {
        expect(sheet()).toBeNull()
      })
    })

    it('lights a chat-strip zone for BOTH session and new-session drags, and only then', async () => {
      disposePane = registry.register({
        area: 'panes',
        data: {},
        id: 'workspace',
        render: () => <div>Chat</div>,
        title: 'Hermes'
      })
      vi.stubGlobal('CSS', { escape: (value: string) => value })

      render(<TreeGroup node={groupFor(['workspace'])} parentAxis="column" />)

      await withDragging(null, () => {
        expect(sheet()).toBeNull()
      })

      await withDragging(SESSION_TILE_DRAG, () => {
        expect(sheet()).not.toBeNull()
      })

      await withDragging(NEW_SESSION_DRAG, () => {
        expect(sheet()).not.toBeNull()
      })
    })
  })
})
