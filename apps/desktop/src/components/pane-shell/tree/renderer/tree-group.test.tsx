import { fireEvent, screen } from '@testing-library/react'
import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { $layoutEditMode } from '../../edit-mode'
import type { GroupNode } from '../model'

import { TreeGroup } from './tree-group'

let root: null | Root = null
let container: HTMLDivElement | null = null
let disposePane: (() => void) | null = null

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
  root = null
  container = null
  disposePane = null
  $layoutEditMode.set(false)
  vi.unstubAllGlobals()
})

describe('TreeGroup', () => {
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

  it('scopes a semantic background tint to the selected zone', () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })

    render(<TreeGroup node={{ ...terminalGroup(false), backgroundTint: 'cyan' }} parentAxis="column" />)

    const zone = globalThis.document.querySelector<HTMLElement>('[data-tree-group="terminal-zone"]')

    expect(zone?.style.getPropertyValue('--ui-chat-surface-background')).toBe(
      'color-mix(in srgb, var(--ui-cyan) 10%, var(--ui-zone-chat-surface-background))'
    )
    expect(zone?.style.getPropertyValue('--ui-editor-surface-background')).toBe(
      'color-mix(in srgb, var(--ui-cyan) 10%, var(--ui-zone-editor-surface-background))'
    )
    expect(zone?.style.getPropertyValue('--ui-sidebar-surface-background')).toBe(
      'color-mix(in srgb, var(--ui-cyan) 10%, var(--ui-zone-sidebar-surface-background))'
    )
  })

  it('offers background tint controls from the existing zone context menu', async () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })

    render(<TreeGroup node={terminalGroup(false)} parentAxis="column" />)
    fireEvent.contextMenu(globalThis.document.querySelector('[data-zone-tabstrip="terminal-zone"]')!)

    expect(await screen.findByText('Background tint')).toBeTruthy()
  })

  it('offers background tint controls from a headerless zone in layout edit mode', async () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    $layoutEditMode.set(true)

    render(<TreeGroup node={{ ...terminalGroup(false), tabStrip: 'never' }} parentAxis="column" />)
    fireEvent.contextMenu(globalThis.document.querySelector('[data-zone-edit-veil="terminal-zone"]')!)

    expect(await screen.findByText('Background tint')).toBeTruthy()
  })

  it('provides keyboard-operable zone actions and native tint menu items in layout edit mode', async () => {
    disposePane = registry.register({
      area: 'panes',
      data: { height: '12rem' },
      id: 'terminal',
      render: () => <div>Terminal</div>,
      title: 'Terminal'
    })
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    $layoutEditMode.set(true)

    render(<TreeGroup node={{ ...terminalGroup(false), tabStrip: 'never' }} parentAxis="column" />)
    fireEvent.click(screen.getByRole('button', { name: 'Zone actions' }))
    const tintTrigger = (await screen.findByText('Background tint')).closest('[role="menuitem"]')

    fireEvent.click(tintTrigger!)

    expect(await screen.findByRole('menuitemradio', { name: 'Red' })).toBeTruthy()
    expect(screen.getByRole('menuitemradio', { name: 'Default background' })).toBeTruthy()
  })
})
