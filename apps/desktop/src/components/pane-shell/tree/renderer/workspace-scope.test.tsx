import { act, type ReactNode } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { registry } from '@/contrib/registry'

import type { GroupNode } from '../model'

import { TreeGroup } from './tree-group'

let root: null | Root = null
let container: HTMLDivElement | null = null
const disposers: Array<() => void> = []

function render(ui: ReactNode) {
  if (!container) {
    container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    root = createRoot(container)
  }

  act(() => root!.render(ui))
}

function register(
  id: string,
  title: string,
  scope: { workspaceMode?: 'sessions' | 'bots'; workspaceOwnerKey?: string } = {}
) {
  disposers.push(
    registry.register({
      area: 'panes',
      data: { placement: 'main' },
      id,
      render: () => <div>{title} content</div>,
      title,
      ...scope
    })
  )
}

const group = (active: string): GroupNode => ({
  active,
  id: 'workspace-scope-zone',
  panes: ['session-a', 'bot-a', 'bot-b'],
  tabStrip: 'always',
  type: 'group'
})

const visibleTabs = () =>
  [...globalThis.document.querySelectorAll<HTMLElement>('[data-tree-tab]')].map(tab => tab.dataset.treeTab)

afterEach(() => {
  if (root) {
    act(() => root!.unmount())
  }

  container?.remove()
  disposers.splice(0).forEach(dispose => dispose())
  act(() => {
    setWorkspaceScope('sessions')
  })
  root = null
  container = null
  vi.unstubAllGlobals()
})

describe('TreeGroup workspace scope', () => {
  it('renders only the current workspace owner and restores owner activity', () => {
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    register('session-a', 'Session A', { workspaceMode: 'sessions' })
    register('bot-a', 'Bot A', { workspaceMode: 'bots', workspaceOwnerKey: 'connection-a::default' })
    register('bot-b', 'Bot B', { workspaceMode: 'bots', workspaceOwnerKey: 'connection-b::default' })

    render(<TreeGroup node={group('session-a')} parentAxis="column" />)
    expect(visibleTabs()).toEqual(['session-a'])
    expect(container?.textContent).toContain('Session A content')

    act(() => setWorkspaceScope('bots', 'connection-a::default'))
    render(<TreeGroup node={group('bot-a')} parentAxis="column" />)
    expect(visibleTabs()).toEqual(['bot-a'])
    expect(container?.textContent).toContain('Bot A content')

    act(() => setWorkspaceScope('bots', 'connection-b::default'))
    render(<TreeGroup node={group('bot-b')} parentAxis="column" />)
    expect(visibleTabs()).toEqual(['bot-b'])
    expect(container?.textContent).toContain('Bot B content')

    act(() => setWorkspaceScope('bots', 'connection-a::default'))
    render(<TreeGroup node={group('bot-b')} parentAxis="column" />)
    expect(visibleTabs()).toEqual(['bot-a'])
    expect(container?.textContent).toContain('Bot A content')

    act(() => setWorkspaceScope('sessions'))
    render(<TreeGroup node={group('bot-b')} parentAxis="column" />)
    expect(visibleTabs()).toEqual(['session-a'])
    expect(container?.textContent).toContain('Session A content')
  })

  it('keeps a global Browser pane visible in Bot Mode', () => {
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    register('bot-a', 'Bot A', { workspaceMode: 'bots', workspaceOwnerKey: 'connection-a::default' })
    register('preview-tile:url:browser', 'Browser')

    const node: GroupNode = {
      active: 'bot-a',
      id: 'workspace-scope-zone',
      panes: ['bot-a', 'preview-tile:url:browser'],
      tabStrip: 'always',
      type: 'group'
    }

    act(() => setWorkspaceScope('bots', 'connection-a::default'))
    render(<TreeGroup node={node} parentAxis="column" />)

    expect(visibleTabs()).toEqual(['bot-a', 'preview-tile:url:browser'])
  })

  it('hides a Sessions-only Browser pane in Bot Mode', () => {
    vi.stubGlobal('CSS', { escape: (value: string) => value })
    register('bot-a', 'Bot A', { workspaceMode: 'bots', workspaceOwnerKey: 'connection-a::default' })
    register('preview-tile:url:browser', 'Browser', { workspaceMode: 'sessions' })

    const node: GroupNode = {
      active: 'bot-a',
      id: 'workspace-scope-zone',
      panes: ['bot-a', 'preview-tile:url:browser'],
      tabStrip: 'always',
      type: 'group'
    }

    act(() => setWorkspaceScope('bots', 'connection-a::default'))
    render(<TreeGroup node={node} parentAxis="column" />)

    expect(visibleTabs()).toEqual(['bot-a'])
    expect(container?.textContent).not.toContain('Browser content')
  })
})
