import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { findGroupOfPane, group, split, type LayoutNode } from '../model'
import {
  $layoutTree,
  adoptContributedPanes,
  isPaneVisible,
  removeTreePane,
  revealTreePane
} from '../store'

interface Zone {
  active: string | undefined
  id: string
  panes: readonly string[]
}

const visibleZones = (tree: LayoutNode): Zone[] => {
  const zones: Zone[] = []

  const walk = (node: LayoutNode): void => {
    if (node.type === 'group') {
      const shown = node.panes.filter(id => isPaneVisible(id))

      if (shown.length > 0) {
        zones.push({ active: shown.includes(node.active ?? '') ? node.active : shown[0], id: node.id, panes: shown })
      }

      return
    }

    node.children.forEach(walk)
  }

  walk(tree)

  return zones
}

describe('tile-less bot chat split drop', () => {
  let disposePane: (() => void) | null = null

  beforeEach(() => {
    $layoutTree.set(null)
    disposePane = null
  })

  afterEach(() => {
    disposePane?.()
    removeTreePane('session-tile:bot-chat')
    $layoutTree.set(null)
  })

  it('adopts a revealed tile pane and fronts it in a real split zone', () => {
    // The tree a Bot Mode main chat + a ⌃T side thread produces: one zone,
    // two tabs. The bot chat rides the uncloseable `workspace` pane itself —
    // it has NO tile pane yet, which is what a drag out of MAIN has to fix.
    $layoutTree.set(
      group(['workspace', 'session-tile:side-thread'], {
        active: 'workspace',
        id: 'grp-main'
      })
    )

    // What openSessionTile mints when the remembered bot-chat scope rescues
    // the drop: a real tile contribution with a `dock` hint naming the
    // anchor + edge. The registry/adoption pass — the same machinery
    // watchSessionTiles drives — turns the contribution into a pane.
    disposePane = registry.register({
      area: 'panes',
      data: {
        dock: { pane: 'workspace', pos: 'right' },
        minWidth: '20rem',
        placement: 'main'
      },
      id: 'session-tile:bot-chat',
      render: () => null,
      title: 'Alpha'
    })

    // The commit path's reveal: un-dismiss + adopt + front.
    revealTreePane('session-tile:bot-chat')

    const after = visibleZones($layoutTree.get()!)

    // BOTH surfaces stay on screen: the workspace (still holding the side
    // thread) and the new tile pane — and the drop fronted the tile.
    expect(after.length).toBeGreaterThanOrEqual(2)

    const chat = after.find(zone => zone.panes.includes('session-tile:bot-chat'))

    expect(chat).toBeDefined()
    expect(chat!.active).toBe('session-tile:bot-chat')

    // The dragged tab lands in its own split, not stacked into the source
    // strip (a stack would leave the side thread without a visible pane).
    expect(chat!.panes).toEqual(['session-tile:bot-chat'])

    // The side thread stays stacked in the workspace zone (mounted behind
    // its active tab) — the split ADDED a pane, it never dropped one.
    const workspaceGroup = findGroupOfPane($layoutTree.get()!, 'workspace')

    expect(workspaceGroup?.panes).toContain('session-tile:side-thread')
  })

  it('leaves no empty zone behind after the bot chat moves out of MAIN', () => {
    $layoutTree.set(group(['workspace', 'session-tile:side-thread'], { active: 'workspace', id: 'grp-main' }))

    disposePane = registry.register({
      area: 'panes',
      data: { dock: { pane: 'workspace', pos: 'right' }, placement: 'main' },
      id: 'session-tile:bot-chat',
      render: () => null,
      title: 'Alpha'
    })

    revealTreePane('session-tile:bot-chat')

    const tree = $layoutTree.get()!

    // Every zone on screen keeps a visible tab; nothing is stranded with
    // zero tabs and no content.
    const leaves: LayoutNode[] = []

    const walk = (node: LayoutNode): void => {
      if (node.type === 'group') {
        if (node.panes.length === 0) {
          leaves.push(node)
        }

        return
      }

      node.children.forEach(walk)
    }

    walk(tree)

    expect(leaves).toEqual([])

    const workspaceGroup = findGroupOfPane(tree, 'workspace')

    // The workspace never leaves the tree (it is uncloseable): the side
    // thread keeps it occupied even though the chat moved out.
    expect(workspaceGroup?.panes).toContain('session-tile:side-thread')
  })

  it('splits beside the dropped edge without displacing the sibling zone', () => {
    $layoutTree.set(
      split('row', [group(['workspace', 'session-tile:side-thread'], { active: 'workspace', id: 'grp-main' })])
    )

    disposePane = registry.register({
      area: 'panes',
      data: { dock: { pane: 'workspace', pos: 'right' }, placement: 'main' },
      id: 'session-tile:bot-chat',
      render: () => null,
      title: 'Alpha'
    })

    adoptContributedPanes()
    revealTreePane('session-tile:bot-chat')

    const tree = $layoutTree.get()!
    const chatGroup = findGroupOfPane(tree, 'session-tile:bot-chat')
    const mainGroup = findGroupOfPane(tree, 'workspace')

    expect(chatGroup).toBeDefined()
    expect(mainGroup).toBeDefined()
    expect(chatGroup!.id).not.toBe(mainGroup!.id)
    expect(mainGroup!.panes).toContain('session-tile:side-thread')
  })
})
