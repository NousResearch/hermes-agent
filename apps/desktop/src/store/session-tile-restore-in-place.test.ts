import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LayoutNode } from '@/components/pane-shell/tree/model'

// Cold start must bring every session tab back WHERE IT WAS. Session tiles are
// stored per gateway-profile bucket, but the layout tree that says where each
// tab sits is shared. Boot starts on the placeholder `default` scope, so the
// tile mirror prunes every persisted tile pane the placeholder bucket doesn't
// list, then re-docks the real bucket's tiles from each tile's stored
// anchor/before/dir. Those hints are written when a tab is opened or dragged
// through openSessionTile and go stale; a bucket that wasn't on screen at quit
// is staler still. A restart therefore moved tabs between zones without the
// user touching anything: the active right-zone chat landed in the main zone,
// and two main-zone chats landed in the right zone.

const pane = (id: string) => `session-tile:${id}`

// Main zone: the workspace, a Bot tab, then three session tabs.
// Right zone: two Bot tabs, a session tab (the one being typed in), a Bot tab.
const persistedTree: LayoutNode = {
  type: 'split',
  id: 'root',
  orientation: 'row',
  children: [
    {
      type: 'group',
      id: 'g-main',
      panes: ['workspace', pane('bot-dropbox'), pane('gaia'), pane('signal'), pane('c1')],
      active: 'workspace'
    },
    {
      type: 'group',
      id: 'g-right',
      panes: [pane('bot-alibaba'), pane('bot-bob'), pane('hermes-pr'), pane('bot-insta')],
      active: pane('hermes-pr')
    }
  ],
  weights: [1, 1]
}

const sessionTile = (storedSessionId: string, anchor: string, before?: string) => ({
  anchor,
  ...(before ? { before } : {}),
  dir: 'center',
  storedSessionId,
  workspaceMode: 'sessions'
})

const botTile = (storedSessionId: string, anchor: string) => ({
  anchor,
  dir: 'center',
  storedSessionId,
  workspaceMode: 'bots',
  workspaceOwnerKey: 'bot:local::chief'
})

// Each bucket's anchors are whatever they were when that bucket last wrote
// them — not where the panes sit in the tree above.
const buckets = {
  // On screen at quit. hermes-pr was opened beside the workspace and later
  // moved into the right zone; gaia/signal/c1 were stacked with the workspace.
  chief: [
    sessionTile('gaia', 'workspace'),
    sessionTile('signal', 'workspace'),
    sessionTile('c1', 'workspace'),
    sessionTile('hermes-pr', 'workspace')
  ],
  // The window's primary profile; not on screen at quit. Its hints for the
  // same chats are from an older arrangement.
  noir: [
    sessionTile('php', 'session-tile:gone'),
    sessionTile('hermes-pr', 'workspace'),
    sessionTile('gaia', pane('bot-alibaba'), pane('bot-alibaba')),
    sessionTile('signal', pane('gaia'), pane('bot-alibaba'))
  ],
  __bots_workspace__: [
    botTile('bot-dropbox', 'workspace'),
    botTile('bot-alibaba', 'workspace'),
    botTile('bot-bob', pane('bot-alibaba')),
    botTile('bot-insta', pane('bot-alibaba'))
  ]
}

async function coldStart() {
  window.localStorage.setItem('hermes.desktop.layoutTree.v2', JSON.stringify(persistedTree))
  window.localStorage.setItem('hermes.desktop.sessionTiles.v2', JSON.stringify(buckets))
  vi.resetModules()

  const tree = await import('@/components/pane-shell/tree/store')
  const model = await import('@/components/pane-shell/tree/model')
  const { registry } = await import('@/contrib/registry')
  const states = await import('@/store/session-states')
  const { $activeGatewayProfile } = await import('@/store/profile')
  const { watchSessionTiles } = await import('@/app/chat/session-tile')

  // Boot order (contrib/controller.tsx): hydrate the tree, then watch tiles.
  registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'chat'
  })
  tree.hydrateContributedPanes()
  watchSessionTiles()

  const zoneOf = (storedSessionId: string) => model.findGroupOfPane(tree.$layoutTree.get()!, pane(storedSessionId))

  const tabs = (groupId: string) =>
    (model.findGroup(tree.$layoutTree.get()!, groupId)?.panes ?? []).map(id => id.replace('session-tile:', ''))

  return { $activeGatewayProfile, states, tabs, tree, zoneOf }
}

/** `ids` in the order they appear in `tabs`, skipping tabs not listed. */
const relativeOrder = (tabs: string[], ids: string[]) => tabs.filter(id => ids.includes(id))

describe('cold start keeps every session tab where it was', () => {
  // The first import of the tile surface transforms a large module graph; do
  // it once up front so no single test pays for it against its timeout.
  beforeAll(async () => {
    await import('@/app/chat/session-tile')
  }, 120_000)

  beforeEach(() => {
    window.localStorage.clear()
  })

  it('a restart on the same profile restores every tab to its zone and slot', async () => {
    const { $activeGatewayProfile, tabs } = await coldStart()

    // Boot adopts the profile that was on screen at quit.
    $activeGatewayProfile.set('chief')

    expect(tabs('g-main')).toEqual(['workspace', 'bot-dropbox', 'gaia', 'signal', 'c1'])
    expect(tabs('g-right')).toEqual(['bot-alibaba', 'bot-bob', 'hermes-pr', 'bot-insta'])
  })

  it('a restart that lands on another profile does not move the chats both profiles have open', async () => {
    const { $activeGatewayProfile, tabs, zoneOf } = await coldStart()

    // Boot adopts the window's primary profile, whose bucket was not on screen.
    $activeGatewayProfile.set('noir')

    // The chat being typed in stays in the right zone, in its slot.
    expect(zoneOf('hermes-pr')?.id).toBe('g-right')
    expect(tabs('g-right')).toEqual(['bot-alibaba', 'bot-bob', 'hermes-pr', 'bot-insta'])

    // gaia / signal stay in the main zone, in order, after the Bot tab.
    expect(zoneOf('gaia')?.id).toBe('g-main')
    expect(zoneOf('signal')?.id).toBe('g-main')
    expect(relativeOrder(tabs('g-main'), ['workspace', 'bot-dropbox', 'gaia', 'signal'])).toEqual([
      'workspace',
      'bot-dropbox',
      'gaia',
      'signal'
    ])
  })

  it('switching profiles away and back returns each tab to the slot it left', async () => {
    const { $activeGatewayProfile, tabs } = await coldStart()
    $activeGatewayProfile.set('chief')

    // c1 is only in `chief`; it leaves the tree while `noir` is on screen.
    $activeGatewayProfile.set('noir')
    $activeGatewayProfile.set('chief')

    expect(tabs('g-main')).toEqual(['workspace', 'bot-dropbox', 'gaia', 'signal', 'c1'])
    expect(tabs('g-right')).toEqual(['bot-alibaba', 'bot-bob', 'hermes-pr', 'bot-insta'])
  })
})
