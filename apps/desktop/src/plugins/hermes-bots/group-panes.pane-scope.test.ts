import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupPanes from './group-panes'
import { pluginSdkMock } from './group-test-utils'

// A fronted room must keep owning the workspace scope even when the SELECTED
// room atom has been cleared.
//
// `$groupChatWorkspace` is the selection, and several paths clear it while a
// room's tab is still open and fronted — opening a bot clears it before it
// knows whether the dismiss succeeded, and it is in-memory so any remount
// starts at null. With it null while the room's tab was still on screen, the
// ~5s roster poll republished the workspace scope from the persisted selected
// bot, and ⌘T then created a chat on THAT bot instead of refusing with "New
// group conversations start in the group composer."
//
// In the reported case the persisted selection still named a specific bot, so
// the chat landed on that bot and was created hidden.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => pluginSdkMock(host))

interface Loaded {
  chat: typeof groupChat
  /** Mark a room's tab as the fronted one, modelling what the SDK actually
   *  does: `openWorkspace` takes the bare key and prefixes it itself.
   *
   *  It takes the SLUG from the module — a hand-rolled slug here silently
   *  disagreed with `labels.slugify` on a name ending in punctuation — but
   *  applies the prefix independently. That asymmetry is deliberate: if the
   *  fixture derived the whole id from `groupChatPaneId`, a regression to the
   *  bare key would make lookup and fixture agree on the wrong string and the
   *  behavioural tests would keep passing. */
  front: (group: null | string) => void
  panes: typeof groupPanes
  queried: string[]
}

async function load(): Promise<Loaded> {
  vi.resetModules()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  const queried: string[] = []
  let visible: null | string = null

  host.paneVisibility = (id: string) => {
    queried.push(id)

    return { get: () => visible !== null && id === visible }
  }

  const panes = await import('./group-panes')
  const chat = await import('./group-chat')

  const front = (group: null | string) => {
    visible = group === null ? null : `plugin-workspace:${panes.groupChatWorkspaceKey(group)}`
  }

  return { chat, front, panes, queried }
}

describe('activeGroupChat', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('keeps a fronted room owning the center when the selection atom is null', async () => {
    const { chat, front, panes } = await load()

    panes.recordGroupMainTab('Alpha, Beta, Gamma', () => {})
    front('Alpha, Beta, Gamma')
    chat.$groupChatWorkspace.set(null)

    // The exact live failure: selection cleared, tab still on screen.
    expect(panes.activeGroupChat()).toBe('Alpha, Beta, Gamma')
  })

  it('lets the selection win when it is set', async () => {
    const { chat, panes } = await load()

    chat.$groupChatWorkspace.set('Weekend Room')

    expect(panes.activeGroupChat()).toBe('Weekend Room')
  })

  it('does not let an open but BACKGROUNDED room own the center', async () => {
    const { chat, panes } = await load()

    // A room whose tab exists behind a bot chat must not block a bot scope, or
    // opening a bot after ever visiting a room would stop working.
    panes.recordGroupMainTab('Planning Room', () => {})
    chat.$groupChatWorkspace.set(null)

    expect(panes.activeGroupChat()).toBeNull()
  })

  it('picks the fronted room out of several open ones', async () => {
    const { chat, front, panes } = await load()

    for (const group of ['Weekend Room', 'Alpha, Beta, Gamma', 'Planning Room']) {
      panes.recordGroupMainTab(group, () => {})
    }

    front('Alpha, Beta, Gamma')

    chat.$groupChatWorkspace.set(null)

    expect(panes.activeGroupChat()).toBe('Alpha, Beta, Gamma')
  })

  it('stops owning the center once the room drops its tab', async () => {
    const { chat, front, panes } = await load()

    panes.recordGroupMainTab('Planning Room', () => {})
    front('Planning Room')
    chat.$groupChatWorkspace.set(null)
    expect(panes.activeGroupChat()).toBe('Planning Room')

    panes.dropGroupMainTab('Planning Room')
    expect(panes.activeGroupChat()).toBeNull()
  })

  it('degrades to the selection on a shell with no paneVisibility', async () => {
    vi.resetModules()

    for (const key of Object.keys(host)) {
      delete host[key]
    }

    const panes = await import('./group-panes')
    const chat = await import('./group-chat')

    panes.recordGroupMainTab('Weekend Room', () => {})
    chat.$groupChatWorkspace.set(null)

    expect(panes.frontedGroupChat()).toBeNull()
    expect(panes.activeGroupChat()).toBeNull()
  })
})

describe('the two pane identifiers', () => {
  it('asks paneVisibility for the PREFIXED id, never the openWorkspace key', async () => {
    const { chat, panes, queried } = await load()

    panes.recordGroupMainTab('Alpha, Beta, Gamma', () => {})
    chat.$groupChatWorkspace.set(null)
    panes.frontedGroupChat()

    // The regression this guards: openWorkspace takes the bare key and builds
    // `plugin-workspace:${key}` itself, while paneVisibility passes its
    // argument straight through. Asking for the bare key matches nothing,
    // forever, and the call still succeeds — it just reports "not visible".
    expect(queried).toEqual(['plugin-workspace:hermes-bots:group:alpha-beta-gamma'])

    for (const id of queried) {
      expect(id.startsWith('plugin-workspace:')).toBe(true)
    }
  })

  it('differ by exactly that prefix', async () => {
    const { panes } = await load()

    expect(panes.groupChatPaneId('Alpha, Beta, Gamma')).toBe(
      `plugin-workspace:${panes.groupChatWorkspaceKey('Alpha, Beta, Gamma')}`
    )
  })
})
