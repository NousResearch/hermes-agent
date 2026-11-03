import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import type { ClientSessionState } from '@/app/types'
import * as model from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { clearSessionDraft } from '@/store/composer'
import * as session from '@/store/session'
import * as states from '@/store/session-states'
import type { TileDock } from '@/store/session-states'

import { openSession } from '../open-session'
import { $workspaceIsPage } from '../routes'

// A plain click on a session (sidebar row, switcher, picker) belongs to the tab
// the user is in. Routing it into main instead replaced whichever tab happened
// to hold main — a tab the user neither picked nor could identify, and one that
// moves whenever main is closed and a neighbour is promoted into it.

beforeAll(() => {
  const disposeWorkspace = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

  tree.watchContributedPanes()
  paneMirror({
    source: states.$sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: tile => tile.dir ?? 'center',
    anchor: tile => tile.anchor,
    before: tile => tile.before,
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()

  return disposeWorkspace
})

const IDS = ['chat-a', 'chat-b', 'draft-a', 'busy-a', 'target']
const SESSIONS = { workspaceMode: 'sessions' } as const

function state(messages: number, busy = false): ClientSessionState {
  return { busy, messages: Array.from({ length: messages }, (_, i) => ({ id: `m${i}` })) } as ClientSessionState
}

function openTile(
  id: string,
  messages: number,
  { busy = false, dir = 'center' }: { busy?: boolean; dir?: TileDock } = {}
) {
  states.openSessionTile(id, dir, 'workspace', undefined, SESSIONS)
  states.patchSessionTile(id, { runtimeId: `rt-${id}` })
  states.$sessionStates.set({ ...states.$sessionStates.get(), [`rt-${id}`]: state(messages, busy) })
  states.focusOpenSession(id, SESSIONS)
}

const tilePane = (id: string) => `session-tile:${id}`
const groupOf = (paneId: string) => model.findGroupOfPane(tree.$layoutTree.get()!, paneId)
const mainGroup = () => groupOf('workspace')!

function reset() {
  IDS.forEach(id => states.discardSessionTile(id))
  IDS.forEach(id => clearSessionDraft(id))
  states.$sessionStates.set({})
  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  session.$selectedStoredSessionId.set(null)
  session.$activeSessionId.set(null)
  session.setSessions([])
  $workspaceIsPage.set(false)
  window.localStorage.clear()

  // Drain the ⌘⇧T stack so one test's closed tab never reopens in the next.
  for (let i = 0; i < 20; i++) {
    states.reopenLastClosedTile()
  }

  IDS.forEach(id => states.discardSessionTile(id))
}

describe('a plain session click opens in the focused tab', () => {
  const navigate = vi.fn()

  beforeEach(() => {
    reset()
    navigate.mockClear()
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
    session.$selectedStoredSessionId.set('main-chat')
  })

  afterEach(reset)

  it('takes the focused tab’s slot and leaves main alone', () => {
    openTile('chat-a', 3)
    openTile('chat-b', 2)
    states.focusOpenSession('chat-a', SESSIONS)

    openSession('target', navigate)

    expect(navigate).not.toHaveBeenCalled()
    expect(mainGroup().panes).toEqual(['workspace', tilePane('target'), tilePane('chat-b')])
    expect(mainGroup().active).toBe(tilePane('target'))
    expect(session.$selectedStoredSessionId.get()).toBe('main-chat')
  })

  it('gives the replaced chat back with ⌘⇧T, in its old slot', () => {
    openTile('chat-a', 3)

    openSession('target', navigate)
    states.reopenLastClosedTile()

    expect(mainGroup().panes).toEqual(['workspace', tilePane('chat-a'), tilePane('target')])
  })

  it('spends an untouched untitled tab without an undo entry', () => {
    openTile('draft-a', 0)

    openSession('target', navigate)
    expect(mainGroup().panes).toEqual(['workspace', tilePane('target')])

    states.reopenLastClosedTile()
    expect(mainGroup().panes).toEqual(['workspace', tilePane('target')])
  })

  it('keeps a busy chat streaming in the background', () => {
    openTile('busy-a', 1, { busy: true })

    openSession('target', navigate)

    expect(mainGroup().panes).toEqual(['workspace', tilePane('target')])
    expect(states.$sessionStates.get()['rt-busy-a']?.busy).toBe(true)
  })

  it('replaces a lone tab inside its own zone instead of collapsing the zone', () => {
    openTile('chat-a', 3, { dir: 'right' })
    const zone = groupOf(tilePane('chat-a'))!.id
    expect(zone).not.toBe(mainGroup().id)

    openSession('target', navigate)

    expect(navigate).not.toHaveBeenCalled()
    expect(groupOf(tilePane('target'))?.id).toBe(zone)
    expect(groupOf(tilePane('target'))?.panes).toEqual([tilePane('target')])
    expect(mainGroup().panes).toEqual(['workspace'])
  })

  it('still loads into main when main is the focused tab', () => {
    openTile('chat-a', 3)
    states.frontMainIfSelected('main-chat')

    openSession('target', navigate)

    expect(navigate).toHaveBeenCalledWith('/target')
    expect(mainGroup().panes).toEqual(['workspace', tilePane('chat-a')])
  })

  it('fronts a chat already open in another tab instead of duplicating it', () => {
    openTile('chat-b', 2)
    openTile('chat-a', 3)

    openSession('chat-b', navigate)

    expect(navigate).not.toHaveBeenCalled()
    expect(mainGroup().panes).toEqual(['workspace', tilePane('chat-b'), tilePane('chat-a')])
    expect(mainGroup().active).toBe(tilePane('chat-b'))
  })

  it('keeps ⌘-click as a new tab beside the focused one', () => {
    openTile('chat-a', 3)

    openSession('target', navigate, 'tab')

    expect(mainGroup().panes).toEqual(['workspace', tilePane('chat-a'), tilePane('target')])
  })
})
