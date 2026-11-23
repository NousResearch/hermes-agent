import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import type { ClientSessionState } from '@/app/types'
import * as model from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { clearSessionDraft, freshDraftScope, onComposerDraftSyncRequest, stashSessionDraft } from '@/store/composer'
import * as session from '@/store/session'
import * as states from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { $workspaceIsPage } from '../routes'

import { frontUntouchedDraftTab } from './reuse-draft-tab'

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
    dir: () => 'center',
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()

  return disposeWorkspace
})

const TILE_IDS = ['draft-a', 'used-b', 'bot-c']

function idleState(messages = 0): ClientSessionState {
  return { busy: false, messages: Array.from({ length: messages }, (_, i) => ({ id: `m${i}` })) } as ClientSessionState
}

function openTile(id: string, runtimeId: string, messages = 0, workspaceMode: 'bots' | 'sessions' = 'sessions') {
  states.openSessionTile(id, 'center', 'workspace', undefined, { workspaceMode })
  states.patchSessionTile(id, { runtimeId })
  states.$sessionStates.set({ ...states.$sessionStates.get(), [runtimeId]: idleState(messages) })
  // The create path fronts what it opened; the user is now looking at this tab.
  states.focusOpenSession(id, { workspaceMode })
}

const activePane = () => model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')?.active

function reset() {
  TILE_IDS.forEach(id => states.discardSessionTile(id))
  TILE_IDS.forEach(id => clearSessionDraft(id))
  states.$sessionStates.set({})
  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  session.$selectedStoredSessionId.set(null)
  session.$activeSessionId.set(null)
  session.setSessions([])
  clearSessionDraft(freshDraftScope())
  $workspaceIsPage.set(false)
  window.localStorage.clear()
}

describe('the tab-strip "+" reuses an untouched New session draft (#145594)', () => {
  beforeEach(() => {
    reset()
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
  })

  afterEach(reset)

  it('fronts the blank draft tab instead of asking for another one', () => {
    session.$selectedStoredSessionId.set('loaded-chat')
    openTile('draft-a', 'rt-a')
    openTile('used-b', 'rt-b', 2)
    expect(activePane()).toBe('session-tile:used-b')

    expect(frontUntouchedDraftTab()).toBe(true)
    expect(activePane()).toBe('session-tile:draft-a')
    expect(states.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['draft-a', 'used-b'])
  })

  it('fronts main when main is the blank new chat', () => {
    openTile('used-b', 'rt-b', 2)
    expect(activePane()).toBe('session-tile:used-b')

    expect(frontUntouchedDraftTab()).toBe(true)
    expect(activePane()).toBe('workspace')
  })

  it('still opens a new tab once every draft has been used', () => {
    session.$selectedStoredSessionId.set('loaded-chat')
    openTile('draft-a', 'rt-a')
    stashSessionDraft('draft-a', 'half-typed question', [])
    openTile('used-b', 'rt-b', 1)
    openTile('bot-c', 'rt-c', 0, 'bots')

    expect(frontUntouchedDraftTab()).toBe(false)

    clearSessionDraft('draft-a')
    session.setSessions([{ id: 'draft-a' } as SessionInfo])
    expect(frontUntouchedDraftTab()).toBe(false)
  })

  it('counts text still in the editor, not yet stashed by the debounce', () => {
    session.$selectedStoredSessionId.set('loaded-chat')
    openTile('draft-a', 'rt-a')
    openTile('used-b', 'rt-b', 1)

    const stop = onComposerDraftSyncRequest(({ mode, target }) => {
      if (mode === 'flush' && target === 'tile:draft-a') {
        stashSessionDraft('draft-a', 'just typed', [])
      }
    })

    try {
      expect(frontUntouchedDraftTab()).toBe(false)
    } finally {
      stop()
    }
  })

  it('does not treat a typed main draft or a page as blank', () => {
    stashSessionDraft(freshDraftScope(), 'draft text', [])
    expect(frontUntouchedDraftTab()).toBe(false)

    clearSessionDraft(freshDraftScope())
    $workspaceIsPage.set(true)
    expect(frontUntouchedDraftTab()).toBe(false)
  })

  it('does not reuse a tab whose session state is not known yet', () => {
    session.$selectedStoredSessionId.set('loaded-chat')
    states.openSessionTile('draft-a', 'center', 'workspace', undefined, { workspaceMode: 'sessions' })

    expect(frontUntouchedDraftTab()).toBe(false)
  })
})
