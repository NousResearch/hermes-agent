import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GroupChatRoom } from './group-chat'
import type { Attachment } from './types'

// group-panes is a leaf over the room store, and the room store reaches the
// gateway through `host`. Nothing here talks to a gateway, so the mock only
// has to satisfy the import graph.
const { host } = vi.hoisted(() => ({
  host: {
    request: vi.fn(async () => ({})),
    requestProfile: vi.fn(async () => ({})),
    state: { connectionId: { get: () => 'local', listen: () => () => undefined } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const nanostores = await import('nanostores')

  return {
    atom: nanostores.atom,
    computed: nanostores.computed,
    host,
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: () => ({ data: [], isLoading: false }),
    useValue: <T>(store: { get: () => T }) => store.get()
  }
})

// The draft map is module-scope, which is the whole point of it (it has to
// outlive a pane), so every test needs a fresh module.
async function loadDrafts() {
  vi.resetModules()

  return import('./group-panes')
}

/** A room record from just the field a draft key is derived from. */
const room = (partial: Partial<GroupChatRoom>): GroupChatRoom => ({ log: [], watermarks: {}, ...partial })

beforeEach(() => {
  vi.clearAllMocks()
})

describe('group composer drafts', () => {
  it('workspace retirement and re-registration restore the exact room draft', async () => {
    const drafts = await loadDrafts()
    const key = drafts.groupComposerDraftKey('Launch room', room({ roomId: 'room-1' }))
    const attachment: Attachment = { data: 'data:image/png;base64,abc', kind: 'image', name: 'plan.png' }

    drafts.updateGroupComposerDraft(key, state => ({
      ...state,
      activeReplyThread: 'thread-1',
      main: 'main draft',
      pendingAttachments: { main: [attachment], 'thread-1': [attachment] },
      replies: { 'thread-1': 'reply draft' }
    }))

    // Dropping the component reference simulates pane retirement. A fresh
    // registration reads the same module-scope, roomId-qualified snapshot.
    const remounted = drafts.groupComposerDraftSnapshot(key)

    expect(remounted.main).toBe('main draft')
    expect(remounted.replies['thread-1']).toBe('reply draft')
    expect(remounted.activeReplyThread).toBe('thread-1')
    expect(remounted.pendingAttachments.main[0].name).toBe('plan.png')
  })

  it('legacy name-keyed drafts migrate when an immutable room id appears', async () => {
    const drafts = await loadDrafts()
    const legacy = drafts.groupComposerDraftKey('Launch room', room({}))
    const current = drafts.groupComposerDraftKey('Renamed room', room({ roomId: 'room-1' }))

    drafts.updateGroupComposerDraft(legacy, state => ({ ...state, main: 'keep me' }))
    drafts.migrateGroupComposerDraft(legacy, current)

    expect(drafts.groupComposerDraftSnapshot(current).main).toBe('keep me')
    expect(drafts.groupComposerDraftSnapshot(legacy).main).toBe('')
  })

  it('a failed send cannot overwrite text entered after the optimistic clear', async () => {
    const drafts = await loadDrafts()
    const key = 'id:room-1'

    drafts.updateGroupComposerDraft(key, state => ({ ...state, main: 'send this' }))
    const before = drafts.groupComposerDraftSnapshot(key)
    const cleared = drafts.updateGroupComposerDraft(key, state => ({ ...state, main: '' }))

    drafts.updateGroupComposerDraft(key, state => ({ ...state, main: 'newer typing' }))

    expect(drafts.restoreGroupComposerDraft(key, cleared.revision, before)).toBeNull()
    expect(drafts.groupComposerDraftSnapshot(key).main).toBe('newer typing')
  })

  it('disband removes only that room draft', async () => {
    const drafts = await loadDrafts()

    drafts.updateGroupComposerDraft('id:a', state => ({ ...state, main: 'a' }))
    drafts.updateGroupComposerDraft('id:b', state => ({ ...state, main: 'b' }))
    drafts.clearGroupComposerDraft('id:a')

    expect(drafts.groupComposerDraftSnapshot('id:a').main).toBe('')
    expect(drafts.groupComposerDraftSnapshot('id:b').main).toBe('b')
  })
})

describe('main-window tab registry', () => {
  // #89788: the in-panel room is a FALLBACK surface. The Map alone can't
  // notify React, so ownership has to move the rev atom too — the first fix
  // read the Map non-reactively and left two live panes driving one engine.
  it('the in-pane room yields to a main tab, and the gate is reactive', async () => {
    const panes = await loadDrafts()

    expect(panes.shouldRenderGroupChatInPane('Launch room')).toBe(true)

    const rev = panes.$groupMainTabsRev.get()
    panes.recordGroupMainTab('Launch room', () => undefined)

    expect(panes.shouldRenderGroupChatInPane('Launch room')).toBe(false)
    expect(panes.$groupMainTabsRev.get()).toBeGreaterThan(rev)

    panes.dropGroupMainTab('Launch room')

    expect(panes.shouldRenderGroupChatInPane('Launch room')).toBe(true)
  })

  it('closing the tab runs its closer and releases the workspace', async () => {
    vi.resetModules()
    const [panes, chat] = await Promise.all([import('./group-panes'), import('./group-chat')])
    const close = vi.fn()

    chat.$groupChatWorkspace.set('Launch room')
    panes.recordGroupMainTab('Launch room', close)
    panes.closeGroupChatMainTab('Launch room')

    expect(close).toHaveBeenCalledOnce()
    expect(chat.$groupChatWorkspace.get()).toBeNull()
    expect(panes.groupChatMainTabs.has('Launch room')).toBe(false)
  })
})
