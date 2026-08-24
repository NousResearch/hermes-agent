import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function between(start, end) {
  const from = source.indexOf(start)
  const to = source.indexOf(end, from)

  assert.notEqual(from, -1, `missing ${start}`)
  assert.notEqual(to, -1, `missing ${end}`)

  return source.slice(from, to)
}

function load() {
  const context = { Map, Object }
  const roomKey = between('function groupChatRoomKey(', '/** Lift any historical projection shape')
  const drafts = between('const groupComposerDrafts = new Map()', 'function GroupChatWorkspace(')

  vm.runInNewContext(
    `${roomKey}\n${drafts}\nglobalThis.drafts = {
      clearGroupComposerDraft,
      groupComposerDraftKey,
      groupComposerDraftSnapshot,
      migrateGroupComposerDraft,
      restoreGroupComposerDraft,
      updateGroupComposerDraft
    }`,
    context
  )

  return context.drafts
}

test('workspace retirement and re-registration restore the exact room draft', () => {
  const drafts = load()
  const key = drafts.groupComposerDraftKey('Launch room', { roomId: 'room-1' })
  const attachment = { data: 'data:image/png;base64,abc', kind: 'image', name: 'plan.png' }

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

  assert.equal(remounted.main, 'main draft')
  assert.equal(remounted.replies['thread-1'], 'reply draft')
  assert.equal(remounted.activeReplyThread, 'thread-1')
  assert.equal(remounted.pendingAttachments.main[0].name, 'plan.png')
})

test('legacy name-keyed drafts migrate when an immutable room id appears', () => {
  const drafts = load()
  const legacy = drafts.groupComposerDraftKey('Launch room', {})
  const current = drafts.groupComposerDraftKey('Renamed room', { roomId: 'room-1' })

  drafts.updateGroupComposerDraft(legacy, state => ({ ...state, main: 'keep me' }))
  drafts.migrateGroupComposerDraft(legacy, current)

  assert.equal(drafts.groupComposerDraftSnapshot(current).main, 'keep me')
  assert.equal(drafts.groupComposerDraftSnapshot(legacy).main, '')
})

test('a failed send cannot overwrite text entered after the optimistic clear', () => {
  const drafts = load()
  const key = 'id:room-1'

  drafts.updateGroupComposerDraft(key, state => ({ ...state, main: 'send this' }))
  const before = drafts.groupComposerDraftSnapshot(key)
  const cleared = drafts.updateGroupComposerDraft(key, state => ({ ...state, main: '' }))

  drafts.updateGroupComposerDraft(key, state => ({ ...state, main: 'newer typing' }))

  assert.equal(drafts.restoreGroupComposerDraft(key, cleared.revision, before), null)
  assert.equal(drafts.groupComposerDraftSnapshot(key).main, 'newer typing')
})

test('disband removes only that room draft', () => {
  const drafts = load()

  drafts.updateGroupComposerDraft('id:a', state => ({ ...state, main: 'a' }))
  drafts.updateGroupComposerDraft('id:b', state => ({ ...state, main: 'b' }))
  drafts.clearGroupComposerDraft('id:a')

  assert.equal(drafts.groupComposerDraftSnapshot('id:a').main, '')
  assert.equal(drafts.groupComposerDraftSnapshot('id:b').main, 'b')
})

test('GroupChatWorkspace owns composer state through the room draft store', () => {
  const workspace = between('function GroupChatWorkspace(', '/** Live closers for group-chat MAIN-window tabs')

  assert.match(workspace, /groupComposerDraftKey\(group, room\)/)
  assert.match(workspace, /restoreGroupComposerDraft\(composerKeyRef\.current, cleared\.revision, before\)/)
  assert.match(workspace, /clearGroupComposerDraft\(composerKeyRef\.current\)/)
  assert.doesNotMatch(workspace, /useState\(''\).*?draft/)
  assert.doesNotMatch(workspace, /useState\(\{\}\).*?replyDrafts/)
})
