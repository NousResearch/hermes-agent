import { afterEach, describe, expect, it } from 'vitest'

import {
  $composerAttachments,
  addComposerAttachment,
  clearPersistedComposerDraft,
  clearStashedComposerAttachments,
  type ComposerAttachment,
  composerDraftStorageKey,
  readPersistedComposerDraft,
  removeComposerAttachment,
  stashComposerAttachments,
  takeComposerAttachments,
  updateComposerAttachment,
  writePersistedComposerDraft
} from './composer'

function attachment(overrides: Partial<ComposerAttachment> & Pick<ComposerAttachment, 'id'>): ComposerAttachment {
  return { kind: 'file', label: 'doc.pdf', ...overrides }
}

describe('updateComposerAttachment', () => {
  afterEach(() => {
    $composerAttachments.set([])
  })

  it('replaces an existing attachment in place', () => {
    addComposerAttachment(attachment({ id: 'file:a', uploadState: 'uploading' }))

    const updated = updateComposerAttachment(attachment({ id: 'file:a', attachedSessionId: 'sess-1' }))

    expect(updated).toBe(true)
    const current = $composerAttachments.get()
    expect(current).toHaveLength(1)
    expect(current[0]?.attachedSessionId).toBe('sess-1')
    expect(current[0]?.uploadState).toBeUndefined()
  })

  it('does NOT resurrect an attachment the user removed mid-upload', () => {
    // Drop → eager upload starts → user removes the chip → upload resolves.
    // The late success must not re-add the removed attachment.
    addComposerAttachment(attachment({ id: 'file:a', uploadState: 'uploading' }))
    removeComposerAttachment('file:a')

    const updated = updateComposerAttachment(attachment({ id: 'file:a', attachedSessionId: 'sess-1' }))

    expect(updated).toBe(false)
    expect($composerAttachments.get()).toHaveLength(0)
  })
})

describe('persisted composer drafts', () => {
  afterEach(() => {
    window.localStorage.clear()
  })

  it('stores and restores text drafts per session scope', () => {
    writePersistedComposerDraft('session-a', 'almost submitted prompt')
    writePersistedComposerDraft('session-b', 'other draft')

    expect(readPersistedComposerDraft('session-a')).toBe('almost submitted prompt')
    expect(readPersistedComposerDraft('session-b')).toBe('other draft')
  })

  it('uses a stable new-session key when no session id exists yet', () => {
    writePersistedComposerDraft(null, 'first prompt draft')

    expect(window.localStorage.getItem(composerDraftStorageKey(null))).toBe('first prompt draft')
    expect(readPersistedComposerDraft(undefined)).toBe('first prompt draft')
  })

  it('removes empty drafts instead of leaving stale text behind', () => {
    writePersistedComposerDraft('session-a', 'saved')
    writePersistedComposerDraft('session-a', '')

    expect(readPersistedComposerDraft('session-a')).toBe('')
    expect(window.localStorage.getItem(composerDraftStorageKey('session-a'))).toBeNull()
  })

  it('can explicitly clear a saved draft after submit', () => {
    writePersistedComposerDraft('session-a', 'saved')
    clearPersistedComposerDraft('session-a')

    expect(readPersistedComposerDraft('session-a')).toBe('')
  })
})

describe('stashed composer attachments', () => {
  afterEach(() => {
    clearStashedComposerAttachments('session-a')
    clearStashedComposerAttachments('session-b')
    clearStashedComposerAttachments(null)
  })

  it('retains and restores attachments per session scope', () => {
    stashComposerAttachments('session-a', [attachment({ id: 'file:a' })])
    stashComposerAttachments('session-b', [attachment({ id: 'image:b', kind: 'image' })])

    expect(takeComposerAttachments('session-a').map(a => a.id)).toEqual(['file:a'])
    expect(takeComposerAttachments('session-b').map(a => a.id)).toEqual(['image:b'])
  })

  it('shares a stable new-session scope with the text draft helpers', () => {
    stashComposerAttachments(null, [attachment({ id: 'file:new' })])

    expect(takeComposerAttachments(undefined).map(a => a.id)).toEqual(['file:new'])
  })

  it('returns cloned attachments so callers cannot mutate the stash', () => {
    stashComposerAttachments('session-a', [attachment({ id: 'file:a', label: 'orig.pdf' })])

    const taken = takeComposerAttachments('session-a')
    taken[0]!.label = 'mutated.pdf'

    expect(takeComposerAttachments('session-a')[0]?.label).toBe('orig.pdf')
  })

  it('drops the scope entry when stashing an empty set', () => {
    stashComposerAttachments('session-a', [attachment({ id: 'file:a' })])
    stashComposerAttachments('session-a', [])

    expect(takeComposerAttachments('session-a')).toEqual([])
  })

  it('clears a scope explicitly after an accepted submit', () => {
    stashComposerAttachments('session-a', [attachment({ id: 'file:a' })])
    clearStashedComposerAttachments('session-a')

    expect(takeComposerAttachments('session-a')).toEqual([])
  })
})
