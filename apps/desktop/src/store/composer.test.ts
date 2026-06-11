import { afterEach, describe, expect, it } from 'vitest'

import {
  $composerAttachments,
  addComposerAttachment,
  clearPersistedComposerDraft,
  COMPOSER_DRAFT_STORAGE_KEY,
  type ComposerAttachment,
  readPersistedComposerDraft,
  removeComposerAttachment,
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

describe('persisted composer draft', () => {
  afterEach(() => {
    window.localStorage.clear()
  })

  it('stores and restores the draft', () => {
    writePersistedComposerDraft('almost submitted prompt')

    expect(readPersistedComposerDraft()).toBe('almost submitted prompt')
    expect(window.localStorage.getItem(COMPOSER_DRAFT_STORAGE_KEY)).toBe('almost submitted prompt')
  })

  it('removes empty drafts instead of leaving stale text behind', () => {
    writePersistedComposerDraft('saved')
    writePersistedComposerDraft('')

    expect(readPersistedComposerDraft()).toBe('')
    expect(window.localStorage.getItem(COMPOSER_DRAFT_STORAGE_KEY)).toBeNull()
  })

  it('can explicitly clear a saved draft after submit', () => {
    writePersistedComposerDraft('saved')
    clearPersistedComposerDraft()

    expect(readPersistedComposerDraft()).toBe('')
  })
})
