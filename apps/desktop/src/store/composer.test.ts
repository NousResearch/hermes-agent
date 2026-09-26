import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $composerAttachments,
  $restoredDraftNotice,
  $voiceConversationStartRequest,
  addComposerAttachment,
  adoptGoneSessionDraft,
  announceGoneSessionDraft,
  clearSessionDraft,
  type ComposerAttachment,
  createComposerAttachmentOccurrenceId,
  createComposerAttachmentScope,
  mainComposerScope,
  migrateSessionDraft,
  removeComposerAttachment,
  requestVoiceConversationStart,
  revokeAttachmentPreviewUrls,
  SESSION_DRAFTS_STORAGE_KEY,
  stashSessionDraft,
  takeSessionDraft,
  takeVoiceConversationStart,
  undoRestoredDraft,
  updateComposerAttachment
} from './composer'

describe('voice conversation start requests', () => {
  it('latches each request until the main composer consumes it once', () => {
    requestVoiceConversationStart()
    const first = $voiceConversationStartRequest.get()

    expect(takeVoiceConversationStart(first)).toBe(true)
    expect(takeVoiceConversationStart(first)).toBe(false)

    requestVoiceConversationStart()
    expect(takeVoiceConversationStart($voiceConversationStartRequest.get())).toBe(true)
  })
})

function attachment(overrides: Partial<ComposerAttachment> & Pick<ComposerAttachment, 'id'>): ComposerAttachment {
  return { kind: 'file', label: 'doc.pdf', ...overrides }
}

function stubRevokeObjectURL() {
  const revokeObjectURL = vi.fn()
  vi.stubGlobal('URL', { ...URL, revokeObjectURL })

  return revokeObjectURL
}

describe('blob preview URL ownership handoff', () => {
  afterEach(() => {
    $composerAttachments.set([])
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('direct-submit handoff keeps blob previews across clear, then revokes when the optimistic consumer is discarded', () => {
    // Mirrors use-composer-submit: clone → clear({ retainPreviewUrls }) → dispatch clone.
    const revokeObjectURL = stubRevokeObjectURL()
    const blobUrl = 'blob:hermes-direct-submit-1'
    addComposerAttachment(
      attachment({ id: 'image:drop', kind: 'image', label: 'Lattice.png', previewUrl: blobUrl })
    )

    const submittedAttachments = $composerAttachments.get().map(item => ({ ...item }))
    mainComposerScope.clear({ retainPreviewUrls: true })

    expect($composerAttachments.get()).toEqual([])
    expect(revokeObjectURL).not.toHaveBeenCalled()
    expect(submittedAttachments[0]?.previewUrl).toBe(blobUrl)

    // Optimistic bubble discarded / replaced without restoring into the composer.
    revokeAttachmentPreviewUrls(submittedAttachments)
    expect(revokeObjectURL).toHaveBeenCalledWith(blobUrl)
  })

  it('still revokes blob previews on a normal clear (no handoff)', () => {
    const revokeObjectURL = stubRevokeObjectURL()
    const blobUrl = 'blob:hermes-clear-1'
    const scope = createComposerAttachmentScope()
    scope.add(attachment({ id: 'image:x', kind: 'image', previewUrl: blobUrl }))

    scope.clear()

    expect(scope.$attachments.get()).toEqual([])
    expect(revokeObjectURL).toHaveBeenCalledWith(blobUrl)
  })
})

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

  it('updates only the exact attachment occurrence captured before an async operation', () => {
    const scope = createComposerAttachmentScope()

    const first = attachment({
      id: 'image:a',
      kind: 'image',
      occurrenceId: createComposerAttachmentOccurrenceId(),
      path: '/tmp/a.png'
    })

    const replacement = attachment({
      id: 'image:a',
      kind: 'image',
      occurrenceId: createComposerAttachmentOccurrenceId(),
      path: '/tmp/a.png'
    })

    scope.add(first)
    scope.remove(first.id)
    scope.add(replacement)

    expect(scope.updateIfCurrent(first, { thumbnailUrl: 'data:image/png;base64,stale' })).toBe(false)
    expect(scope.$attachments.get()).toEqual([replacement])
    expect(scope.updateIfCurrent(replacement, { thumbnailUrl: 'data:image/png;base64,current' })).toBe(true)
    expect(scope.$attachments.get()[0]?.thumbnailUrl).toBe('data:image/png;base64,current')
  })

  it('recognizes the same attachment occurrence after a session-draft clone', () => {
    const scope = createComposerAttachmentScope()

    const original = attachment({
      id: 'image:draft',
      kind: 'image',
      occurrenceId: createComposerAttachmentOccurrenceId(),
      path: '/tmp/draft.png'
    })

    stashSessionDraft('session-a', '', [original])
    const restored = takeSessionDraft('session-a').attachments[0]!
    scope.add(restored)

    expect(restored).not.toBe(original)
    expect(scope.updateIfCurrent(original, { thumbnailUrl: 'data:image/png;base64,current' })).toBe(true)
    expect(scope.$attachments.get()[0]?.thumbnailUrl).toBe('data:image/png;base64,current')
    clearSessionDraft('session-a')
  })

  it('merges concurrent staging fields without discarding an existing thumbnail', () => {
    const scope = createComposerAttachmentScope()

    const original = attachment({
      id: 'image:staging',
      kind: 'image',
      occurrenceId: createComposerAttachmentOccurrenceId(),
      path: 'C:\\Users\\alice\\Pictures\\photo.png'
    })

    scope.add(original)
    expect(scope.updateIfCurrent(original, { thumbnailUrl: 'data:image/png;base64,current' })).toBe(true)
    expect(
      scope.updateIfCurrent(original, {
        attachedSessionId: 'session-1',
        path: '/root/.hermes/attachments/photo.png',
        uploadState: undefined
      })
    ).toBe(true)

    expect(scope.$attachments.get()[0]).toMatchObject({
      attachedSessionId: 'session-1',
      path: '/root/.hermes/attachments/photo.png',
      thumbnailUrl: 'data:image/png;base64,current'
    })
  })

  it('removes submitted occurrences while preserving unrelated attachments', () => {
    const scope = createComposerAttachmentScope()

    const submitted = attachment({
      id: 'image:submitted',
      kind: 'image',
      occurrenceId: 'occurrence-submitted'
    })

    const other = attachment({ id: 'file:other', occurrenceId: 'occurrence-other' })

    scope.add(submitted)
    scope.add(other)
    scope.removeOccurrences([submitted])

    expect(scope.$attachments.get()).toEqual([other])
  })

  it('preserves a same-id replacement of a submitted occurrence', () => {
    const scope = createComposerAttachmentScope()

    const submitted = attachment({
      id: 'image:submitted',
      kind: 'image',
      occurrenceId: 'occurrence-submitted'
    })

    const replacement = attachment({
      ...submitted,
      occurrenceId: 'occurrence-replacement'
    })

    scope.add(replacement)
    scope.removeOccurrences([submitted])

    expect(scope.$attachments.get()).toEqual([replacement])
  })

  it('preserves a newer same-id legacy attachment and still emits on successful cleanup', () => {
    const scope = createComposerAttachmentScope()
    const submitted = attachment({ id: 'url:https://example.com', kind: 'url', label: 'old' })
    const replacement = attachment({ id: submitted.id, kind: 'url', label: 'new' })
    const listener = vi.fn()
    const unlisten = scope.$attachments.listen(listener)

    scope.add(submitted)
    scope.remove(submitted.id)
    scope.add(replacement)
    listener.mockClear()

    scope.removeOccurrences([submitted])

    expect(scope.$attachments.get()).toEqual([replacement])
    expect(listener).toHaveBeenCalledTimes(1)
    unlisten()
  })

  it('removes the exact submitted legacy attachment', () => {
    const scope = createComposerAttachmentScope()
    const submitted = attachment({ id: 'url:https://example.com', kind: 'url' })

    scope.add(submitted)
    scope.removeOccurrences([submitted])

    expect(scope.$attachments.get()).toEqual([])
  })
})

describe('session drafts', () => {
  afterEach(() => {
    for (const scope of ['session-a', 'session-b', null]) {
      clearSessionDraft(scope)
    }

    window.localStorage.clear()
  })

  it('keeps drafts isolated per session scope', () => {
    stashSessionDraft('session-a', 'draft a', [])
    stashSessionDraft('session-b', 'draft b', [attachment({ id: 'image:b', kind: 'image' })])

    expect(takeSessionDraft('session-a')).toEqual({ attachments: [], text: 'draft a' })
    expect(takeSessionDraft('session-b').text).toBe('draft b')
    expect(takeSessionDraft('session-b').attachments.map(a => a.id)).toEqual(['image:b'])
  })

  it('scopes the unsaved new-session draft separately from real sessions', () => {
    stashSessionDraft(null, 'new chat draft', [])
    stashSessionDraft('session-a', 'session draft', [])

    expect(takeSessionDraft(null).text).toBe('new chat draft')
    expect(takeSessionDraft(undefined).text).toBe('new chat draft')
    expect(takeSessionDraft('session-a').text).toBe('session draft')
  })

  it('persists draft text (not attachments) to localStorage', () => {
    stashSessionDraft('session-a', 'survives reload', [attachment({ id: 'file:a' })])

    const persisted = JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<
      string,
      string
    >

    expect(persisted['session-a']).toBe('survives reload')
  })

  it('evicts empty drafts instead of leaving stale entries behind', () => {
    stashSessionDraft('session-a', 'saved', [])
    stashSessionDraft('session-a', '   ', [])

    expect(takeSessionDraft('session-a')).toEqual({ attachments: [], text: '' })
  })

  it('clears a stashed draft after an accepted submit', () => {
    stashSessionDraft('session-a', 'sent prompt', [attachment({ id: 'file:a' })])
    clearSessionDraft('session-a')

    expect(takeSessionDraft('session-a')).toEqual({ attachments: [], text: '' })
  })

  it('returns clones so callers cannot mutate the stash', () => {
    stashSessionDraft('session-a', 'draft', [attachment({ id: 'file:a' })])

    const taken = takeSessionDraft('session-a')
    taken.attachments[0]!.label = 'mutated'

    expect(takeSessionDraft('session-a').attachments[0]?.label).toBe('doc.pdf')
  })

  it('restores a gone session draft only into an EMPTY fresh chat, and Undo puts it back where it was (#111868)', () => {
    // Never clobber what the user is already typing in the new chat.
    stashSessionDraft('session-a', 'from the dead session', [])
    stashSessionDraft(null, 'already composing here', [])
    announceGoneSessionDraft('session-a')

    expect(adoptGoneSessionDraft()).toBe(false)
    expect($restoredDraftNotice.get()).toBeNull()
    expect(takeSessionDraft(null).text).toBe('already composing here')
    expect(takeSessionDraft('session-a').text).toBe('from the dead session')

    // Empty fresh chat → restored; Undo (text untouched) returns it to the
    // dead key, so the same recovery path can find it again later.
    clearSessionDraft(null)
    announceGoneSessionDraft('session-a')

    expect(adoptGoneSessionDraft()).toBe(true)
    expect(takeSessionDraft(null).text).toBe('from the dead session')
    expect(undoRestoredDraft('from the dead session')).toBe(true)
    expect(takeSessionDraft(null).text).toBe('')
    expect(takeSessionDraft('session-a').text).toBe('from the dead session')
    expect($restoredDraftNotice.get()).toBeNull()

    // Once the user has edited the restored text, Undo would destroy their
    // work: it only dismisses.
    announceGoneSessionDraft('session-a')
    adoptGoneSessionDraft()

    expect(undoRestoredDraft('from the dead session, edited')).toBe(false)
    expect(takeSessionDraft(null).text).toBe('from the dead session')
  })

  it('migrates a tip-keyed draft onto the post-compression tip', () => {
    const tipBefore = '20260720_062637_ad96b3'
    const tipAfter = '20260720_071049_a28905'

    stashSessionDraft(tipBefore, 'half typed while thinking', [])

    expect(migrateSessionDraft(tipBefore, tipAfter)).toBe(true)
    expect(takeSessionDraft(tipAfter).text).toBe('half typed while thinking')
    expect(takeSessionDraft(tipBefore).text).toBe('')

    clearSessionDraft(tipAfter)
  })

  it('does not overwrite a destination draft or its attachments during migration', () => {
    const destinationAttachment = attachment({ id: 'file:destination' })
    stashSessionDraft(null, 'new chat draft', [attachment({ id: 'file:source' })])
    stashSessionDraft('to', 'already typed on new tip', [destinationAttachment])

    expect(migrateSessionDraft(null, 'to')).toBe(false)
    expect(takeSessionDraft('to').text).toBe('already typed on new tip')
    expect(takeSessionDraft('to').attachments).toEqual([destinationAttachment])
    expect(takeSessionDraft(null).text).toBe('new chat draft')

    clearSessionDraft(null)
    clearSessionDraft('to')
  })
})
