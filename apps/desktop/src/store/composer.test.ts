import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $composerAttachments,
  $voiceConversationStartRequest,
  addComposerAttachment,
  clearSessionDraft,
  type ComposerAttachment,
  createComposerAttachmentOccurrenceId,
  createComposerAttachmentScope,
  MAX_COMPOSER_TEXT_CHARS,
  migrateSessionDraft,
  removeComposerAttachment,
  requestVoiceConversationStart,
  SESSION_DRAFTS_STORAGE_KEY,
  stashSessionDraft,
  takeSessionDraft,
  takeVoiceConversationStart,
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

  it('drops an oversized draft from localStorage so a huge paste cannot poison the next launch (#98562)', () => {
    stashSessionDraft('session-a', 'x'.repeat(MAX_COMPOSER_TEXT_CHARS + 1), [])

    const persisted = JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<
      string,
      string
    >

    expect(persisted['session-a']).toBeUndefined()
  })

  it('keeps an oversized draft in memory for the current session', () => {
    const oversized = 'x'.repeat(MAX_COMPOSER_TEXT_CHARS + 1)
    stashSessionDraft('session-a', oversized, [])

    expect(takeSessionDraft('session-a').text).toBe(oversized)
  })

  it('persists a draft exactly at the cap', () => {
    stashSessionDraft('session-a', 'x'.repeat(MAX_COMPOSER_TEXT_CHARS), [])

    const persisted = JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<
      string,
      string
    >

    expect(persisted['session-a']).toHaveLength(MAX_COMPOSER_TEXT_CHARS)
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

  it('migrates a tip-keyed draft onto the post-compression tip', () => {
    const tipBefore = '20260720_062637_ad96b3'
    const tipAfter = '20260720_071049_a28905'

    stashSessionDraft(tipBefore, 'half typed while thinking', [])

    expect(migrateSessionDraft(tipBefore, tipAfter)).toBe(true)
    expect(takeSessionDraft(tipAfter).text).toBe('half typed while thinking')
    expect(takeSessionDraft(tipBefore).text).toBe('')

    clearSessionDraft(tipAfter)
  })

  it('does not overwrite a non-empty destination draft during migration', () => {
    stashSessionDraft('from', 'old tip draft', [])
    stashSessionDraft('to', 'already typed on new tip', [])

    expect(migrateSessionDraft('from', 'to')).toBe(false)
    expect(takeSessionDraft('to').text).toBe('already typed on new tip')
    expect(takeSessionDraft('from').text).toBe('old tip draft')

    clearSessionDraft('from')
    clearSessionDraft('to')
  })
})
