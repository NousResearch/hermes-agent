import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import {
  $composerAttachments,
  $voiceConversationStartRequest,
  addComposerAttachment,
  clearSessionDraft,
  type ComposerAttachment,
  createComposerAttachmentOccurrenceId,
  createComposerAttachmentScope,
  migrateSessionDraft,
  reloadPersistedDrafts,
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

  it('persists draft text (not attachments) to localStorage under a profile-scoped key', () => {
    stashSessionDraft('session-a', 'survives reload', [attachment({ id: 'file:a' })])

    const persisted = JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<
      string,
      string
    >

    expect(persisted['default:session-a']).toBe('survives reload')
    // The bare, profile-less v3 shape must not come back: it is what let one
    // profile's draft restore into another's composer.
    expect(persisted['session-a']).toBeUndefined()
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

describe('session drafts are scoped per profile', () => {
  const session = (id: string, profile: string): SessionInfo => ({ id, profile }) as unknown as SessionInfo

  afterEach(() => {
    // The in-memory stash outlives localStorage.clear(), so drop each profile's
    // buckets through the real API before resetting the profile atoms.
    for (const profile of ['default', 'uwr', 'saint']) {
      $activeGatewayProfile.set(profile)
      $newChatProfile.set(null)
      clearSessionDraft(null)

      for (const scope of ['uwr-session', 'saint-session', 'session-a']) {
        clearSessionDraft(scope)
      }

      $newChatProfile.set(profile)
      clearSessionDraft(null)
    }

    $sessions.set([])
    $activeGatewayProfile.set('default')
    $newChatProfile.set(null)
    window.localStorage.clear()
  })

  // Regression: a draft (text AND its staged image attachments) stashed while
  // one profile was live could be restored into a composer belonging to
  // another profile, and submitted there. The draft map keyed on session id
  // alone, so one flat namespace served every profile in the renderer.
  it("does not share a namespace between two profiles' drafts", () => {
    $sessions.set([session('uwr-session', 'uwr'), session('saint-session', 'saint')])

    $activeGatewayProfile.set('uwr')
    stashSessionDraft('uwr-session', 'reference images prompt', [attachment({ id: 'image:ref', kind: 'image' })])

    $activeGatewayProfile.set('saint')
    stashSessionDraft('saint-session', 'saint prompt', [])

    const persisted = Object.keys(
      JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<string, string>
    )

    expect(persisted).toContain('uwr:uwr-session')
    expect(persisted).toContain('saint:saint-session')
    // Neither profile may reach the other's draft through the flat v3 key.
    expect(takeSessionDraft('saint-session').attachments).toEqual([])
  })

  // The unsent `__new__` bucket was a single global slot, so every new-session
  // composer in every profile read and wrote the same draft.
  it("keeps each profile's unsent new-session draft in its own bucket", () => {
    $activeGatewayProfile.set('uwr')
    stashSessionDraft(null, 'uwr new chat', [attachment({ id: 'image:ref', kind: 'image' })])

    $activeGatewayProfile.set('saint')

    expect(takeSessionDraft(null)).toEqual({ attachments: [], text: '' })

    stashSessionDraft(null, 'saint new chat', [])
    $activeGatewayProfile.set('uwr')

    expect(takeSessionDraft(null).text).toBe('uwr new chat')
    expect(takeSessionDraft(null).attachments.map(a => a.id)).toEqual(['image:ref'])
  })

  // A background profile keeps streaming during a live swap, so a draft must
  // stay keyed to the profile that OWNS its session, not whichever profile
  // happens to be live when the composer reads it back.
  it("keys a draft to its session's owning profile, not the live one", () => {
    $sessions.set([session('uwr-session', 'uwr')])

    $activeGatewayProfile.set('uwr')
    stashSessionDraft('uwr-session', 'half typed', [])

    $activeGatewayProfile.set('saint')

    expect(takeSessionDraft('uwr-session').text).toBe('half typed')
  })

  // The new-chat picker points the composer at a profile before any session
  // exists, so an unsent draft belongs to the picked target.
  it('scopes a new-session draft to the new-chat picker target when set', () => {
    $activeGatewayProfile.set('saint')
    $newChatProfile.set('uwr')
    stashSessionDraft(null, 'drafted for uwr', [])

    $newChatProfile.set(null)

    expect(takeSessionDraft(null)).toEqual({ attachments: [], text: '' })

    $newChatProfile.set('uwr')
    expect(takeSessionDraft(null).text).toBe('drafted for uwr')
  })

  it('discards the legacy profile-less v3 map instead of misattributing it', () => {
    window.localStorage.setItem('hermes:composer-drafts:v3', JSON.stringify({ __new__: 'ambiguous', 'session-a': 'x' }))

    reloadPersistedDrafts()

    expect(window.localStorage.getItem('hermes:composer-drafts:v3')).toBeNull()
    expect(takeSessionDraft('session-a')).toEqual({ attachments: [], text: '' })
    expect(takeSessionDraft(null)).toEqual({ attachments: [], text: '' })
  })
})
