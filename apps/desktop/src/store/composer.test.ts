import { afterEach, describe, expect, it } from 'vitest'

import {
  $composerAttachments,
  $composerContextReferences,
  addComposerAttachment,
  clearComposerContextReferences,
  clearSessionDraft,
  type ComposerAttachment,
  composerContextBlocksFromDraft,
  composerContextReferencePreview,
  nextComposerContextReferenceLabel,
  reconcileComposerContextReferences,
  removeComposerAttachment,
  SESSION_DRAFTS_STORAGE_KEY,
  setComposerSelectionReference,
  setComposerTerminalSelection,
  stashSessionDraft,
  takeSessionDraft,
  terminalContextBlocksFromDraft,
  updateComposerAttachment
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

describe('composer context references', () => {
  afterEach(() => {
    clearComposerContextReferences()
  })

  it('keeps terminal selection blocks compatible with the existing shortcut', () => {
    setComposerTerminalSelection('zsh:4-6', 'npm test\nPASS')

    expect(terminalContextBlocksFromDraft('@terminal:`zsh:4-6`')).toEqual(['```terminal\nnpm test\nPASS\n```'])
  })

  it('turns generic selection refs into hidden selected-context blocks', () => {
    setComposerSelectionReference('_selection', 'Selected assistant text')

    expect(composerContextBlocksFromDraft('@selection:`_selection` please use this')).toEqual([
      'Selected context (_selection):\n```text\nSelected assistant text\n```'
    ])
    expect(composerContextReferencePreview('selection', '_selection')).toBe('Selected assistant text')
  })

  it('allocates unique labels for repeated floating selections', () => {
    expect(nextComposerContextReferenceLabel('selection', '_selection')).toBe('_selection')
    setComposerSelectionReference('_selection', 'first')

    expect(nextComposerContextReferenceLabel('selection', '_selection')).toBe('_selection-2')
  })

  it('reconciles deleted context chips out of the preview store', () => {
    setComposerSelectionReference('_selection', 'first')
    setComposerTerminalSelection('zsh:1', 'second')

    expect(Object.keys($composerContextReferences.get())).toHaveLength(2)
    reconcileComposerContextReferences('@terminal:`zsh:1`')
    expect($composerContextReferences.get()).toEqual({ 'terminal:zsh:1': 'second' })
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

    const persisted = JSON.parse(window.localStorage.getItem(SESSION_DRAFTS_STORAGE_KEY) ?? '{}') as Record<string, string>

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
})
