import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $composerAttachments,
  addComposerAttachment,
  clearComposerAttachments,
  clearSessionDraft,
  type ComposerAttachment,
  createComposerAttachmentScope,
  mainComposerScope,
  removeComposerAttachment,
  SESSION_DRAFTS_STORAGE_KEY,
  stashSessionDraft,
  takeSessionDraft,
  updateComposerAttachment
} from './composer'

function attachment(overrides: Partial<ComposerAttachment> & Pick<ComposerAttachment, 'id'>): ComposerAttachment {
  return { kind: 'file', label: 'doc.pdf', ...overrides }
}

/** Some node/vitest host combos ship a non-functional localStorage stub
 *  (`getItem`/`clear` missing). Provide an in-memory implementation so draft
 *  persistence tests exercise real store behavior rather than the host stub. */
function installMemoryLocalStorage() {
  const data = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return data.size
    },
    clear() {
      data.clear()
    },
    getItem(key) {
      return data.has(key) ? data.get(key)! : null
    },
    key(index) {
      return [...data.keys()][index] ?? null
    },
    removeItem(key) {
      data.delete(key)
    },
    setItem(key, value) {
      data.set(key, String(value))
    }
  }

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: storage,
    writable: true
  })

  return storage
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

describe('session drafts', () => {
  beforeEach(() => {
    installMemoryLocalStorage()
  })

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
})

describe('attachment scope change notifications (#68417)', () => {
  beforeEach(() => {
    installMemoryLocalStorage()
  })

  afterEach(() => {
    $composerAttachments.set([])
    clearComposerAttachments()
    mainComposerScope.setOnChange(null)

    for (const scope of ['session-a', 'session-b', null]) {
      clearSessionDraft(scope)
    }

    window.localStorage.clear()
  })

  it('notifies on structural add/remove/clear but not upload-state-only changes', () => {
    const scope = createComposerAttachmentScope()
    const seen: number[] = []

    scope.setOnChange(attachments => {
      seen.push(attachments.length)
    })

    scope.add(attachment({ id: 'file:a' }))
    scope.setUploadState('file:a', 'uploading')
    scope.remove('file:a')
    scope.add(attachment({ id: 'file:b' }))
    scope.clear()
    // No-op clear must not re-notify.
    scope.clear()
    // Missing id must not re-notify.
    scope.remove('file:missing')

    expect(seen).toEqual([1, 0, 1, 0])
  })

  it('does not resurrect removed attachments after a session-scope restore', () => {
    // Mirrors the desktop composer path:
    // 1) stash attachments with the session
    // 2) user removes a chip (onChange immediately re-stashes)
    // 3) session leave/re-enter reloads via takeSessionDraft + loadIntoComposer
    const scope = createComposerAttachmentScope()

    scope.setOnChange(attachments => {
      stashSessionDraft('session-a', 'still typing', attachments)
    })

    scope.add(attachment({ id: 'file:a', label: 'IndexTest.html' }))
    scope.add(attachment({ id: 'file:b', label: 'WechatTest.html' }))
    expect(takeSessionDraft('session-a').attachments.map(a => a.id)).toEqual(['file:a', 'file:b'])

    // User clicks × on both chips without typing (no text-debounce path).
    scope.remove('file:a')
    scope.remove('file:b')
    expect(scope.$attachments.get()).toHaveLength(0)
    expect(takeSessionDraft('session-a').attachments).toEqual([])

    // Session switch away + back restores from the stash — must stay empty.
    const restored = takeSessionDraft('session-a')
    scope.$attachments.set(restored.attachments.map(a => ({ ...a })))

    expect(scope.$attachments.get()).toEqual([])
    expect(restored.text).toBe('still typing')
  })

  it('wires the main composer helpers through the same notification path', () => {
    const onChange = vi.fn()

    mainComposerScope.setOnChange(onChange)
    addComposerAttachment(attachment({ id: 'file:main' }))
    removeComposerAttachment('file:main')
    clearComposerAttachments()
    mainComposerScope.setOnChange(null)

    expect(onChange).toHaveBeenCalled()
    expect($composerAttachments.get()).toEqual([])
  })
})
