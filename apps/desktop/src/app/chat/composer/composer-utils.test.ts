import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { afterEach, describe, expect, it } from 'vitest'

import type { ComposerAttachment } from '@/store/composer'

import {
  acceptsTriggerCompletion,
  implicitSlashAcceptIndex,
  isPendingDraftPersistCurrent,
  liveComposerDraft,
  type PendingDraftPersist,
  queuedEditBufferClean,
  queuedEditStashPlan,
  type QueueEditState,
  shouldDisableComposerInput,
  slashArgStage,
  slashChipKindForItem,
  slashCommandToken,
  type TriggerAcceptInput
} from './composer-utils'
import { normalizeComposerEditorDom, RICH_INPUT_SLOT } from './rich-editor'

const item = (group: string): Unstable_TriggerItem =>
  ({ id: 'x', type: 'slash', label: 'x', metadata: { group } }) as unknown as Unstable_TriggerItem

describe('shouldDisableComposerInput', () => {
  it.each(['idle', 'connecting', 'closed', 'error'] as const)(
    'keeps the draft editable while the gateway is %s',
    gatewayState => {
      expect(shouldDisableComposerInput(true, gatewayState)).toBe(false)
    }
  )

  it('fails closed when connection atoms disagree about an open gateway', () => {
    expect(shouldDisableComposerInput(true, 'open')).toBe(true)
  })

  it.each(['idle', 'connecting', 'open', 'closed', 'error'] as const)(
    'never disables an otherwise enabled composer while the gateway is %s',
    gatewayState => {
      expect(shouldDisableComposerInput(false, gatewayState)).toBe(false)
    }
  )
})

describe('slashArgStage', () => {
  it('is true only once the query is past the command name', () => {
    expect(slashArgStage('personality')).toBe(false)
    expect(slashArgStage('personality alice')).toBe(true)
  })
})

describe('slashCommandToken', () => {
  it('extracts the lowercased /command token', () => {
    expect(slashCommandToken('Personality alice')).toBe('/personality')
    expect(slashCommandToken('model')).toBe('/model')
  })

  it('handles an empty query', () => {
    expect(slashCommandToken('')).toBe('/')
  })
})

describe('slashChipKindForItem', () => {
  it('maps completion groups to chip kinds', () => {
    expect(slashChipKindForItem(item('Skills'))).toBe('skill')
    expect(slashChipKindForItem(item('Themes'))).toBe('theme')
    expect(slashChipKindForItem(item('Commands'))).toBe('command')
  })
})

describe('acceptsTriggerCompletion', () => {
  const press = (key: string, overrides: Partial<TriggerAcceptInput> = {}) =>
    acceptsTriggerCompletion({
      activeExplicit: false,
      freeTextArgStage: false,
      key,
      kind: '/',
      query: 'personality alic',
      ...overrides
    })

  it('accepts on Enter / Tab / Space for a finite option list', () => {
    expect(press('Enter')).toBe(true)
    expect(press('Tab')).toBe(true)
    expect(press(' ')).toBe(true)
  })

  it('ignores keys that are neither navigation nor acceptance', () => {
    expect(press('a')).toBe(false)
    expect(press('Escape')).toBe(false)
  })

  it('lets an `@` mention take a literal space', () => {
    expect(press(' ', { kind: '@', query: 'src/comp' })).toBe(false)
    expect(press('Enter', { kind: '@', query: 'src/comp' })).toBe(true)
  })

  it('types a space on a bare `/ ` instead of accepting', () => {
    expect(press(' ', { query: '' })).toBe(false)
  })

  // The `/goal <prose>` class: the popover may be live over free-form text, so
  // the keys that mean something else in prose must keep meaning it.
  it('sends the prose rather than the unchosen first row', () => {
    expect(press('Enter', { freeTextArgStage: true, query: 'goal ship the redesign' })).toBe(false)
    expect(press(' ', { freeTextArgStage: true, query: 'goal ship the' })).toBe(false)
  })

  it('accepts on Enter once the user has arrowed to a row deliberately', () => {
    expect(press('Enter', { activeExplicit: true, freeTextArgStage: true, query: 'goal stat' })).toBe(true)
  })

  it('keeps Tab as the explicit accept even over free text', () => {
    expect(press('Tab', { freeTextArgStage: true, query: 'goal stat' })).toBe(true)
  })
})

describe('implicitSlashAcceptIndex', () => {
  const rows = ['/compress', '/review', '/resume']

  it('completes a prefix of the highlighted row', () => {
    expect(implicitSlashAcceptIndex('com', rows, 0, false)).toBe(0)
  })

  it('keeps a fully typed command even when another row is highlighted', () => {
    expect(implicitSlashAcceptIndex('review', rows, 0, false)).toBe(1)
  })

  it('does not steal when the typed token is not a prefix of any row', () => {
    expect(implicitSlashAcceptIndex('review', ['/compress', '/resume'], 0, false)).toBeNull()
  })

  it('takes the only prefix match when the highlight is a leftover', () => {
    expect(implicitSlashAcceptIndex('rev', ['/compress', '/review', '/resume'], 0, false)).toBe(1)
  })

  it('honours an arrowed pick even when it is not a prefix', () => {
    expect(implicitSlashAcceptIndex('review', rows, 0, true)).toBe(0)
  })

  // Regression for #98535: bare `/` + arrow-key selection must commit the
  // highlighted row, not return null.  Before the fix, `!typed` fired first
  // and the explicit-pick branch was never reached.
  it('honours an arrowed pick on a bare `/` query', () => {
    expect(implicitSlashAcceptIndex('/', rows, 1, true)).toBe(1)
    expect(implicitSlashAcceptIndex('/', rows, 2, true)).toBe(2)
    // detectTrigger('/') yields query '', not '/'. Same empty token, same pick.
    expect(implicitSlashAcceptIndex('', rows, 1, true)).toBe(1)
  })

  it('still returns null on a bare `/` with no arrow-key selection', () => {
    expect(implicitSlashAcceptIndex('/', rows, 0, false)).toBeNull()
    expect(implicitSlashAcceptIndex('', rows, 0, false)).toBeNull()
  })

  it('matches an arg-stage prefix against the full completion text', () => {
    expect(implicitSlashAcceptIndex('personality alic', ['/personality alice', '/personality none'], 0, false)).toBe(0)
  })
})

describe('isPendingDraftPersistCurrent (#54527 integrity guard)', () => {
  it('accepts a write when the pending entry still matches what was captured', () => {
    const entry: PendingDraftPersist = { scope: 'session-a', text: 'hello' }

    expect(isPendingDraftPersistCurrent(entry, entry)).toBe(true)
    expect(isPendingDraftPersistCurrent({ scope: 'session-a', text: 'hello' }, entry)).toBe(true)
  })

  it('rejects when the pending slot was cleared (session swap / newer flush already committed)', () => {
    const entry: PendingDraftPersist = { scope: 'session-a', text: 'hello' }

    expect(isPendingDraftPersistCurrent(null, entry)).toBe(false)
  })

  it('rejects when the pending slot now belongs to a different session (the #54527 misroute shape)', () => {
    const captured: PendingDraftPersist = { scope: 'session-a', text: 'carefully composed prompt' }
    const supersededBy: PendingDraftPersist = { scope: 'session-b', text: 'different draft' }

    expect(isPendingDraftPersistCurrent(supersededBy, captured)).toBe(false)
  })

  it('rejects when the pending slot was replaced by a newer keystroke in the same session', () => {
    const captured: PendingDraftPersist = { scope: 'session-a', text: 'first draft' }
    const supersededBy: PendingDraftPersist = { scope: 'session-a', text: 'first draft continued' }

    expect(isPendingDraftPersistCurrent(supersededBy, captured)).toBe(false)
  })

  it('rejects when nothing was ever captured', () => {
    expect(isPendingDraftPersistCurrent(null, null)).toBe(false)
  })
})

/** Real contentEditable, built the way `empty-composer.test.ts` builds one. */
function editorWith(text: string): HTMLDivElement {
  const el = document.createElement('div')

  el.dataset.slot = RICH_INPUT_SLOT
  el.contentEditable = 'true'
  el.append(document.createTextNode(text))
  normalizeComposerEditorDom(el)
  document.body.append(el)

  return el
}

// editorWith appends to the shared JSDOM body; empty it so the element does not
// leak into other cases in this file.
afterEach(() => {
  document.body.replaceChildren()
})

describe('liveComposerDraft (stale-mirror guard for the ArrowUp recall)', () => {
  it('reads the live editor text even when the mirror is still empty', () => {
    // The race this exists for: a keystroke or paste flushed only by the
    // coalesced rAF, so `draftRef.current` holds the pre-keystroke text while
    // the editor already holds what the user typed. The recall guard must see
    // the typed text, not the stale empty mirror.
    const editor = editorWith('just typed this')

    expect(liveComposerDraft(editor, '')).toBe('just typed this')
  })

  it('falls back to the mirror before the editor mounts', () => {
    expect(liveComposerDraft(null, 'mirrored draft')).toBe('mirrored draft')
  })
})

describe('queuedEditBufferClean / queuedEditStashPlan (payload-aware dirty decision, #88621 review R2)', () => {
  const fileA: ComposerAttachment = { id: 'file:a', kind: 'file', label: 'a.txt' }
  const fileA_clone = { ...fileA }
  const fileB: ComposerAttachment = { id: 'file:b', kind: 'file', label: 'b.txt' }

  const editing: QueueEditState = {
    attachments: [fileB],
    draft: 'original draft',
    entryId: 'e1',
    entryText: 'queued words',
    entryAttachments: [fileA],
    sessionKey: 's1'
  }

  it('clean = untouched text AND untouched attachment set', () => {
    expect(queuedEditBufferClean(editing, 'queued words', [fileA])).toBe(true)
    expect(queuedEditBufferClean(editing, 'queued words', [fileA_clone])).toBe(true)
  })

  it('a chip added with unchanged text is DIRTY — the old text-only check destroyed it', () => {
    expect(queuedEditBufferClean(editing, 'queued words', [fileA, fileB])).toBe(false)
  })

  it('a chip removed with unchanged text is DIRTY', () => {
    expect(queuedEditBufferClean(editing, 'queued words', [])).toBe(false)
  })

  it('changed text stays dirty regardless of attachments', () => {
    expect(queuedEditBufferClean(editing, 'edited words', [fileA])).toBe(false)
  })

  it('a reordering of the same chips is dirty (order is part of the payload)', () => {
    const first: ComposerAttachment = { id: 'file:1', kind: 'file', label: '1' }
    const second: ComposerAttachment = { id: 'file:2', kind: 'file', label: '2' }
    const reordered: QueueEditState = { ...editing, entryAttachments: [first, second] }

    expect(queuedEditBufferClean(reordered, 'queued words', [second, first])).toBe(false)
    expect(queuedEditBufferClean(reordered, 'queued words', [first, second])).toBe(true)
  })

  it('clean stash plan keeps the pre-edit draft with ITS OWN attachments', () => {
    const plan = queuedEditStashPlan(editing, 'queued words', [fileA])

    expect(plan.text).toBe('original draft')
    expect(plan.attachments).toEqual([fileB])
  })

  it('dirty (attachment-only) stash plan keeps the DISPLACED draft; the buffer rides the notice (R1)', () => {
    const live = [fileA, fileB]

    const plan = queuedEditStashPlan(editing, 'queued words', live)

    // R1: the stash is the displaced pre-edit draft's owner — the dirty
    // buffer is published separately by the caller as a salvage notice.
    expect(plan.text).toBe('original draft')
    expect(plan.attachments).toEqual([fileB])
  })
})
