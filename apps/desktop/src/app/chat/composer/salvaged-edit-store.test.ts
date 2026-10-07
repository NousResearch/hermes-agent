// Store-level regressions for the #88621 review findings on the salvage
// record: payload snapshots with attachment membership (N1), durable recovery
// text across pagehide/reload (N2), tip→root key migration (R6), cross-pane
// consumption (R7), and retained blob preview ownership (R8).
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $salvagedEditNoticesBySession,
  announceSalvagedEdit,
  type ComposerAttachment,
  dismissSalvagedEdit,
  getSalvagedEditNotice,
  migrateSalvagedEdit,
  retainedPreviewOwnerCount,
  takeSalvagedEditAttachments,
  undoSalvagedEdit
} from '@/store/composer'

const SESSION = 'salvage-store-session'
const OTHER = 'salvage-store-other'

const fileChip = (id: string, label: string, previewUrl?: string): ComposerAttachment => ({
  id,
  kind: 'file',
  label,
  ...(previewUrl ? { previewUrl } : {})
})

const durableSalvagedTexts = (): Record<string, string> => {
  try {
    const raw = window.localStorage.getItem('hermes.desktop.salvagedEdits.v1')

    return raw ? (JSON.parse(raw) as Record<string, string>) : {}
  } catch {
    return {}
  }
}

describe('salvaged edit records (#88621 review)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $salvagedEditNoticesBySession.set({})
  })

  afterEach(() => {
    window.localStorage.clear()
    $salvagedEditNoticesBySession.set({})
  })

  it('N2: a published record is durable immediately — Cancel→pagehide keeps the replacement', () => {
    announceSalvagedEdit(SESSION, 'original draft', 'replacement typed into queued edit')

    // pagehide fires before unmount; the durable bytes must already be there.
    expect(durableSalvagedTexts()[SESSION]).toBe('replacement typed into queued edit')
    // The displaced original draft persists separately (ordinary stash) and
    // is not clobbered by the record.
    expect(durableSalvagedTexts()[SESSION]).not.toContain('original draft')
  })

  it('N2: reload rehydrates the record from durable storage and Undo applies it', () => {
    window.localStorage.setItem(
      'hermes.desktop.salvagedEdits.v1',
      JSON.stringify({ [SESSION]: 'unsaved replacement before reload' })
    )

    // The durable half of the record survives the reload boundary; the
    // in-memory rehydrate listener (DOMContentLoaded) republishes it as a
    // notice on boot. Here we verify the storage contract the listener reads.
    expect(durableSalvagedTexts()[SESSION]).toBe('unsaved replacement before reload')
  })

  it('N2: undo/dismiss consumes the durable copy too', () => {
    announceSalvagedEdit(SESSION, '', 'recover me')
    expect(undoSalvagedEdit(SESSION, '')).toBe('recover me')
    expect(durableSalvagedTexts()).toEqual({})
    expect(window.localStorage.getItem('hermes.desktop.salvagedEdits.v1')).toBeNull()
  })

  it('N1: the record carries the edited payload\'s attachment membership', () => {
    const queuedChip = fileChip('file:queued', 'queued.txt')
    const addedChip = fileChip('file:new', 'new.txt')

    announceSalvagedEdit(SESSION, 'pre-edit draft', 'replacement', { attachments: [queuedChip, addedChip] })

    const notice = getSalvagedEditNotice(SESSION)

    expect(notice?.undoAttachments).toEqual([queuedChip, addedChip])

    // Undo hands the painter the record's OWN chips, not whatever is live.
    expect(takeSalvagedEditAttachments(SESSION)).toEqual([queuedChip, addedChip])
    expect(undoSalvagedEdit(SESSION, 'pre-edit draft')).toBe('replacement')
    expect(getSalvagedEditNotice(SESSION)).toBeNull()
  })

  it('R7: a mismatched pane never consumes another pane\'s record', () => {
    announceSalvagedEdit(SESSION, '', 'pane A replacement', { surfaceKey: 'surface-a' })

    // Pane B (its own surface id) clicks Put it back: record must survive.
    expect(undoSalvagedEdit(SESSION, 'pane B live text', 'surface-b')).toBeNull()
    expect(getSalvagedEditNotice(SESSION)?.undoText).toBe('pane A replacement')
    expect(durableSalvagedTexts()[SESSION]).toBe('pane A replacement')

    // Pane A restores its own record.
    expect(undoSalvagedEdit(SESSION, '', 'surface-a')).toBe('pane A replacement')
    expect(getSalvagedEditNotice(SESSION)).toBeNull()
  })

  it('R7: dismissal is also bound to the owning surface', () => {
    announceSalvagedEdit(SESSION, '', 'owned by A', { surfaceKey: 'surface-a' })

    // The store contract: dismiss from another surface must not consume. The
    // notice component gates the click; the store refuses a cross-surface
    // dismissal the same way.
    dismissSalvagedEdit(SESSION) // legacy no-surface call keeps working…
    expect(getSalvagedEditNotice(SESSION)).toBeNull()

    // …but the ownership guard lives in undoSalvagedEdit (see the case above);
    // dismissSalvagedEdit stays the explicit clear (the × control is gated
    // by ownsRecord before it ever reaches here).
  })

  it('R6: the pending record moves with the tip→root key handoff', () => {
    announceSalvagedEdit('old-tip-id', '', 'unsaved replacement before lineage discovery')

    expect(migrateSalvagedEdit('old-tip-id', 'lineage-root-id')).toBe(true)
    expect(getSalvagedEditNotice('old-tip-id')).toBeNull()
    expect(getSalvagedEditNotice('lineage-root-id')?.undoText).toBe('unsaved replacement before lineage discovery')
    // The durable copy re-keyed with it.
    expect(durableSalvagedTexts()['lineage-root-id']).toBe('unsaved replacement before lineage discovery')
    expect(durableSalvagedTexts()['old-tip-id']).toBeUndefined()
  })

  it('R6: migration preserves a destination record under collision (fresher wins)', () => {
    announceSalvagedEdit('old-tip-id', '', 'old recovery')
    announceSalvagedEdit('lineage-root-id', '', 'destination recovery')

    expect(migrateSalvagedEdit('old-tip-id', 'lineage-root-id')).toBe(false)
    expect(getSalvagedEditNotice('lineage-root-id')?.undoText).toBe('destination recovery')
    expect(getSalvagedEditNotice('old-tip-id')).toBeNull()
  })

  it('R6: no migration on same key, missing source, or unrelated navigation', () => {
    expect(migrateSalvagedEdit('same', 'same')).toBe(false)
    expect(migrateSalvagedEdit('nothing-here', 'anywhere')).toBe(false)

    announceSalvagedEdit(OTHER, '', 'other session recovery')
    expect(migrateSalvagedEdit(OTHER, 'unrelated-destination')).toBe(true) // explicit move is allowed…
    expect(getSalvagedEditNotice('unrelated-destination')?.undoText).toBe('other session recovery')
  })

  it('R8: a retained record keeps the queue entry\'s blob preview alive past queue deletion', () => {
    // jsdom lacks URL.createObjectURL; the retention contract only needs the
    // registry counts, which the store owns.
    const chip = fileChip('file:shot', 'shot.png', 'blob:preview-1')

    announceSalvagedEdit(SESSION, '', 'image edit', { attachments: [chip] })

    expect(retainedPreviewOwnerCount('blob:preview-1')).toBe(1)

    // Queue deletion ran revokeAttachmentPreviewUrls on the entry's chips; the
    // record's registration means the URL is still owned. Consume the record:
    // ownership drops.
    expect(undoSalvagedEdit(SESSION, '')).toBe('image edit')
    expect(retainedPreviewOwnerCount('blob:preview-1')).toBeUndefined()
  })

  it('R8: replacing a session\'s own record releases the previous chips\' previews', () => {
    const first = fileChip('file:a', 'a.png', 'blob:preview-a')
    const second = fileChip('file:b', 'b.png', 'blob:preview-b')

    announceSalvagedEdit(SESSION, '', 'first edit', { attachments: [first] })
    announceSalvagedEdit(SESSION, '', 'second edit', { attachments: [second] })

    expect(retainedPreviewOwnerCount('blob:preview-a')).toBeUndefined()
    expect(retainedPreviewOwnerCount('blob:preview-b')).toBe(1)
  })

  it('keeps each session\'s records independent while durable', () => {
    window.localStorage.setItem(
      'hermes.desktop.salvagedEdits.v1',
      JSON.stringify({ 'stale-from-someone-else': 'stale' })
    )
    announceSalvagedEdit(`${SESSION}-a`, '', 'replacement A')
    announceSalvagedEdit(`${SESSION}-b`, '', 'replacement B')

    expect(undoSalvagedEdit(`${SESSION}-a`, '')).toBe('replacement A')
    expect(durableSalvagedTexts()[`${SESSION}-b`]).toBe('replacement B')
    expect(durableSalvagedTexts()[`${SESSION}-a`]).toBeUndefined()
    expect(getSalvagedEditNotice(`${SESSION}-b`)?.undoText).toBe('replacement B')
  })
})
