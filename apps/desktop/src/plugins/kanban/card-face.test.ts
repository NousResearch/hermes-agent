import { describe, expect, it } from 'vitest'

import { cardFace, splitMetaBlock, splitSyncTitle, workSummary } from './card-face'

// The exact presentation the fleet sync adapter stamps onto a local task
// (conductors/scripts/fleet_kanban_remote.py: SYNC_PREFIXES + render_meta_block).
const META_LINE =
  '> Fleet: revision 6 | point Conductor: none | campaign: none | repository: none | canonical status: ready'

const BLOCK = `<!-- fleet-kanban:meta -->\n${META_LINE}\n<!-- /fleet-kanban:meta -->`

describe('splitSyncTitle', () => {
  it.each([
    ['pending', '[Sync pending] '],
    ['conflict', '[Sync conflict] '],
    ['error', '[Sync error] ']
  ])('lifts the %s prefix off the readable title', (state, prefix) => {
    expect(splitSyncTitle(`${prefix}CANARY TB create`)).toEqual({ syncState: state, title: 'CANARY TB create' })
  })

  it('leaves an undecorated title alone', () => {
    expect(splitSyncTitle('CANARY TB create')).toEqual({ syncState: null, title: 'CANARY TB create' })
  })

  it('only recognises the prefix at the very start', () => {
    const title = 'Note: [Sync pending] is a real phrase here'

    expect(splitSyncTitle(title)).toEqual({ syncState: null, title })
  })
})

describe('splitMetaBlock', () => {
  it('lifts the block and its blank-line separator off the readable body', () => {
    expect(splitMetaBlock(`${BLOCK}\n\nRun the rotation from the node itself.`)).toEqual({
      body: 'Run the rotation from the node itself.',
      meta: [META_LINE]
    })
  })

  it('tolerates a single newline separator', () => {
    expect(splitMetaBlock(`${BLOCK}\nShort body`)).toEqual({ body: 'Short body', meta: [META_LINE] })
  })

  it('yields no body for a block-only body (the live canary rows)', () => {
    expect(splitMetaBlock(BLOCK)).toEqual({ body: null, meta: [META_LINE] })
  })

  it('leaves a body without the closing marker untouched', () => {
    const body = '<!-- fleet-kanban:meta -->\nhalf a block'

    expect(splitMetaBlock(body)).toEqual({ body, meta: [] })
  })

  it('leaves an undecorated body alone and normalises empty to null', () => {
    expect(splitMetaBlock('Plain description')).toEqual({ body: 'Plain description', meta: [] })
    expect(splitMetaBlock(null)).toEqual({ body: null, meta: [] })
    expect(splitMetaBlock(undefined)).toEqual({ body: null, meta: [] })
    expect(splitMetaBlock('')).toEqual({ body: null, meta: [] })
  })
})

describe('cardFace', () => {
  const task = { body: `${BLOCK}\n\nReadable body`, title: '[Sync conflict] Readable title' }

  it('reads title and body through both decorations at once on the fleet board', () => {
    expect(cardFace(task, true)).toEqual({
      body: 'Readable body',
      meta: [META_LINE],
      summary: 'Readable body',
      syncState: 'conflict',
      title: 'Readable title'
    })
  })

  it('shows the same task literally off the fleet board', () => {
    expect(cardFace(task, false)).toEqual({
      body: task.body,
      meta: [],
      summary: task.body,
      syncState: null,
      title: task.title
    })
  })

  it('prefers a real work summary over the body, never the administrative note', () => {
    expect(cardFace({ ...task, latest_summary: 'Rotated the canary; PR #12 opened.' }, true).summary).toBe(
      'Rotated the canary; PR #12 opened.'
    )
    expect(cardFace({ ...task, latest_summary: 'status changed to todo (dashboard/direct)' }, true).summary).toBe(
      'Readable body'
    )
    expect(workSummary('status changed to ready (dashboard/direct)')).toBeNull()
    expect(workSummary(null)).toBeNull()
  })
})
