import { describe, expect, it } from 'vitest'

import { cardFace, splitMetaBlock, splitSyncTitle } from './card-face'

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
  it('reads title and body through both decorations at once', () => {
    expect(cardFace({ body: `${BLOCK}\n\nReadable body`, title: '[Sync conflict] Readable title' })).toEqual({
      body: 'Readable body',
      meta: [META_LINE],
      syncState: 'conflict',
      title: 'Readable title'
    })
  })
})
