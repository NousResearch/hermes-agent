import { beforeEach, describe, expect, it } from 'vitest'

import {
  aliasMediaCardMeta,
  mediaCardMeta,
  type MediaDeliverableMeta,
  pruneMediaDeliverables,
  recordMediaDeliverable,
  resetMediaDeliverables
} from './media-store'

const row: MediaDeliverableMeta = {
  kind: 'image',
  mime: 'image/png',
  origin: 'serve',
  path: '/tmp/hermes-media/clip.png',
  receivedAt: 1,
  size: 1234
}

describe('media deliverable registry (M4)', () => {
  beforeEach(() => {
    resetMediaDeliverables()
  })

  it('records rows and drops invalid payloads', () => {
    expect(recordMediaDeliverable({}, 1)).toBe(false)
    expect(recordMediaDeliverable({ path: '' }, 1)).toBe(false)
    expect(recordMediaDeliverable({ path: 42 }, 1)).toBe(false)
    expect(recordMediaDeliverable(null, 1)).toBe(false)

    expect(recordMediaDeliverable(row, 1)).toBe(true)
    expect(mediaCardMeta(row.path)).toEqual(row)
  })

  it('downgrades unknown kinds to file', () => {
    expect(recordMediaDeliverable({ kind: 'hologram', path: '/tmp/x.bin' }, 1)).toBe(true)
    expect(mediaCardMeta('/tmp/x.bin')?.kind).toBe('file')
  })

  it('dedupes by path and keeps the newest row', () => {
    expect(recordMediaDeliverable(row, 1)).toBe(true)
    expect(recordMediaDeliverable({ ...row, size: 999 }, 2)).toBe(true)
    expect(mediaCardMeta(row.path)?.size).toBe(999)
    expect(mediaCardMeta(row.path)?.receivedAt).toBe(2)
  })

  it('aliases file:// transcriptions to the raw ref', () => {
    expect(recordMediaDeliverable(row, 1)).toBe(true)

    aliasMediaCardMeta('file:///tmp/hermes-media/clip.png', row.path)

    expect(mediaCardMeta('file:///tmp/hermes-media/clip.png')).toEqual(row)
    expect(aliasMediaCardMeta('file:///nope.png', '/nope.png')).toBeUndefined()
  })

  it('prunes by counted path allowlist, keeping aliased spellings of kept paths', () => {
    const other = { ...row, path: '/tmp/hermes-media/other.mp4' }

    recordMediaDeliverable(row, 1)
    recordMediaDeliverable(other, 2)
    aliasMediaCardMeta('file:///tmp/hermes-media/clip.png', row.path)

    pruneMediaDeliverables([row.path])

    expect(mediaCardMeta(row.path)).toEqual(row)
    expect(mediaCardMeta('file:///tmp/hermes-media/clip.png')).toEqual(row)
    expect(mediaCardMeta(other.path)).toBeNull()
  })

  it('prunes everything when called with no allowlist', () => {
    recordMediaDeliverable(row, 1)
    pruneMediaDeliverables()
    expect(mediaCardMeta(row.path)).toBeNull()
  })
})
