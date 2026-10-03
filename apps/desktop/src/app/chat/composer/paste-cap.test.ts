import { afterEach, describe, expect, it, vi } from 'vitest'

import { MAX_COMPOSER_TEXT_CHARS } from '@/store/composer'

import { onComposerInsertRequest } from './focus'
import { preparePastedText, routeClipboardToComposer } from './paste-to-focus'

/** Minimal DataTransfer stand-in: text/plain only (the cap path never sees files). */
function clipboard({ text = '' } = {}): DataTransfer {
  return {
    files: { item: () => null, length: 0 },
    getData: (type: string) => (type === 'text' || type === 'text/plain' ? text : ''),
    items: []
  } as unknown as DataTransfer
}

/** The bus defers dispatch a macrotask; flush it. */
const flushBus = () => new Promise(resolve => setTimeout(resolve, 1))

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

// #98562: a multi-megabyte paste used to run linkifyUrls/pathifyRefs and the
// chip-DOM build synchronously over the whole blob on the UI thread, freezing
// the app — and the resulting draft rehydrated on the next launch to freeze
// it again. preparePastedText clamps oversized input to a plain-text prefix
// instead; normal-sized pastes keep the chipping behavior unchanged.
describe('preparePastedText', () => {
  it('chips links in a normal-sized paste', () => {
    expect(preparePastedText('see https://example.com/docs')).toContain('@url:')
  })

  it('clamps an oversized paste to a plain-text prefix without chipping', () => {
    const oversized = `${'a'.repeat(MAX_COMPOSER_TEXT_CHARS)}https://example.com/never-reached`

    const prepared = preparePastedText(oversized)

    expect(prepared).toBe('a'.repeat(MAX_COMPOSER_TEXT_CHARS))
    expect(prepared).not.toContain('@url:')
  })

  it('still chips a paste exactly at the cap', () => {
    const tail = ' https://example.com/x'
    const atCap = `${'a'.repeat(MAX_COMPOSER_TEXT_CHARS - tail.length)}${tail}`

    expect(atCap).toHaveLength(MAX_COMPOSER_TEXT_CHARS)
    expect(preparePastedText(atCap)).toContain('@url:')
  })
})

describe('routeClipboardToComposer oversized paste', () => {
  it('inserts the clamped plain-text prefix instead of the full blob', async () => {
    const inserts: string[] = []
    const off = onComposerInsertRequest(({ text }) => inserts.push(text))

    const oversized = `${'b'.repeat(MAX_COMPOSER_TEXT_CHARS)}https://example.com/later`

    expect(routeClipboardToComposer(clipboard({ text: oversized }))).toBe(true)
    await flushBus()
    off()

    expect(inserts).toHaveLength(1)
    expect(inserts[0]).toBe('b'.repeat(MAX_COMPOSER_TEXT_CHARS))
    expect(inserts[0]).not.toContain('@url:')
  })

  it('keeps link chipping for a normal-sized window paste', async () => {
    const inserts: string[] = []
    const off = onComposerInsertRequest(({ text }) => inserts.push(text))

    routeClipboardToComposer(clipboard({ text: 'see https://example.com/docs' }))
    await flushBus()
    off()

    expect(inserts[0]).toContain('@url:')
  })
})
