import { afterEach, describe, expect, it, vi } from 'vitest'

/**
 * The shared emoji catalog: one loader for the composer's `:shortcode:`
 * completions AND the session stamp picker's Emoji panel, so these cases are
 * about the two things both surfaces inherit — what a query matches, and that a
 * failed load is retried instead of caching an empty catalog for the session.
 */

const DATA = [
  { emoji: '😂', hexcode: '1F602', label: 'Face with tears of joy', tags: ['laugh'] },
  { emoji: '🐙', hexcode: '1F419', label: 'Octopus', tags: ['animal'] }
]

const SHORTCODES: Record<string, string | string[]> = { '1F602': ['joy', 'laughing'], '1F419': 'octopus' }

/** A fresh module registry per case, with the bundled files served from memory. */
async function loadIndex() {
  vi.resetModules()

  const fetchMock = vi.fn((url: string) =>
    Promise.resolve({
      json: () => Promise.resolve(url.includes('shortcodes') ? SHORTCODES : DATA)
    } as Response)
  )

  vi.stubGlobal('fetch', fetchMock)

  return { module: await import('./emoji-index'), fetchMock }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('searchEmoji', () => {
  it('ranks a shortcode prefix first and still answers from tags and labels', async () => {
    const { module } = await loadIndex()

    expect((await module.searchEmoji('oct')).map(entry => entry.emoji)).toEqual(['🐙'])
    // A tag or label substring is a hit too, so an emoji is reachable by the word
    // a user actually thinks of ("laugh" → 😂, not "joy").
    expect((await module.searchEmoji('laugh')).map(entry => entry.emoji)).toContain('😂')
    expect(await module.searchEmoji('zzzz')).toEqual([])
  })

  it('never hands back more than the asked-for budget', async () => {
    const { module } = await loadIndex()

    expect(await module.searchEmoji('', 1)).toHaveLength(1)
    // The panel asks for a bounded grid, not the whole catalog.
    expect((await module.searchEmoji('', 48)).length).toBeLessThanOrEqual(48)
  })

  it('loads the catalog once, and reports that it is in memory', async () => {
    const { module, fetchMock } = await loadIndex()

    expect(module.isEmojiIndexLoaded()).toBe(false)

    await module.searchEmoji('oct')

    expect(module.isEmojiIndexLoaded()).toBe(true)
    const reads = fetchMock.mock.calls.length

    await module.searchEmoji('joy')

    expect(fetchMock.mock.calls.length).toBe(reads)
  })

  it('retries a failed load rather than caching an empty catalog', async () => {
    vi.resetModules()

    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('offline'))
      .mockImplementation((url: string) =>
        Promise.resolve({
          json: () => Promise.resolve(url.includes('shortcodes') ? SHORTCODES : DATA) } as Response)
      )

    vi.stubGlobal('fetch', fetchMock)

    const module = await import('./emoji-index')

    await expect(module.searchEmoji('oct')).rejects.toThrow('offline')
    expect(module.isEmojiIndexLoaded()).toBe(false)
    // The next query is a fresh attempt — a stamp panel that opened while the
    // backend was busy is not dead for the rest of the session.
    expect((await module.searchEmoji('oct')).map(entry => entry.emoji)).toEqual(['🐙'])
  })
})
