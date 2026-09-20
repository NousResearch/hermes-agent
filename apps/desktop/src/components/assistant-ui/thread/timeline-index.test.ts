import { beforeEach, describe, expect, it, vi } from 'vitest'

const { api } = vi.hoisted(() => ({ api: vi.fn() }))
vi.mock('@/api/client', () => ({
  capabilityScoped: (scope: object) => scope,
  hermesApi: api
}))

beforeEach(() => {
  vi.resetModules()
  api.mockReset()
})

const page = (id = 1, more = false) => ({
  entries: [{ row_id: id, preview: `Prompt ${id}` }],
  pagination: { next_cursor: more ? id : null, has_more: more }
})

describe('timeline metadata index', () => {
  it('fetches one lightweight page and coalesces concurrent callers', async () => {
    api.mockResolvedValue(page(1, true))
    const { fetchTimelineIndex } = await import('./timeline-index')
    const scope = { connectionId: 'local', profile: 'default' }
    const a = fetchTimelineIndex('session', scope)
    const b = fetchTimelineIndex('session', scope)
    expect(a).toBe(b)
    expect((await a).entries).toEqual([{ id: 'history:1', rowId: 1, preview: 'Prompt 1' }])
    expect(api).toHaveBeenCalledTimes(1)
    expect(api.mock.calls[0][0]).toMatchObject({ connectionId: 'local', profile: 'default', passive: true })
    expect(api.mock.calls[0][0].path).toContain('/timeline?limit=500')
    expect(api.mock.calls[0][0].path).not.toContain('/messages')
  })

  it('continues only on demand and deduplicates overlapping metadata', async () => {
    api.mockResolvedValueOnce(page(1, true)).mockResolvedValueOnce({
      entries: [
        { row_id: 1, preview: 'Prompt 1' },
        { row_id: 2, preview: 'Prompt 2' }
      ],
      pagination: { next_cursor: null, has_more: false }
    })
    const { fetchTimelineIndex } = await import('./timeline-index')
    await fetchTimelineIndex('session', 'default')
    expect(api).toHaveBeenCalledTimes(1)
    const result = await fetchTimelineIndex('session', 'default')
    expect(result.entries.map(e => e.rowId)).toEqual([1, 2])
    expect(result.complete).toBe(true)
    expect(api.mock.calls[1][0].path).toContain('after_row_id=1')
    await fetchTimelineIndex('session', 'default')
    expect(api).toHaveBeenCalledTimes(2)
  })

  it('isolates identical session IDs by owning connection and profile', async () => {
    api.mockResolvedValueOnce(page(1)).mockResolvedValueOnce(page(2))
    const { fetchTimelineIndex } = await import('./timeline-index')
    const a = await fetchTimelineIndex('same', { connectionId: 'local', profile: 'one' })
    const b = await fetchTimelineIndex('same', { connectionId: 'remote', profile: 'two' })
    expect(a.entries[0].rowId).toBe(1)
    expect(b.entries[0].rowId).toBe(2)
  })

  it('clears failed requests so an explicit retry works without caching a false empty result', async () => {
    api.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(page())
    const { fetchTimelineIndex } = await import('./timeline-index')
    await expect(fetchTimelineIndex('session', 'default')).rejects.toThrow('offline')
    expect((await fetchTimelineIndex('session', 'default')).complete).toBe(true)
  })

  it('names the mark before an anchor, paging a partial index only as far as one lookup may', async () => {
    api
      .mockResolvedValueOnce({
        entries: [{ row_id: 10, preview: 'Prompt 10' }, { row_id: 20, preview: 'Prompt 20' }],
        pagination: { next_cursor: 20, has_more: true }
      })
      .mockResolvedValueOnce({
        entries: [{ row_id: 30, preview: 'Prompt 30' }, { row_id: 40, preview: 'Prompt 40' }],
        pagination: { next_cursor: null, has_more: false }
      })
      .mockResolvedValue(page(1, true))
    const { previousPromptRowId } = await import('./timeline-index')

    // The anchor sits past the first page; the lookup advances the shared index.
    expect(await previousPromptRowId('session', 'default', 40)).toBe(30)
    expect(api).toHaveBeenCalledTimes(2)
    // Complete index: a mark before it needs no further request.
    expect(await previousPromptRowId('session', 'default', 20)).toBe(10)
    expect(api).toHaveBeenCalledTimes(2)
    // The oldest mark has nothing before it, and an unknown anchor is never guessed.
    expect(await previousPromptRowId('session', 'default', 10)).toBeNull()
    expect(await previousPromptRowId('session', 'default', undefined)).toBeNull()
    expect(api).toHaveBeenCalledTimes(2)
    // A session whose index never completes nor covers the anchor: the lookup
    // stops at its per-lookup ceiling instead of walking the whole session.
    expect(await previousPromptRowId('endless', 'default', 99_999)).toBe(1)
    expect(api).toHaveBeenCalledTimes(5)
  })
})
