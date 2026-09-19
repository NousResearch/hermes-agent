import { describe, expect, it } from 'vitest'

import {
  extrasFromSessionsConfig,
  mergedSidebarRecentsExclude,
  resolveSidebarRecentsExclude,
  SIDEBAR_EXCLUDED_SOURCES
} from './sidebar-excluded-sources'

describe('mergedSidebarRecentsExclude', () => {
  it('defaults extras to a2a on top of the built-in recents denylist', () => {
    const merged = mergedSidebarRecentsExclude(undefined)

    expect(merged).toEqual(expect.arrayContaining(['a2a', 'cron', 'kanban']))
    expect(merged.indexOf('cron')).toBeLessThan(merged.indexOf('a2a'))
  })

  it('lets an empty extras array restore today\'s built-in-only list', () => {
    const merged = mergedSidebarRecentsExclude([])

    expect(merged).toEqual(SIDEBAR_EXCLUDED_SOURCES)
    expect(merged).not.toContain('a2a')
    expect(merged).toContain('cron')
  })

  it('replaces the default extra when a custom array is provided', () => {
    const merged = mergedSidebarRecentsExclude(['custom'])

    expect(merged).toContain('cron')
    expect(merged).toContain('custom')
    expect(merged).not.toContain('a2a')
  })

  it('keeps a2a when the provided array includes it', () => {
    expect(mergedSidebarRecentsExclude(['a2a'])).toContain('a2a')
  })

  it('dedupes extras that already appear in the built-in list', () => {
    const merged = mergedSidebarRecentsExclude(['cron', 'custom'])

    expect(merged.filter(source => source === 'cron')).toHaveLength(1)
    expect(merged).toContain('custom')
  })
})

describe('extrasFromSessionsConfig', () => {
  it('defaults to a2a when the key is missing, null, or not an array', () => {
    expect(extrasFromSessionsConfig(undefined)).toEqual(['a2a'])
    expect(extrasFromSessionsConfig(null)).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({})).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({ sessions: null })).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: null } })).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: 'a2a' } })).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: 1 } })).toEqual(['a2a'])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: { a2a: true } } })).toEqual(['a2a'])
  })

  it('uses the provided array, including empty, after dropping blank/non-string entries', () => {
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: [] } })).toEqual([])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: ['custom'] } })).toEqual(['custom'])
    expect(extrasFromSessionsConfig({ sessions: { exclude_sources: ['a2a', '', 1, '  ', 'foo'] } })).toEqual([
      'a2a',
      'foo'
    ])
  })
})

describe('resolveSidebarRecentsExclude', () => {
  it('still excludes a2a when config fetch fails', async () => {
    const merged = await resolveSidebarRecentsExclude(async () => {
      throw new Error('config unavailable')
    })

    expect(merged).toContain('a2a')
    expect(merged).toContain('cron')
  })

  it('honors an explicit empty exclude_sources array from config', async () => {
    const merged = await resolveSidebarRecentsExclude(async () => ({ sessions: { exclude_sources: [] } }))

    expect(merged).not.toContain('a2a')
    expect(merged).toContain('cron')
  })
})
