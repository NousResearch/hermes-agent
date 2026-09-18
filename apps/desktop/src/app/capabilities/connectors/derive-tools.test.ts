import { describe, expect, it } from 'vitest'

import {
  availableQuickActions,
  categoryCounts,
  conflictDifference,
  deprecatedCount,
  editorCounts,
  EMPTY_TOOLS_FILTER,
  expandQuickAction,
  facetChips,
  filterTools,
  hintChips,
  isTinyConnector,
  isUntouchedByQuickActions,
  matchingQuickAction,
  quickActionById,
  toolRows,
  UNCATEGORISED
} from './derive-tools'
import { NO_WRITE_TOOLS, ORG_DISABLED, TINY_TOOLS, toolFixtures, TOOLS } from './fixtures'

const tools = toolFixtures()
const slugs = (rows: { slug: string }[]) => rows.map(row => row.slug)

describe('filtering tools', () => {
  it('hides deprecated tools until they are asked for', () => {
    expect(slugs(filterTools(tools, EMPTY_TOOLS_FILTER))).not.toContain('LINEAR_SYNC_LEGACY')
    expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, showDeprecated: true }))).toContain('LINEAR_SYNC_LEGACY')
  })

  it('matches the query against the slug and the name, with no debounce to wait for', () => {
    expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, query: 'delete_comment' }))).toEqual([
      'LINEAR_DELETE_COMMENT'
    ])
    expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, query: 'comment on' }))).toEqual(['LINEAR_ADD_COMMENT'])
  })

  it('filters by facet and by hint', () => {
    expect(filterTools(tools, { ...EMPTY_TOOLS_FILTER, facet: 'destructive' })).toHaveLength(3)
    expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, hint: 'openWorldHint' })).sort()).toEqual([
      'LINEAR_ATTACH_LINK',
      'LINEAR_RUN_WEBHOOK'
    ])
  })

  it('lets a tool in two categories match either one', () => {
    for (const category of ['issues', 'search']) {
      expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, category }))).toContain('LINEAR_SEARCH_ISSUES')
    }
  })

  it('collects the tools with no category into one bucket', () => {
    expect(slugs(filterTools(tools, { ...EMPTY_TOOLS_FILTER, category: UNCATEGORISED })).sort()).toEqual([
      'LINEAR_PING',
      'LINEAR_RUN_WEBHOOK'
    ])
  })
})

describe('counting what the chips offer', () => {
  it('counts a facet once per tool and orders them least to most costly', () => {
    expect(facetChips(tools)).toEqual([
      { count: 4, value: 'read' },
      { count: 5, value: 'write' },
      { count: 3, value: 'destructive' },
      { count: 2, value: 'unclassified' }
    ])
  })

  it('does not offer a hint chip that only restates a facet chip', () => {
    const values = hintChips(tools).map(entry => entry.value)

    expect(values).not.toContain('readOnlyHint')
    expect(values).not.toContain('destructiveHint')
    expect(values).toEqual(['createHint', 'updateHint', 'deleteHint', 'idempotentHint', 'openWorldHint'])
  })

  it('offers no chips when a single value would filter to the list you are on', () => {
    expect(facetChips(toolRows(TINY_TOOLS, new Set()))).toEqual([])
    expect(hintChips(toolRows(TINY_TOOLS, new Set()))).toEqual([])
  })

  it('adds an Uncategorised bucket only when some tools have none', () => {
    const counts = categoryCounts(tools)

    expect(counts.at(-1)).toEqual({ count: 2, value: UNCATEGORISED })
    expect(counts.find(entry => entry.value === 'issues')).toEqual({ count: 7, value: 'issues' })
  })

  it('offers no category picker at all when the connector has no categories', () => {
    expect(categoryCounts(toolRows(TINY_TOOLS, new Set()))).toEqual([])
  })

  it('counts the deprecated tools the toggle names', () => {
    expect(deprecatedCount(tools)).toBe(1)
    expect(deprecatedCount(toolRows(TINY_TOOLS, new Set()))).toBe(0)
  })
})

describe('a tiny connector drops its chrome', () => {
  it('draws the line at eight tools', () => {
    expect(isTinyConnector(toolRows(TINY_TOOLS, new Set()))).toBe(true)
    expect(isTinyConnector(tools)).toBe(false)
    expect(isTinyConnector(tools.slice(0, 8))).toBe(true)
    expect(isTinyConnector(tools.slice(0, 9))).toBe(false)
  })
})

describe('quick actions', () => {
  const readOnly = quickActionById('read-only')
  const noDestructive = quickActionById('no-destructive')

  it('never touches a tool with an unknown effect or a deprecated one', () => {
    for (const action of [readOnly, noDestructive]) {
      const expansion = expandQuickAction(action, tools)

      expect(expansion).not.toContain('LINEAR_RUN_WEBHOOK')
      expect(expansion).not.toContain('LINEAR_PING')
      expect(expansion).not.toContain('LINEAR_SYNC_LEGACY')
    }

    expect(
      tools
        .filter(isUntouchedByQuickActions)
        .map(tool => tool.slug)
        .sort()
    ).toEqual(['LINEAR_PING', 'LINEAR_RUN_WEBHOOK', 'LINEAR_SYNC_LEGACY'])
  })

  it('never writes a rule the organisation already wrote', () => {
    for (const slug of ORG_DISABLED) {
      expect(expandQuickAction(readOnly, tools)).not.toContain(slug)
      expect(expandQuickAction(noDestructive, tools)).not.toContain(slug)
    }
  })

  it('expands to a concrete list over the tools loaded now', () => {
    expect(expandQuickAction(noDestructive, tools).sort()).toEqual(['LINEAR_ARCHIVE_ISSUE', 'LINEAR_DELETE_COMMENT'])
    expect(expandQuickAction(readOnly, tools)).toHaveLength(6)
  })

  it('offers all three actions when each promises something different', () => {
    expect(availableQuickActions(tools).map(action => action.id)).toEqual([
      'read-only',
      'no-destructive',
      'everything-on'
    ])
  })

  it('de-duplicates Read only away when the connector has no write facet', () => {
    const rows = toolRows(NO_WRITE_TOOLS, new Set())

    expect(expandQuickAction(readOnly, rows)).toEqual(expandQuickAction(noDestructive, rows))
    expect(availableQuickActions(rows).map(action => action.id)).toEqual(['no-destructive', 'everything-on'])
  })

  it('keeps only Everything on when no narrowing has anything to narrow', () => {
    expect(availableQuickActions(toolRows(TINY_TOOLS, new Set())).map(action => action.id)).toEqual(['everything-on'])
  })

  it('names the action the person pressed, not the first that fits', () => {
    const rows = toolRows(NO_WRITE_TOOLS, new Set())
    const expansion = expandQuickAction(readOnly, rows)

    expect(matchingQuickAction(expansion, rows, 'read-only')?.id).toBe('read-only')
    expect(matchingQuickAction(expansion, rows, 'no-destructive')?.id).toBe('no-destructive')
  })

  it('ignores hand-toggled tools a quick action could never have written', () => {
    const expansion = expandQuickAction(noDestructive, tools)

    expect(matchingQuickAction([...expansion, 'LINEAR_PING'], tools)?.id).toBe('no-destructive')
  })

  it('reads an empty list as Everything on, never as a narrowing with nothing to do', () => {
    expect(matchingQuickAction([], tools)?.id).toBe('everything-on')
    expect(matchingQuickAction(['LINEAR_CREATE_ISSUE'], tools)).toBeNull()
  })
})

describe('what the footer counts', () => {
  it('reports the change, not the total', () => {
    expect(editorCounts(['a', 'b', 'c'], ['a'])).toEqual({ backOn: 0, off: 2 })
    expect(editorCounts(['a'], ['a', 'b'])).toEqual({ backOn: 1, off: 0 })
    expect(editorCounts(['b'], ['a'])).toEqual({ backOn: 1, off: 1 })
    expect(editorCounts(['a'], ['a'])).toEqual({ backOn: 0, off: 0 })
  })

  it('names what the other version does in both directions', () => {
    expect(conflictDifference(['a', 'b'], ['b', 'c'])).toEqual({ theyOff: 1, theyOn: 1 })
    expect(conflictDifference([], ['a'])).toEqual({ theyOff: 0, theyOn: 1 })
  })
})

describe('turning wire tools into rows', () => {
  it('strikes what the organisation took away and leaves it off', () => {
    const locked = tools.find(tool => tool.slug === 'LINEAR_DELETE_PROJECT')

    expect(locked).toMatchObject({ lockedBy: 'org', on: false })
  })

  it('reads the personal disabled list for everything else', () => {
    const rows = toolFixtures(['LINEAR_CREATE_ISSUE'])

    expect(rows.find(tool => tool.slug === 'LINEAR_CREATE_ISSUE')?.on).toBe(false)
    expect(rows.find(tool => tool.slug === 'LINEAR_LIST_ISSUES')?.on).toBe(true)
    expect(rows).toHaveLength(TOOLS.length)
  })
})
