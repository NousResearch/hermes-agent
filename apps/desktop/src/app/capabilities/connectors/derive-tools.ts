import { FACET_ORDER, HINT_ORDER } from './hint-vocabulary'
import type {
  ConflictDifference,
  FacetSummaryRow,
  QuickAction,
  QuickActionId,
  ToolInput,
  ToolRowModel,
  ToolsEditorCounts,
  ToolsEditorStatus,
  ToolsFilter
} from './types'

export const UNCATEGORISED = '__uncategorised__'

export const TINY_CONNECTOR_MAX = 8

export const EMPTY_TOOLS_FILTER: ToolsFilter = {
  category: null,
  facet: null,
  hint: null,
  query: '',
  showDeprecated: false
}

export function isTinyConnector(tools: readonly ToolRowModel[]): boolean {
  return tools.length <= TINY_CONNECTOR_MAX
}

export function toolDisplayName(slug: string): string {
  const words = slug
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_.\-/]+/g, ' ')
    .trim()

  return words.charAt(0).toUpperCase() + words.slice(1)
}

export function categoryLabel(name: string): string {
  return name.replace(/_/g, ' ')
}

export function toolInCategory(tool: ToolRowModel, category: string): boolean {
  return category === UNCATEGORISED ? tool.categories.length === 0 : tool.categories.includes(category)
}

export function toolMatchesQuery(tool: ToolRowModel, query: string): boolean {
  const needle = query.trim().toLowerCase()

  return needle.length === 0 || tool.slug.toLowerCase().includes(needle) || tool.name.toLowerCase().includes(needle)
}

export function filterTools(tools: readonly ToolRowModel[], filter: ToolsFilter): ToolRowModel[] {
  return tools.filter(
    tool =>
      (filter.showDeprecated || !tool.deprecated) &&
      toolMatchesQuery(tool, filter.query) &&
      (filter.facet === null || tool.facet === filter.facet) &&
      (filter.hint === null || tool.hints.includes(filter.hint)) &&
      (filter.category === null || toolInCategory(tool, filter.category))
  )
}

export interface CountedValue {
  count: number
  value: string
}

function countBy(tools: readonly ToolRowModel[], pick: (tool: ToolRowModel) => readonly string[]): Map<string, number> {
  const counts = new Map<string, number>()

  for (const tool of tools) {
    for (const value of pick(tool)) {
      counts.set(value, (counts.get(value) ?? 0) + 1)
    }
  }

  return counts
}

function ordered(counts: Map<string, number>, order: readonly string[]): CountedValue[] {
  const known = order.filter(value => counts.has(value)).map(value => ({ count: counts.get(value)!, value }))

  const rest = [...counts.keys()]
    .filter(value => !order.includes(value))
    .sort()
    .map(value => ({ count: counts.get(value)!, value }))

  return [...known, ...rest]
}

export function facetCounts(tools: readonly ToolRowModel[]): CountedValue[] {
  return ordered(
    countBy(tools, tool => [tool.facet]),
    FACET_ORDER
  )
}

export function facetChips(tools: readonly ToolRowModel[]): CountedValue[] {
  const counts = facetCounts(tools)

  return counts.length >= 2 ? counts : []
}

const FACET_ECHO_HINTS: readonly string[] = ['destructiveHint', 'readOnlyHint']

export function hintChips(tools: readonly ToolRowModel[]): CountedValue[] {
  const counts = ordered(
    countBy(tools, tool => tool.hints.filter(hint => !FACET_ECHO_HINTS.includes(hint))),
    HINT_ORDER
  )

  return counts.length >= 2 ? counts : []
}

export function categoryCounts(tools: readonly ToolRowModel[]): CountedValue[] {
  const counts = countBy(tools, tool => tool.categories)

  if (counts.size === 0) {
    return []
  }

  const none = tools.filter(tool => tool.categories.length === 0).length

  const named = [...counts.entries()]
    .map(([value, count]) => ({ count, value }))
    .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value))

  return none > 0 ? [...named, { count: none, value: UNCATEGORISED }] : named
}

export function deprecatedCount(tools: readonly ToolRowModel[]): number {
  return tools.filter(tool => tool.deprecated).length
}

export interface ToolReadInput {
  hasData: boolean
  listHasApp: boolean
  pending: boolean
  reason: null | string
}

export function toolReadStatus({ hasData, listHasApp, pending, reason }: ToolReadInput): ToolsEditorStatus | null {
  if (reason === 'CONNECTOR_NOT_FOUND') {
    return listHasApp ? 'unavailable' : 'gone'
  }

  if (hasData) {
    return null
  }

  if (reason !== null) {
    return reason === 'NEEDS_NOUS_AUTH' ? 'signedOut' : 'unavailable'
  }

  return pending ? 'loading' : null
}

export function facetTools(tools: readonly ToolRowModel[], facet: string): string[] {
  return tools.filter(tool => tool.facet === facet && !tool.deprecated && tool.lockedBy === null).map(tool => tool.slug)
}

export function facetSummary(tools: readonly ToolRowModel[], isOn: (slug: string) => boolean): FacetSummaryRow[] {
  const live = tools.filter(tool => !tool.deprecated)

  return facetCounts(live).map(({ count, value }) => {
    const on = live.filter(tool => tool.facet === value && isOn(tool.slug)).length
    const switchable = facetTools(live, value).length

    return {
      facet: value,
      locked: switchable === 0,
      on,
      switchState: on === 0 ? 'off' : on === switchable ? 'on' : 'mixed',
      total: count
    }
  })
}

const plainWords = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()

export function describesNothingNew(tool: ToolRowModel): boolean {
  const description = plainWords(tool.description)

  return description === '' || description === plainWords(tool.name)
}

export const QUICK_ACTIONS = [
  { facets: ['destructive'], id: 'no-destructive' },
  { facets: ['destructive', 'write'], id: 'read-only' },
  { facets: [], id: 'everything-on' }
] as const satisfies readonly QuickAction[]

const QUICK_ACTION_DISPLAY: readonly QuickActionId[] = ['read-only', 'no-destructive', 'everything-on']

export function quickActionById(id: QuickActionId): QuickAction {
  return QUICK_ACTIONS.find(action => action.id === id)!
}

export function isUntouchedByQuickActions(tool: ToolRowModel): boolean {
  return tool.deprecated || tool.facet === 'unclassified'
}

export function expandQuickAction(action: QuickAction, tools: readonly ToolRowModel[]): string[] {
  return tools
    .filter(tool => !isUntouchedByQuickActions(tool) && action.facets.includes(tool.facet) && tool.lockedBy === null)
    .map(tool => tool.slug)
}

export function sameSet(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) {
    return false
  }

  const seen = new Set(a)

  return b.every(value => seen.has(value))
}

export function availableQuickActions(tools: readonly ToolRowModel[]): QuickAction[] {
  const taken: string[][] = []
  const out: QuickAction[] = []

  for (const action of QUICK_ACTIONS) {
    if (action.id === 'everything-on') {
      out.push(action)

      continue
    }

    const expansion = expandQuickAction(action, tools)

    if (expansion.length === 0 || taken.some(previous => sameSet(previous, expansion))) {
      continue
    }

    taken.push(expansion)
    out.push(action)
  }

  return out.sort((a, b) => QUICK_ACTION_DISPLAY.indexOf(a.id) - QUICK_ACTION_DISPLAY.indexOf(b.id))
}

export function matchingQuickAction(
  disabled: readonly string[],
  tools: readonly ToolRowModel[],
  prefer: QuickActionId | null = null
): QuickAction | null {
  const bySlug = new Map(tools.map(tool => [tool.slug, tool]))

  const comparable = disabled.filter(slug => {
    const tool = bySlug.get(slug)

    return tool !== undefined && !isUntouchedByQuickActions(tool)
  })

  const matches = QUICK_ACTIONS.filter(action => {
    const expansion = expandQuickAction(action, tools)

    return action.id === 'everything-on'
      ? comparable.length === 0
      : expansion.length > 0 && sameSet(comparable, expansion)
  })

  if (matches.length === 0) {
    return null
  }

  return matches.find(action => action.id === prefer) ?? matches[0]
}

export function editorCounts(local: readonly string[], baseline: readonly string[]): ToolsEditorCounts {
  const before = new Set(baseline)
  const after = new Set(local)

  return {
    backOn: baseline.filter(slug => !after.has(slug)).length,
    off: local.filter(slug => !before.has(slug)).length
  }
}

export function conflictDifference(theirs: readonly string[], mine: readonly string[]): ConflictDifference {
  const mineSet = new Set(mine)
  const theirSet = new Set(theirs)

  return {
    theyOff: theirs.filter(slug => !mineSet.has(slug)).length,
    theyOn: mine.filter(slug => !theirSet.has(slug)).length
  }
}

export function toolRows(
  tools: readonly ToolInput[],
  disabled: ReadonlySet<string>,
  orgDisabled: ReadonlySet<string> = new Set()
): ToolRowModel[] {
  return tools.map(tool => ({
    categories: tool.categories,
    deprecated: tool.deprecated,
    description: tool.description,
    facet: tool.facet,
    hints: tool.hints,
    lockedBy: orgDisabled.has(tool.slug) ? 'org' : null,
    name: tool.name,
    on: !orgDisabled.has(tool.slug) && !disabled.has(tool.slug),
    slug: tool.slug
  }))
}
