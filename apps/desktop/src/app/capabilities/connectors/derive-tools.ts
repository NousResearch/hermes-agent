// The pure layer inside an opened connector: filtering, counting, and the quick
// actions. Split out of `derive.ts` (which re-exports it) because the page and
// the tool editor are two different subjects and one file for both would be
// twice the size anyone wants to read.

import { FACET_ORDER, HINT_ORDER } from './hint-vocabulary'
import type {
  ConflictDifference,
  QuickAction,
  QuickActionId,
  ToolInput,
  ToolRowModel,
  ToolsEditorCounts,
  ToolsFilter
} from './types'

/** The bucket for tools the provider tagged with no category. A sentinel rather
 *  than the word, because it travels as a select VALUE beside real provider tags
 *  (lower-case underscore slugs) and must never collide with one. Its label
 *  comes from i18n like every other visible word. */
export const UNCATEGORISED = '__uncategorised__'

/** Under this many tools, the chrome costs more than it saves: a person can read
 *  the whole list faster than they can compose a filter for it. */
export const TINY_CONNECTOR_MAX = 8

export const EMPTY_TOOLS_FILTER: ToolsFilter = {
  category: null,
  facet: null,
  hint: null,
  query: '',
  showDeprecated: false
}

/** A connector small enough to drop search, chips, the category select and the
 *  quick actions. Derived from the list, never switched on per connector. */
export function isTinyConnector(tools: readonly ToolRowModel[]): boolean {
  return tools.length <= TINY_CONNECTOR_MAX
}

/** The provider's tags are lower-case underscore slugs; they are shown as words,
 *  and the raw tag stays the filter value so matching still speaks the wire. */
export function categoryLabel(name: string): string {
  return name.replace(/_/g, ' ')
}

export function toolInCategory(tool: ToolRowModel, category: string): boolean {
  return category === UNCATEGORISED ? tool.categories.length === 0 : tool.categories.includes(category)
}

/** Client-side substring over slug and name, no debounce: the list is already in
 *  memory, so a delay would only make typing feel slower than it is. */
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

/** Facet chips render only the facets present, and only when at least two exist:
 *  a lone chip filters to the list you are already looking at. Deprecated tools
 *  are counted only while they are on screen, so a count always matches the rows.
 */
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

/** Two hints only restate the facet beside them, and a chip row that offers both
 *  `Read` and `Read only` — or `Destructive` twice — asks the reader to tell two
 *  identical filters apart. They keep their words everywhere else; they just do
 *  not get a chip. */
const FACET_ECHO_HINTS: readonly string[] = ['destructiveHint', 'readOnlyHint']

/** Hint chips filter too, so they follow the same "only when it narrows" rule. */
export function hintChips(tools: readonly ToolRowModel[]): CountedValue[] {
  const counts = ordered(
    countBy(tools, tool => tool.hints.filter(hint => !FACET_ECHO_HINTS.includes(hint))),
    HINT_ORDER
  )

  return counts.length >= 2 ? counts : []
}

/** Categories by count, with an `Uncategorised` bucket only when the connector
 *  has categories AND some tools have none. A tool with two categories is
 *  counted under both, and matches either in the filter. */
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

// ------------------------------------------------------------ quick actions

/** De-duplication precedence, not display order. When two actions expand to the
 *  same list the earlier one wins, so a connector with no write facet keeps
 *  `Turn off destructive`: both presses do the same thing, and the narrower
 *  promise is the honest one to print on the button. */
export const QUICK_ACTIONS = [
  { facets: ['destructive'], id: 'no-destructive' },
  { facets: ['destructive', 'write'], id: 'read-only' },
  { facets: [], id: 'everything-on' }
] as const satisfies readonly QuickAction[]

const QUICK_ACTION_DISPLAY: readonly QuickActionId[] = ['read-only', 'no-destructive', 'everything-on']

export function quickActionById(id: QuickActionId): QuickAction {
  return QUICK_ACTIONS.find(action => action.id === id)!
}

/** A tool a quick action must never touch: the provider gave no behaviour hint,
 *  or it marked the tool deprecated. Both stay exactly as the person left them. */
export function isUntouchedByQuickActions(tool: ToolRowModel): boolean {
  return tool.deprecated || tool.facet === 'unclassified'
}

/** A quick action expands to a concrete list over the tools loaded now.
 *  Deprecated and unclassified tools are excluded, and so is anything the org
 *  already took away — a personal rule must not restate the org's. */
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

/** The actions worth offering for this connector. An action that expands to
 *  nothing is left out, and so is one that expands to exactly what an action
 *  already listed does. `Everything on` survives both rules: its empty expansion
 *  is not "nothing to do", it is "turn the list back on". */
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

/** Which quick action, if any, the current list matches. Only the part of the
 *  list a quick action could have written is compared, so a hand-toggled
 *  unclassified or deprecated tool does not hide the action's name.
 *
 *  Two actions can expand to the same list, so `prefer` carries the id of the
 *  action the person actually pressed; without it the first definition would win
 *  and rename their choice under them. */
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

    // An empty expansion only ever means `Everything on`: for the narrowing
    // actions it means "this connector has no such tools", not "you pressed it".
    return action.id === 'everything-on'
      ? comparable.length === 0
      : expansion.length > 0 && sameSet(comparable, expansion)
  })

  if (matches.length === 0) {
    return null
  }

  return matches.find(action => action.id === prefer) ?? matches[0]
}

// ------------------------------------------------------------ editor counts

/** What the dirty footer says, as two numbers. A delta, not a total: the person
 *  is about to write a change, and the change is what they need to check. */
export function editorCounts(local: readonly string[], baseline: readonly string[]): ToolsEditorCounts {
  const before = new Set(baseline)
  const after = new Set(local)

  return {
    backOn: baseline.filter(slug => !after.has(slug)).length,
    off: local.filter(slug => !before.has(slug)).length
  }
}

/** What the other editor's saved version does that this one does not. Named,
 *  because "keep mine" then "save" is an overwrite, and the reader is entitled to
 *  know what it costs before they press it. */
export function conflictDifference(theirs: readonly string[], mine: readonly string[]): ConflictDifference {
  const mineSet = new Set(mine)
  const theirSet = new Set(theirs)

  return {
    theyOff: theirs.filter(slug => !mineSet.has(slug)).length,
    theyOn: mine.filter(slug => !theirSet.has(slug)).length
  }
}

/** Wire tools plus the two policy lists become the rows the editor renders. */
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
