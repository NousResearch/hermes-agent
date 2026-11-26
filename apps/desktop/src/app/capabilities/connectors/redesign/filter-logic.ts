import { UNCATEGORISED } from '../derive-tools'
import type { ConnectorCardModel, ToolRowModel } from '../types'

import {
  cardIdentity,
  categoryOfCard,
  categoryOrderForMode,
  categoryPreviewLimit,
  type ConnectionCategoryId,
  type ConnectionCategoryKey,
  type ConnectionInterfaceMode,
  connectionRank,
  type ConnectionSubgroupId,
  isConnectedCard,
  subgroupOrderForCategory,
  suggestedCardsForMode,
  taxonomyAppForCard
} from './taxonomy'

/**
 * The redesigned Connections page's two filter engines (Figma `1649:30213` and
 * `1565:31077`).
 *
 * Both obey the same contract: selections inside one group are OR'd, groups are
 * AND'd, free text narrows without becoming a chip, and choosing every value in
 * a group snaps the group back to `All` (an empty array).
 */

export type ConnectionStatusFilter = 'any' | 'connected' | 'not_connected'

export interface ConnectionsFilterState {
  categories: readonly ConnectionCategoryId[]
  query: string
  status: ConnectionStatusFilter
}

export const EMPTY_CONNECTIONS_FILTER: ConnectionsFilterState = {
  categories: [],
  query: '',
  status: 'any'
}

export type ToolFacet = 'destructive' | 'read' | 'write'

export type ToolStatusFilter = 'any' | 'off' | 'on'

export interface ConnectionToolsFilterState {
  areas: readonly string[]
  facets: readonly ToolFacet[]
  query: string
  status: ToolStatusFilter
}

export const EMPTY_CONNECTION_TOOLS_FILTER: ConnectionToolsFilterState = {
  areas: [],
  facets: [],
  query: '',
  status: 'any'
}

export const CONNECTION_TOOL_FACETS = ['read', 'write', 'destructive'] as const satisfies readonly ToolFacet[]

function plain(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

export function isConnectionsFilterActive(filter: ConnectionsFilterState): boolean {
  return filter.query.trim().length > 0 || filter.status !== 'any' || filter.categories.length > 0
}

export function isConnectionToolsFilterActive(filter: ConnectionToolsFilterState): boolean {
  return (
    filter.query.trim().length > 0 ||
    filter.status !== 'any' ||
    filter.facets.length > 0 ||
    filter.areas.length > 0
  )
}

// `tool.facet` is a free string on the wire; only the three known facets filter.
const isToolFacet = (value: string): value is ToolFacet =>
  value === 'read' || value === 'write' || value === 'destructive'

function matchesQuery(parts: readonly (string | undefined)[], query: string): boolean {
  const needle = plain(query)

  if (needle.length === 0) {
    return true
  }

  return parts.some(part => part !== undefined && plain(part).includes(needle))
}

function matchesConnectionCategory(
  card: ConnectorCardModel,
  categories: readonly ConnectionCategoryId[]
): boolean {
  if (categories.length === 0) {
    return true
  }

  const category = categoryOfCard(card)

  return category !== 'other' && categories.includes(category)
}

/** Query AND status AND categories; categories are OR'd among themselves. */
export function filterConnectionCards(
  cards: readonly ConnectorCardModel[],
  filter: ConnectionsFilterState
): ConnectorCardModel[] {
  return cards.filter(
    card =>
      matchesQuery([card.name, card.slug, card.description], filter.query) &&
      (filter.status === 'any' || (filter.status === 'connected') === isConnectedCard(card)) &&
      matchesConnectionCategory(card, filter.categories)
  )
}

export interface ConnectionStatusCounts {
  any: number
  connected: number
  not_connected: number
}

export function countByStatus(cards: readonly ConnectorCardModel[]): ConnectionStatusCounts {
  const counts: ConnectionStatusCounts = { any: 0, connected: 0, not_connected: 0 }

  for (const card of cards) {
    counts.any += 1
    counts[isConnectedCard(card) ? 'connected' : 'not_connected'] += 1
  }

  return counts
}

export type ConnectionCategoryCounts = Record<ConnectionCategoryKey, number>

function zeroCategoryCounts(): ConnectionCategoryCounts {
  return {
    'customers-growth': 0,
    'data-analytics': 0,
    'design-content': 0,
    engineering: 0,
    money: 0,
    other: 0,
    reference: 0,
    'travel-fitness': 0,
    'work-productivity': 0
  }
}

export function countByCategory(cards: readonly ConnectorCardModel[]): ConnectionCategoryCounts {
  const counts = zeroCategoryCounts()

  for (const card of cards) {
    counts[categoryOfCard(card)] += 1
  }

  return counts
}

function toggleFilterValue<T extends string>(
  selected: readonly T[],
  value: T,
  available: readonly T[]
): readonly T[] {
  const chosen = new Set(selected)

  if (chosen.has(value)) {
    chosen.delete(value)
  } else {
    chosen.add(value)
  }

  const next = available.filter(entry => chosen.has(entry))

  return next.length === available.length ? [] : next
}

/** Toggling the last missing category selects all and snaps back to `All` (`[]`). */
export function toggleCategoryFilter(
  selected: readonly ConnectionCategoryId[],
  category: ConnectionCategoryId,
  available: readonly ConnectionCategoryId[]
): readonly ConnectionCategoryId[] {
  return toggleFilterValue(selected, category, available)
}

export function toggleToolFacetFilter(
  selected: readonly ToolFacet[],
  facet: ToolFacet,
  available: readonly ToolFacet[]
): readonly ToolFacet[] {
  return toggleFilterValue(selected, facet, available)
}

/** Areas are tool categories; `UNCATEGORISED` keeps tools with no category reachable. */
export function toggleToolAreaFilter(
  selected: readonly string[],
  area: string,
  available: readonly string[]
): readonly string[] {
  return toggleFilterValue(selected, area, available)
}

/** Status and each selected category/facet/area count as one chip; search never does. */
export function activeConnectionsChipCount(filter: ConnectionsFilterState): number {
  return filter.categories.length + (filter.status === 'any' ? 0 : 1)
}

export function activeToolsChipCount(filter: ConnectionToolsFilterState): number {
  return filter.facets.length + filter.areas.length + (filter.status === 'any' ? 0 : 1)
}

function matchesToolArea(tool: ToolRowModel, areas: readonly string[]): boolean {
  if (areas.length === 0) {
    return true
  }

  if (tool.categories.length === 0) {
    return areas.includes(UNCATEGORISED)
  }

  return tool.categories.some(category => areas.includes(category))
}

/** Query AND status AND facets AND areas; values inside each group are OR'd. */
export function filterConnectionTools(
  tools: readonly ToolRowModel[],
  filter: ConnectionToolsFilterState
): ToolRowModel[] {
  return tools.filter(
    tool =>
      matchesQuery([tool.name, tool.slug, tool.description, ...tool.categories], filter.query) &&
      (filter.facets.length === 0 ||
        (isToolFacet(tool.facet) && filter.facets.includes(tool.facet))) &&
      (filter.status === 'any' || (filter.status === 'on') === tool.on) &&
      matchesToolArea(tool, filter.areas)
  )
}

export interface ConnectionToolAreaGroup {
  area: string
  tools: ToolRowModel[]
}

/** Busiest area first, then alphabetical; uncategorised tools sit last. */
export function groupToolsByArea(tools: readonly ToolRowModel[]): ConnectionToolAreaGroup[] {
  const groups = new Map<string, ToolRowModel[]>()

  const add = (area: string, tool: ToolRowModel) => {
    const rows = groups.get(area)

    if (rows) {
      rows.push(tool)
    } else {
      groups.set(area, [tool])
    }
  }

  for (const tool of tools) {
    if (tool.categories.length === 0) {
      add(UNCATEGORISED, tool)
    } else {
      for (const area of tool.categories) {
        add(area, tool)
      }
    }
  }

  return [...groups.entries()]
    .map(([area, rows]) => ({ area, tools: rows }))
    .sort((a, b) => {
      if (a.area === UNCATEGORISED) {
        return 1
      }

      if (b.area === UNCATEGORISED) {
        return -1
      }

      return b.tools.length - a.tools.length || a.area.localeCompare(b.area)
    })
}

export type ConnectionSectionId = 'connected' | 'suggested' | ConnectionCategoryKey

export type ConnectionSectionKind = 'category' | 'connected' | 'suggested'

export interface ConnectionCardSection {
  cards: ConnectorCardModel[]
  id: ConnectionSectionId
  kind: ConnectionSectionKind
  /** Total apps in this section before the preview cap (`See all N`). */
  totalCount: number
}

function compareWithinCategory(a: ConnectorCardModel, b: ConnectorCardModel): number {
  const difference = connectionRank(a) - connectionRank(b)

  return difference === 0 ? a.name.localeCompare(b.name) : difference
}

function compareConnected(a: ConnectorCardModel, b: ConnectorCardModel, mode: ConnectionInterfaceMode): number {
  const order = categoryOrderForMode(mode)
  const position = (card: ConnectorCardModel) => {
    const index = order.indexOf(categoryOfCard(card) as ConnectionCategoryId)

    return index === -1 ? order.length : index
  }
  const categoryDifference = position(a) - position(b)

  return categoryDifference === 0 ? compareWithinCategory(a, b) : categoryDifference
}

function sortWithinCategory(
  cards: readonly ConnectorCardModel[],
  category: ConnectionCategoryKey
): ConnectorCardModel[] {
  return [...cards].sort((a, b) =>
    category === 'other' ? a.name.localeCompare(b.name) : compareWithinCategory(a, b)
  )
}

function otherSection(cards: readonly ConnectorCardModel[]): ConnectionCardSection[] {
  const others = cards.filter(card => categoryOfCard(card) === 'other')

  if (others.length === 0) {
    return []
  }

  const sorted = sortWithinCategory(others, 'other')

  return [{ cards: sorted, id: 'other', kind: 'category' as const, totalCount: sorted.length }]
}

/**
 * Groups cards by their taxonomy order. Used by the filtered main page; the
 * per-category page sorts a single category through the same helper.
 */
function orderedCategorySections(
  cards: readonly ConnectorCardModel[],
  mode: ConnectionInterfaceMode
): ConnectionCardSection[] {
  const sections = categoryOrderForMode(mode).flatMap(category => {
    const inCategory = cards.filter(card => categoryOfCard(card) === category)

    if (inCategory.length === 0) {
      return []
    }

    const sorted = sortWithinCategory(inCategory, category)

    return [
      {
        cards: sorted,
        id: category,
        kind: 'category' as const,
        totalCount: sorted.length
      }
    ]
  })

  return [...sections, ...otherSection(cards)]
}

/** The unfiltered main page: Connected, Suggested, then capped category previews. */
function mainPageSections(
  cards: readonly ConnectorCardModel[],
  mode: ConnectionInterfaceMode
): ConnectionCardSection[] {
  const connected = cards.filter(isConnectedCard).sort((a, b) => compareConnected(a, b, mode))
  const suggested = suggestedCardsForMode(cards, mode)
  const shown = new Set([...connected, ...suggested].map(cardIdentity))
  const sections: ConnectionCardSection[] = []

  if (connected.length > 0) {
    sections.push({ cards: connected, id: 'connected', kind: 'connected', totalCount: connected.length })
  }

  if (suggested.length > 0) {
    sections.push({ cards: suggested, id: 'suggested', kind: 'suggested', totalCount: suggested.length })
  }

  for (const category of categoryOrderForMode(mode)) {
    const inCategory = cards.filter(card => categoryOfCard(card) === category)

    if (inCategory.length === 0) {
      continue
    }

    const preview = sortWithinCategory(inCategory, category)
      .filter(card => !shown.has(cardIdentity(card)))
      .slice(0, categoryPreviewLimit(category))

    // Even with no preview left (every app already sits in Connected or
    // Suggested), keep the category: it must stay searchable/filterable and its
    // See-All page reachable.
    sections.push({ cards: preview, id: category, kind: 'category', totalCount: inCategory.length })
  }

  return [...sections, ...otherSection(cards.filter(card => !shown.has(cardIdentity(card))))]
}

/**
 * Figma `1538:13131` / `1649:30213`.
 *
 * Unfiltered: Connected, Suggested for you, then each non-empty category in the
 * mode's order, capped to its preview limit. Filtered or searching: grouped by
 * category with no cap and no Suggested section.
 */
export function groupCardsForMainPage(
  cards: readonly ConnectorCardModel[],
  mode: ConnectionInterfaceMode,
  filter: ConnectionsFilterState
): ConnectionCardSection[] {
  if (isConnectionsFilterActive(filter)) {
    return orderedCategorySections(filterConnectionCards(cards, filter), mode)
  }

  return mainPageSections(cards, mode)
}

/**
 * Figma `1591:17731`: the category's See-All page split into its taxonomy
 * subgroups, in display order, skipping empty ones. Cards inside a subgroup
 * follow the static `connectionRank`.
 */
export function groupCardsBySubgroup(
  cards: readonly ConnectorCardModel[],
  category: ConnectionCategoryId,
  filter: ConnectionsFilterState
): { id: ConnectionSubgroupId; cards: ConnectorCardModel[] }[] {
  const matched = filterConnectionCards(cards, filter).filter(card => categoryOfCard(card) === category)

  return subgroupOrderForCategory(category).flatMap(id => {
    const inSubgroup = matched.filter(card => taxonomyAppForCard(card)?.subgroup === id)

    if (inSubgroup.length === 0) {
      return []
    }

    return [{ id, cards: inSubgroup.sort(compareWithinCategory) }]
  })
}

/** The category page: every matching card, in subgroup then static rank order. */
export function groupCardsForCategory(
  cards: readonly ConnectorCardModel[],
  category: ConnectionCategoryKey,
  filter: ConnectionsFilterState
): ConnectionCardSection {
  if (category === 'other') {
    const matched = filterConnectionCards(cards, filter).filter(card => categoryOfCard(card) === 'other')
    const sorted = sortWithinCategory(matched, 'other')

    return { cards: sorted, id: 'other', kind: 'category', totalCount: sorted.length }
  }

  const grouped = groupCardsBySubgroup(cards, category, filter).flatMap(group => group.cards)

  return { cards: grouped, id: category, kind: 'category', totalCount: grouped.length }
}
