import { atom, computed, type ReadableAtom } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'

import {
  $sidebarCardRows,
  $sidebarGrouping,
  $sidebarOrdering,
  $sidebarPrFilter,
  $sidebarProfileFilter,
  $sidebarProjectFilter,
  $sidebarRowMeta,
  $sidebarSessionOrderIds,
  $sidebarShowArchived,
  $sidebarStatusFilter,
  setSidebarCardRows,
  setSidebarGrouping,
  setSidebarOrdering,
  setSidebarSessionOrderIds,
  setSidebarShowArchived,
  SIDEBAR_GROUPINGS,
  SIDEBAR_PR_FILTERS,
  SIDEBAR_ROW_META_OPTIONS,
  SIDEBAR_SORT_KEYS,
  SIDEBAR_STATUS_FILTERS,
  type SidebarGrouping,
  type SidebarOrdering,
  type SidebarRowMeta
} from './layout'
import { $profileScope, ALL_PROFILES, normalizeProfileKey, selectProfile, setShowAllProfiles } from './profile'
import type { PullRequestBucket } from './pull-requests'
import type { SessionStatusBucket } from './session-dot-state'

const SIDEBAR_VIEWS_STORAGE_KEY = 'hermes.desktop.sidebarViews'
const SIDEBAR_VIEWS_VERSION = 1 as const
const MAX_VIEW_NAME_LENGTH = 80

export interface SidebarViewFilters {
  profiles: string[]
  projects: string[]
  pullRequests: PullRequestBucket[]
  showArchived: boolean
  statuses: SessionStatusBucket[]
}

export interface SidebarViewState {
  cardRows: boolean
  filters: SidebarViewFilters
  grouping: SidebarGrouping
  manualOrderIds: string[]
  ordering: SidebarOrdering
  profileScope: string
  rowMeta: SidebarRowMeta[]
}

export interface SavedSidebarView {
  createdAt: number
  id: string
  name: string
  state: SidebarViewState
  updatedAt: number
}

export interface SavedSidebarViewsState {
  version: typeof SIDEBAR_VIEWS_VERSION
  views: SavedSidebarView[]
}

const EMPTY_SAVED_VIEWS: SavedSidebarViewsState = { version: SIDEBAR_VIEWS_VERSION, views: [] }
const SIDEBAR_ORDERINGS: readonly SidebarOrdering[] = [...SIDEBAR_SORT_KEYS, 'manual']

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function sanitizeName(value: unknown): string {
  return typeof value === 'string' ? value.trim().slice(0, MAX_VIEW_NAME_LENGTH) : ''
}

function sanitizeStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }

  return [...new Set(value.filter((item): item is string => typeof item === 'string' && item.length > 0))]
}

function sanitizeAllowedList<T extends string>(value: unknown, allowed: readonly T[]): T[] {
  return sanitizeStringList(value).filter((item): item is T => allowed.includes(item as T))
}

function sanitizeSavedSidebarView(value: unknown): SavedSidebarView | null {
  if (!isRecord(value) || !isRecord(value.state) || !isRecord(value.state.filters)) {
    return null
  }

  const id = typeof value.id === 'string' ? value.id.trim() : ''
  const name = sanitizeName(value.name)
  const createdAt = typeof value.createdAt === 'number' && Number.isFinite(value.createdAt) ? value.createdAt : null
  const updatedAt = typeof value.updatedAt === 'number' && Number.isFinite(value.updatedAt) ? value.updatedAt : null

  const grouping = SIDEBAR_GROUPINGS.includes(value.state.grouping as SidebarGrouping)
    ? (value.state.grouping as SidebarGrouping)
    : null

  const ordering = SIDEBAR_ORDERINGS.includes(value.state.ordering as SidebarOrdering)
    ? (value.state.ordering as SidebarOrdering)
    : null

  const profileScope = typeof value.state.profileScope === 'string' ? value.state.profileScope.trim() : ''

  if (
    !id ||
    !name ||
    createdAt === null ||
    updatedAt === null ||
    !grouping ||
    !ordering ||
    !profileScope ||
    (grouping === 'profile' && profileScope !== ALL_PROFILES)
  ) {
    return null
  }

  return {
    createdAt,
    id,
    name,
    state: {
      cardRows: value.state.cardRows === true,
      filters: {
        profiles: sanitizeStringList(value.state.filters.profiles),
        projects: sanitizeStringList(value.state.filters.projects),
        pullRequests: sanitizeAllowedList(value.state.filters.pullRequests, SIDEBAR_PR_FILTERS),
        showArchived: value.state.filters.showArchived === true,
        statuses: sanitizeAllowedList(value.state.filters.statuses, SIDEBAR_STATUS_FILTERS)
      },
      grouping,
      manualOrderIds: sanitizeStringList(value.state.manualOrderIds),
      ordering,
      profileScope,
      rowMeta: sanitizeAllowedList(value.state.rowMeta, SIDEBAR_ROW_META_OPTIONS)
    },
    updatedAt
  }
}

export const sidebarViewsCodec: Codec<SavedSidebarViewsState> = {
  decode: raw => {
    const parsed = JSON.parse(raw) as unknown

    if (!isRecord(parsed) || parsed.version !== SIDEBAR_VIEWS_VERSION || !Array.isArray(parsed.views)) {
      return EMPTY_SAVED_VIEWS
    }

    const seen = new Set<string>()
    const views: SavedSidebarView[] = []

    for (const value of parsed.views) {
      const view = sanitizeSavedSidebarView(value)

      if (view && !seen.has(view.id)) {
        seen.add(view.id)
        views.push(view)
      }
    }

    return { version: SIDEBAR_VIEWS_VERSION, views }
  },
  encode: value => JSON.stringify(value)
}

export const $savedSidebarViews = persistentAtom<SavedSidebarViewsState>(
  SIDEBAR_VIEWS_STORAGE_KEY,
  EMPTY_SAVED_VIEWS,
  sidebarViewsCodec
)

// Selection is session-local. The view state itself is already persisted by
// the layout atoms, and deriving the active id after restart avoids a second
// persisted pointer that could drift from the actual configuration.
const $preferredSavedSidebarViewId = atom<string | null>(null)

function copySidebarViewState(state: SidebarViewState): SidebarViewState {
  return {
    ...state,
    filters: {
      ...state.filters,
      profiles: [...state.filters.profiles],
      projects: [...state.filters.projects],
      pullRequests: [...state.filters.pullRequests],
      statuses: [...state.filters.statuses]
    },
    manualOrderIds: [...state.manualOrderIds],
    rowMeta: [...state.rowMeta]
  }
}

export function captureSidebarViewState(): SidebarViewState {
  return {
    cardRows: $sidebarCardRows.get(),
    filters: {
      profiles: [...$sidebarProfileFilter.get()],
      projects: [...$sidebarProjectFilter.get()],
      pullRequests: [...$sidebarPrFilter.get()],
      showArchived: $sidebarShowArchived.get(),
      statuses: [...$sidebarStatusFilter.get()]
    },
    grouping: $sidebarGrouping.get(),
    manualOrderIds: [...$sidebarSessionOrderIds.get()],
    ordering: $sidebarOrdering.get(),
    profileScope: $profileScope.get(),
    rowMeta: [...$sidebarRowMeta.get()]
  }
}

function sameStringSet(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every(value => b.includes(value))
}

function sameStringOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index])
}

export function sidebarViewStateMatches(a: SidebarViewState, b: SidebarViewState): boolean {
  return (
    a.cardRows === b.cardRows &&
    a.grouping === b.grouping &&
    a.ordering === b.ordering &&
    a.profileScope === b.profileScope &&
    a.filters.showArchived === b.filters.showArchived &&
    sameStringSet(a.rowMeta, b.rowMeta) &&
    sameStringSet(a.filters.statuses, b.filters.statuses) &&
    sameStringSet(a.filters.projects, b.filters.projects) &&
    sameStringSet(a.filters.profiles, b.filters.profiles) &&
    sameStringSet(a.filters.pullRequests, b.filters.pullRequests) &&
    (a.ordering !== 'manual' || sameStringOrder(a.manualOrderIds, b.manualOrderIds))
  )
}

export const $activeSavedSidebarViewId: ReadableAtom<string | null> = computed(
  [
    $savedSidebarViews,
    $sidebarCardRows,
    $sidebarGrouping,
    $sidebarOrdering,
    $sidebarPrFilter,
    $sidebarProfileFilter,
    $sidebarProjectFilter,
    $sidebarRowMeta,
    $sidebarSessionOrderIds,
    $sidebarShowArchived,
    $sidebarStatusFilter,
    $profileScope,
    $preferredSavedSidebarViewId
  ],
  () => {
    const current = captureSidebarViewState()
    const preferredId = $preferredSavedSidebarViewId.get()
    const preferred = $savedSidebarViews.get().views.find(view => view.id === preferredId)

    if (preferred && sidebarViewStateMatches(preferred.state, current)) {
      return preferred.id
    }

    return $savedSidebarViews.get().views.find(view => sidebarViewStateMatches(view.state, current))?.id ?? null
  }
)

export function savedSidebarViewRequiresProfileSwitch(view: SavedSidebarView): boolean {
  return view.state.profileScope !== ALL_PROFILES && view.state.profileScope !== $profileScope.get()
}

interface SaveSidebarViewOptions {
  id?: string
  now?: number
}

function createViewId(): string {
  return globalThis.crypto.randomUUID()
}

export function saveCurrentSidebarView(
  rawName: string,
  { id = createViewId(), now = Date.now() }: SaveSidebarViewOptions = {}
): SavedSidebarView | null {
  const name = sanitizeName(rawName)

  if (!name || !id.trim()) {
    return null
  }

  const view: SavedSidebarView = {
    createdAt: now,
    id: id.trim(),
    name,
    state: copySidebarViewState(captureSidebarViewState()),
    updatedAt: now
  }

  const current = $savedSidebarViews.get()

  $savedSidebarViews.set({ ...current, views: [...current.views.filter(item => item.id !== view.id), view] })
  $preferredSavedSidebarViewId.set(view.id)

  return view
}

export function renameSavedSidebarView(id: string, rawName: string, now = Date.now()): boolean {
  const name = sanitizeName(rawName)
  const current = $savedSidebarViews.get()
  const index = current.views.findIndex(view => view.id === id)

  if (!name || index < 0) {
    return false
  }

  const views = [...current.views]
  views[index] = { ...views[index], name, updatedAt: now }
  $savedSidebarViews.set({ ...current, views })

  return true
}

export function updateSavedSidebarView(id: string, now = Date.now()): boolean {
  const current = $savedSidebarViews.get()
  const index = current.views.findIndex(view => view.id === id)

  if (index < 0) {
    return false
  }

  const views = [...current.views]
  views[index] = {
    ...views[index],
    state: copySidebarViewState(captureSidebarViewState()),
    updatedAt: now
  }
  $savedSidebarViews.set({ ...current, views })
  $preferredSavedSidebarViewId.set(id)

  return true
}

export function deleteSavedSidebarView(id: string): boolean {
  const current = $savedSidebarViews.get()
  const views = current.views.filter(view => view.id !== id)

  if (views.length === current.views.length) {
    return false
  }

  $savedSidebarViews.set({ ...current, views })

  if ($preferredSavedSidebarViewId.get() === id) {
    $preferredSavedSidebarViewId.set(null)
  }

  return true
}

export function applySavedSidebarView(id: string): boolean {
  const view = $savedSidebarViews.get().views.find(item => item.id === id)

  if (!view) {
    return false
  }

  const state = copySidebarViewState(view.state)

  if (state.profileScope === ALL_PROFILES) {
    setShowAllProfiles(true)
  } else {
    selectProfile(normalizeProfileKey(state.profileScope))
  }

  setSidebarGrouping(state.grouping)
  setSidebarOrdering(state.ordering)

  if (state.ordering === 'manual') {
    setSidebarSessionOrderIds(state.manualOrderIds)
  }

  $sidebarRowMeta.set(state.rowMeta)
  setSidebarCardRows(state.cardRows)
  $sidebarStatusFilter.set(state.filters.statuses)
  $sidebarProjectFilter.set(state.filters.projects)
  $sidebarProfileFilter.set(state.filters.profiles)
  $sidebarPrFilter.set(state.filters.pullRequests)
  setSidebarShowArchived(state.filters.showArchived)
  $preferredSavedSidebarViewId.set(view.id)

  return true
}
