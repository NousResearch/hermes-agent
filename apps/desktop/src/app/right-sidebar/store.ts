import { atom, map, type MapStore } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

export type RightSidebarIslandId = 'primary' | 'secondary'
export type RightSidebarTabId = 'files' | 'git' | 'terminal'

export interface RightSidebarIsland {
  active: RightSidebarTabId | null
  tabs: RightSidebarTabId[]
}

export interface RightSidebarWorkspaceLayout {
  primary: RightSidebarIsland
  secondary: RightSidebarIsland
  splitRatio: number
}

interface LegacyLayout {
  mode?: 'single' | 'split' | 'terminal'
  splitRatio?: number
  upperPanel?: 'files' | 'git'
}

const ALL_PANELS: readonly RightSidebarTabId[] = ['files', 'git', 'terminal']
const TAKEOVER_KEY = 'hermes.desktop.terminalTakeover'
const WORKSPACE_LAYOUT_KEY = 'hermes.desktop.rightSidebarLayouts.v2'
const LEGACY_WORKSPACE_LAYOUT_KEY = 'hermes.desktop.rightSidebarLayouts.v1'

const DEFAULT_LAYOUT: RightSidebarWorkspaceLayout = {
  primary: { active: 'files', tabs: [...ALL_PANELS] },
  secondary: { active: null, tabs: [] },
  splitRatio: 0.6
}

function clampRatio(value: number) {
  return Math.max(0.25, Math.min(0.75, Number(value) || DEFAULT_LAYOUT.splitRatio))
}

function isPanel(value: unknown): value is RightSidebarTabId {
  return typeof value === 'string' && ALL_PANELS.includes(value as RightSidebarTabId)
}

function normalizeIsland(value: unknown): RightSidebarIsland {
  const candidate = value && typeof value === 'object' ? (value as Partial<RightSidebarIsland>) : {}
  const tabs = Array.isArray(candidate.tabs) ? candidate.tabs.filter(isPanel) : []
  const uniqueTabs = [...new Set(tabs)]
  const active = isPanel(candidate.active) && uniqueTabs.includes(candidate.active) ? candidate.active : uniqueTabs[0] ?? null

  return { active, tabs: uniqueTabs }
}

function normalizeLayout(value: unknown): RightSidebarWorkspaceLayout | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as Partial<RightSidebarWorkspaceLayout>
  const primary = normalizeIsland(candidate.primary)
  const secondary = normalizeIsland(candidate.secondary)
  const assigned = new Set<RightSidebarTabId>()

  let uniquePrimary = primary.tabs.filter(panel => {
    if (assigned.has(panel)) {
      return false
    }

    assigned.add(panel)

    return true
  })

  let uniqueSecondary = secondary.tabs.filter(panel => {
    if (assigned.has(panel)) {
      return false
    }

    assigned.add(panel)

    return true
  })

  for (const panel of ALL_PANELS) {
    if (!assigned.has(panel)) {
      uniquePrimary.push(panel)
    }
  }

  if (uniquePrimary.length === 0) {
    uniquePrimary = uniqueSecondary
    uniqueSecondary = []
  }

  return {
    primary: normalizeIsland({ active: primary.active, tabs: uniquePrimary }),
    secondary: normalizeIsland({ active: secondary.active, tabs: uniqueSecondary }),
    splitRatio: clampRatio(candidate.splitRatio ?? DEFAULT_LAYOUT.splitRatio)
  }
}

function migrateLegacyLayout(value: unknown): RightSidebarWorkspaceLayout | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const legacy = value as LegacyLayout
  const upper = legacy.upperPanel === 'git' ? 'git' : 'files'
  const other = upper === 'files' ? 'git' : 'files'

  if (legacy.mode === 'split') {
    return {
      primary: { active: upper, tabs: [upper, other] },
      secondary: { active: 'terminal', tabs: ['terminal'] },
      splitRatio: clampRatio(legacy.splitRatio ?? DEFAULT_LAYOUT.splitRatio)
    }
  }

  return {
    primary: {
      active: legacy.mode === 'terminal' ? 'terminal' : upper,
      tabs: [upper, other, 'terminal']
    },
    secondary: { active: null, tabs: [] },
    splitRatio: clampRatio(legacy.splitRatio ?? DEFAULT_LAYOUT.splitRatio)
  }
}

function parseLayouts(raw: string | null, migrate: boolean): Record<string, RightSidebarWorkspaceLayout> {
  const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {}

  return Object.fromEntries(
    Object.entries(parsed).flatMap(([key, value]) => {
      const layout = migrate ? migrateLegacyLayout(value) : normalizeLayout(value)

      return layout ? [[key, layout]] : []
    })
  )
}

function loadWorkspaceLayouts(): Record<string, RightSidebarWorkspaceLayout> {
  try {
    const current = parseLayouts(window.localStorage.getItem(WORKSPACE_LAYOUT_KEY), false)

    if (Object.keys(current).length > 0) {
      return current
    }

    return parseLayouts(window.localStorage.getItem(LEGACY_WORKSPACE_LAYOUT_KEY), true)
  } catch {
    return {}
  }
}

export const $rightSidebarTab = atom<RightSidebarTabId>('files')
export const $rightSidebarWorkspaceLayouts: MapStore<Record<string, RightSidebarWorkspaceLayout>> =
  map(loadWorkspaceLayouts())
export const $terminalTakeover = atom(storedBoolean(TAKEOVER_KEY, false))

$rightSidebarWorkspaceLayouts.subscribe(layouts => {
  try {
    window.localStorage.setItem(WORKSPACE_LAYOUT_KEY, JSON.stringify(layouts))
  } catch {
    // Workspace layout persistence is best-effort.
  }
})

$terminalTakeover.subscribe(active => persistBoolean(TAKEOVER_KEY, active))

export function rightSidebarLayoutFor(workspaceId: string): RightSidebarWorkspaceLayout {
  return $rightSidebarWorkspaceLayouts.get()[workspaceId] ?? DEFAULT_LAYOUT
}

export function updateRightSidebarLayout(
  workspaceId: string,
  update: (layout: RightSidebarWorkspaceLayout) => RightSidebarWorkspaceLayout
) {
  const next = normalizeLayout(update(rightSidebarLayoutFor(workspaceId))) ?? DEFAULT_LAYOUT

  $rightSidebarWorkspaceLayouts.setKey(workspaceId, next)
  $rightSidebarTab.set(
    next.secondary.active === 'terminal' || next.primary.active === 'terminal'
      ? 'terminal'
      : next.primary.active ?? next.secondary.active ?? 'files'
  )
}

export function selectRightSidebarPanel(workspaceId: string, panel: RightSidebarTabId) {
  updateRightSidebarLayout(workspaceId, layout => {
    const islandId = layout.secondary.tabs.includes(panel) ? 'secondary' : 'primary'

    return { ...layout, [islandId]: { ...layout[islandId], active: panel } }
  })
}

export function moveRightSidebarPanel(
  workspaceId: string,
  panel: RightSidebarTabId,
  targetIsland: RightSidebarIslandId,
  targetIndex?: number
) {
  updateRightSidebarLayout(workspaceId, layout => {
    const sourceIsland: RightSidebarIslandId = layout.secondary.tabs.includes(panel) ? 'secondary' : 'primary'
    const sourceTabs = layout[sourceIsland].tabs.filter(item => item !== panel)
    const targetTabs = sourceIsland === targetIsland ? sourceTabs : layout[targetIsland].tabs.filter(item => item !== panel)
    const insertionIndex = Math.max(0, Math.min(targetIndex ?? targetTabs.length, targetTabs.length))
    targetTabs.splice(insertionIndex, 0, panel)

    const next = {
      ...layout,
      [sourceIsland]: normalizeIsland({
        active: layout[sourceIsland].active === panel ? sourceTabs[0] : layout[sourceIsland].active,
        tabs: sourceTabs
      }),
      [targetIsland]: normalizeIsland({ active: panel, tabs: targetTabs })
    }

    return next
  })
}

export function setSidebarSplitRatio(workspaceId: string, splitRatio: number) {
  updateRightSidebarLayout(workspaceId, layout => ({ ...layout, splitRatio }))
}

export const setRightSidebarTab = (tab: RightSidebarTabId) => $rightSidebarTab.set(tab)
export const setTerminalTakeover = (active: boolean) => $terminalTakeover.set(active)

/** A command queued to run in the embedded terminal. The terminal pane flushes
 *  (and clears) it once its session is live, so a value set before the pane
 *  mounts still runs. Cleared after flush so a later remount can't replay it. */
export const $terminalInjection = atom<null | string>(null)

/** Open the terminal pane and run a command in it. Used to disconnect external
 *  (CLI-managed) providers, which Hermes can't clear via the API — the user
 *  sees exactly what runs instead of Hermes silently deleting their creds. */
export const runInTerminal = (command: string) => {
  const trimmed = command.trim()

  if (!trimmed) {
    return
  }

  setTerminalTakeover(true)
  $terminalInjection.set(trimmed)
}
