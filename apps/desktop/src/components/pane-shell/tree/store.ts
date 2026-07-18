/**
 * Layout tree store: one persisted tree replaces paneStates side/band
 * overrides. The DEFAULT tree is declared by the app root (like config);
 * the persisted tree is the user's customization; reset returns to default.
 */

import { atom, computed } from 'nanostores'

import { SIDEBAR_COLLAPSE_MEDIA_QUERY } from '@/app/layout-constants'
import { setPluginEnabled } from '@/contrib/plugins-store'
import { registry } from '@/contrib/registry'
import { translateNow } from '@/i18n'
import { readJson, readKey, writeJson, writeKey } from '@/lib/storage'
import { notify } from '@/store/notifications'
import { clearAllPaneSizeOverrides } from '@/store/panes'
import { isSecondaryWindow } from '@/store/windows'

import {
  allPaneIds,
  type DropPosition,
  findGroup,
  findGroupOfPane,
  groupLeafIds,
  insertAtGroup,
  isLayoutNode,
  type LayoutNode,
  mergeZonesWithPane as mergeZonesWithPaneOp,
  mirrorTreeHorizontal,
  movePane as movePaneOp,
  normalize,
  removePane,
  reorderPaneInGroup as reorderPaneInGroupOp,
  type RootEdge,
  setActivePane as setActivePaneOp,
  setGroupHeaderHidden as setGroupHeaderHiddenOp,
  setGroupMinimized,
  setSplitWeights as setSplitWeightsOp,
  splitGroupZone as splitGroupZoneOp,
  type SplitNode
} from './model'
import { rootChildSide } from './renderer/track-model'

// v2: v1 trees were saved against placeholder panes with index-order zone
// assignment (chat could land in a corner cell). Retire them wholesale.
const STORAGE_KEY = 'hermes.desktop.layoutTree.v2'

writeKey('hermes.desktop.layoutTree.v1', null)

let defaultTree: LayoutNode | null = null

function loadPersisted(): LayoutNode | null {
  const parsed = readJson<unknown>(STORAGE_KEY)

  // Canonicalize on load: strips stale attributes older code persisted
  // (e.g. explicit headerHidden on lone-pane zones) and re-flattens.
  return isLayoutNode(parsed) ? normalize(parsed) : null
}

function persist(tree: LayoutNode | null) {
  // A secondary window (single-chat pop-out) shares the origin's localStorage;
  // writing its stripped-down DEFAULT tree back would wipe the primary's layout.
  if (isSecondaryWindow()) {
    return
  }

  writeJson(STORAGE_KEY, tree)
}

/** The live tree (null until a default is declared). A secondary window ignores
 *  the persisted (primary) layout and boots to the default — nothing but its
 *  own routed session. */
export const $layoutTree = atom<LayoutNode | null>(isSecondaryWindow() ? null : loadPersisted())

/**
 * Which layout preset the current tree came from; `'custom'` after the user
 * rearranges anything. Drives the picker's active highlight.
 */
export const $activePresetId = atom<string>(readKey('hermes.desktop.layoutPreset.active') ?? 'default')

export function markActivePreset(id: string) {
  $activePresetId.set(id)
  writeKey('hermes.desktop.layoutPreset.active', id)
}

/** Pane id being dragged (tree drag session), null when idle. Also set to the
 *  SESSION_TILE_DRAG sentinel while a sidebar session is dragged over the tree,
 *  so the SAME zone overlay lights up (see session-tile-drop-bridge). */
export const $treeDragging = atom<string | null>(null)

/** Sentinel `$treeDragging` value for a session (not a pane) drag — the zone
 *  overlay renders its normal targets, scoped to session-hosting zones. */
export const SESSION_TILE_DRAG = '__session-tile-drag__'

/**
 * Panes hidden by app chrome toggles (titlebar sidebar / right-sidebar
 * buttons). The tree KEEPS the zone and its mounted content; a zone whose
 * every pane is hidden collapses to nothing until a toggle brings it back.
 * Not persisted here — each binding's store owns persistence.
 */
export const $hiddenTreePanes = atom<ReadonlySet<string>>(new Set())

/** Add/remove `item` in a readonly set, returning a fresh set — or null when
 *  membership already matches `present` (so callers can early-out on a no-op). */
function toggledSet<T>(set: ReadonlySet<T>, item: T, present: boolean): Set<T> | null {
  if (set.has(item) === present) {
    return null
  }

  const next = new Set(set)

  if (present) {
    next.add(item)
  } else {
    next.delete(item)
  }

  return next
}

export function setTreePaneHidden(paneId: string, hidden: boolean) {
  const next = toggledSet($hiddenTreePanes.get(), paneId, hidden)

  if (!next) {
    return
  }

  $hiddenTreePanes.set(next)

  // Unhiding is an intent to SEE the pane — front it in its group.
  if (!hidden) {
    revealTreePane(paneId)
  }
}

/**
 * CLOSE — the tab context menu's "Close". Two routes:
 *  - a registered closer (core panes whose visibility an app store owns:
 *    review/terminal/preview/sessions) closes through that store, so the
 *    titlebar/statusbar toggles stay truthful;
 *  - everything else (plugin panes, unbound core panes) is DISMISSED: removed
 *    from the tree and remembered so adoption doesn't re-add it. Reveal
 *    intent (a preview target, ⌘G) or a layout reset un-dismisses.
 */
const DISMISSED_KEY = 'hermes.desktop.dismissedPanes.v1'

function loadDismissed(): ReadonlySet<string> {
  return new Set(readJson<string[]>(DISMISSED_KEY) ?? [])
}

export const $dismissedPanes = atom<ReadonlySet<string>>(loadDismissed())

function saveDismissed(next: ReadonlySet<string>) {
  $dismissedPanes.set(next)
  writeJson(DISMISSED_KEY, next.size === 0 ? null : [...next])
}

function setDismissed(paneId: string, dismissed: boolean) {
  const next = toggledSet($dismissedPanes.get(), paneId, dismissed)

  if (next) {
    saveDismissed(next)
  }
}

const paneClosers: Record<string, () => void> = {}
const paneOpeners: Record<string, () => void> = {}

/** Route a pane's Close through the app store that owns its visibility. */
export function registerPaneCloser(paneId: string, close: () => void) {
  paneClosers[paneId] = close
}

/**
 * Route a pane's "show it" intent through the app store that owns its
 * visibility — the mirror of `registerPaneCloser`, so a preset can reveal a
 * toggle-gated pane (e.g. the terminal, whose visibility ⌃`/`$terminalTakeover`
 * owns) while the toggle stays truthful. Only panes that opt in via
 * `data.revealOnPreset` are opened on preset apply.
 */
export function registerPaneOpener(paneId: string, open: () => void) {
  paneOpeners[paneId] = open
}

// TOOL PANELS (terminal, logs, …): their toggle COLLAPSES the zone to a rail
// (tab stays) instead of hiding it, and the tab's ✕ REMOVES it (vs a session
// tile, whose ✕ closes the session). Membership tells the renderer which
// semantics a tab gets. See bindPaneCollapse in the controller.
const collapsePanes = new Set<string>()

export function markCollapsePane(paneId: string) {
  collapsePanes.add(paneId)
}

export function isCollapsePane(paneId: string): boolean {
  return collapsePanes.has(paneId)
}

const resetHandlers = new Set<() => void>()

/** Run during a layout reset, BEFORE generic adoption — lets an owner
 *  pre-place its panes into the fresh default tree (session tiles collapse
 *  into main as tabs) so adoption sees them already placed and never scatters
 *  them to their old edges. */
export function registerLayoutResetHandler(fn: () => void): () => void {
  resetHandlers.add(fn)

  return () => {
    resetHandlers.delete(fn)
  }
}

/** The zone the user last interacted with (clicked / focused into) — the ⌘W
 *  target when nothing is DOM-focused (activeElement is often `body` after a
 *  click lands on a non-focusable surface). Tracked by trackActiveTreeGroup. */
export const $activeTreeGroup = atom<null | string>(null)

/** Record the interacted zone (pointerdown / focusin). Idempotent. */
export function noteActiveTreeGroup(groupId: null | string) {
  if (groupId !== $activeTreeGroup.get()) {
    $activeTreeGroup.set(groupId)
  }
}

/** Install the active-zone tracker (call once from the tree root). Records the
 *  `[data-tree-group]` under each pointerdown / focusin so ⌘W knows which
 *  zone's tab to close even when nothing is DOM-focused. */
export function trackActiveTreeGroup(): () => void {
  const track = (event: Event) => {
    const el = event.target instanceof HTMLElement ? event.target : null
    const groupId = el?.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup

    if (groupId) {
      noteActiveTreeGroup(groupId)
    }
  }

  window.addEventListener('pointerdown', track, true)
  window.addEventListener('focusin', track, true)

  return () => {
    window.removeEventListener('pointerdown', track, true)
    window.removeEventListener('focusin', track, true)
  }
}

const isUncloseablePane = (paneId: string): boolean =>
  Boolean(
    (registry.getArea('panes').find(c => c.id === paneId)?.data as { uncloseable?: boolean } | undefined)?.uncloseable
  )

/** ⌘W "main tabs always": close the MAIN (workspace) zone's active tab, unless
 *  it's the uncloseable workspace itself. Returns false when there's nothing to
 *  close, so ⌘W stays a no-op — it never closes the window. */
export function closeWorkspaceTab(): boolean {
  const tree = $layoutTree.get()
  const active = tree ? findGroupOfPane(tree, 'workspace')?.active : null

  if (!active || isUncloseablePane(active)) {
    return false
  }

  closeTreePane(active)

  return true
}

/** Closeable siblings of `paneId` within its group, split by position — powers
 *  the tab menu's Close-others / Close-to-the-right verbs (and their enablement). */
function closeableTreeSiblings(paneId: string): { others: string[]; right: string[] } {
  const tree = $layoutTree.get()
  const panes = (tree ? findGroupOfPane(tree, paneId) : null)?.panes ?? []
  const idx = panes.indexOf(paneId)

  return {
    others: panes.filter(id => id !== paneId && !isUncloseablePane(id)),
    right: panes.filter((id, i) => i > idx && !isUncloseablePane(id))
  }
}

/** Closeable-tab counts for a tab's menu enablement (`all` includes self). */
export function treeTabCloseTargets(paneId: string): { all: number; others: number; right: number } {
  const { others, right } = closeableTreeSiblings(paneId)

  return { all: others.length + (isUncloseablePane(paneId) ? 0 : 1), others: others.length, right: right.length }
}

export function closeOtherTreeTabs(paneId: string): void {
  closeableTreeSiblings(paneId).others.forEach(closeTreePane)
}

export function closeTreeTabsToRight(paneId: string): void {
  closeableTreeSiblings(paneId).right.forEach(closeTreePane)
}

/** Close every closeable tab in `paneId`'s group (the uncloseable workspace stays). */
export function closeAllTreeTabs(paneId: string): void {
  const tree = $layoutTree.get()
  const panes = (tree ? findGroupOfPane(tree, paneId) : null)?.panes ?? []

  panes.filter(id => !isUncloseablePane(id)).forEach(closeTreePane)
}

/** Pane ids in the tree under a `${prefix}:` namespace — lets a mirror prune
 *  panes the SHARED (cross-profile) tree persisted for tiles that no longer
 *  back the current profile (a profile switch reloads with the other profile's
 *  tile panes still stacked in). */
export function treePanesWithPrefix(prefix: string): string[] {
  const tree = $layoutTree.get()

  return tree ? allPaneIds(tree).filter(id => id.startsWith(prefix)) : []
}

/** ⌘1…⌘9: activate the Nth tab of the FOCUSED zone (the interaction tracker's
 *  group), but only when it's a real tab strip (≥2 panes). Returns false so the
 *  caller falls back to its default (profile switch) — the number keys mean
 *  "switch tab" only while a multi-tab zone holds focus. */
export function activateTreeTabSlot(slot: number): boolean {
  const groupId = $activeTreeGroup.get()
  const tree = $layoutTree.get()
  const panes = (groupId && tree ? findGroup(tree, groupId)?.panes : null) ?? []

  if (panes.length < 2 || slot < 1 || slot > panes.length) {
    return false
  }

  activateTreePane(groupId!, panes[slot - 1])

  return true
}

/** ⌃Tab / ⌃⇧Tab: cycle the FOCUSED zone's tabs (wrapping) — but only a
 *  session/main strip with ≥2 tabs. Returns false so the caller falls back to
 *  the recent-session switcher when the focus isn't a chat tab strip. */
export function cycleTreeTabInFocusedZone(direction: 1 | -1): boolean {
  const groupId = $activeTreeGroup.get()
  const tree = $layoutTree.get()
  const group = groupId && tree ? findGroup(tree, groupId) : null
  const panes = group?.panes ?? []

  if (panes.length < 2 || !panes.some(id => id === 'workspace' || id.startsWith('session-tile:'))) {
    return false
  }

  const idx = Math.max(0, panes.indexOf(group!.active ?? ''))
  activateTreePane(group!.id, panes[(idx + direction + panes.length) % panes.length])

  return true
}

/** Remove a pane from the tree WITHOUT a dismissal record — for surfaces
 *  whose lifecycle an owner store drives (session tiles): the owner removes
 *  the contribution too, and a later re-open must re-adopt cleanly. */
export function removeTreePane(paneId: string) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(removePane(tree, paneId))
  }
}

/** The layout's root ROW — the split that contains main + the side columns.
 *  Usually the root itself (Default, Focus); in a column-root layout (Terminal
 *  deck, Quad) it's the row child that holds sessions/workspace/files. Returns
 *  null when the tree has no row split with side-eligible panes. */
function rootRow(): SplitNode | null {
  const tree = $layoutTree.get()

  if (!tree || tree.type !== 'split') {
    return null
  }

  if (tree.orientation === 'row') {
    return tree
  }

  // Column root: find the row child that contains the main pane — that's the
  // row the side-collapse system operates on (sessions left, files right).
  const panes = registry.getArea('panes')

  const hasMain = (node: LayoutNode): boolean => {
    if (node.type === 'group') {
      return node.panes.some(
        id => (panes.find(p => p.id === id)?.data as { placement?: string } | undefined)?.placement === 'main'
      )
    }

    return node.children.some(hasMain)
  }

  return (
    (tree.children.find(child => child.type === 'split' && child.orientation === 'row' && hasMain(child)) as
      | SplitNode
      | undefined) ?? null
  )
}

/** Which root-row side a pane currently lives in, or null when it's nested
 *  with main (dragged into the middle) — where a side collapse can't hide it.
 *  Lets side-bound closers (files/sessions) fall back to dismissal. */
export function paneRootSide(paneId: string): null | TreeSide {
  const row = rootRow()

  if (!row) {
    return null
  }

  const panes = registry.getArea('panes')
  const child = row.children.find(c => allPaneIds(c).includes(paneId))

  return child ? rootChildSide(child, id => panes.find(p => p.id === id)) : null
}

/** The closer-less Close: dismiss the pane (removed + remembered; reveal
 *  intent or a layout reset un-dismisses). */
export function dismissTreePane(paneId: string) {
  const tree = $layoutTree.get()

  if (tree) {
    setDismissed(paneId, true)
    commit(removePane(tree, paneId))
  }
}

export function closeTreePane(paneId: string) {
  const closer = paneClosers[paneId]

  if (closer) {
    closer()

    return
  }

  // A plugin's pane: Close = DISABLE the plugin — the same switch as
  // Settings → Plugins, so recovery is discoverable and symmetric. The
  // contribution unregisters but the pane id STAYS in the tree, so
  // re-enabling restores it exactly where it was. (Dismissal + removal
  // would strand the pane with no way back short of a layout reset.)
  const source = registry.getArea('panes').find(c => c.id === paneId)?.source

  if (source?.startsWith('plugin:')) {
    const pluginId = source.slice('plugin:'.length)
    void setPluginEnabled(pluginId, false)
    notify({
      kind: 'info',
      title: translateNow('zones.pluginDisabled', pluginId),
      message: translateNow('zones.pluginDisabledBody')
    })

    return
  }

  dismissTreePane(paneId)
}

/**
 * POSITIONAL side collapse — the titlebar's left/right sidebar toggles (and
 * ⌘B / ⌘J). Everything on that side of the MAIN zone in the root row hides
 * together, whatever panes live there (this is what makes the buttons agree
 * with a rearranged layout; the flip derivation works the same way). An AND
 * on top of per-pane visibility: zone shown ⇔ side open ∧ some pane shown.
 */
export type TreeSide = 'left' | 'right'

export const $collapsedTreeSides = atom<ReadonlySet<TreeSide>>(new Set())

// Side visibility is DERIVED from an app store (the binding owns persistence
// + button state); reveals flow back through its setter so they never
// disagree with the flag.
const sideOpeners: Partial<Record<TreeSide, (open: boolean) => void>> = {}

export function setTreeSideCollapsed(side: TreeSide, collapsed: boolean) {
  const next = toggledSet($collapsedTreeSides.get(), side, collapsed)

  if (next) {
    $collapsedTreeSides.set(next)
  }

  // Opening a side is an intent to SEE it — heal any pane of that side that a
  // stale dismissal record removed from the tree, so ⌘B/⌘J can never press on
  // nothing. Closing chrome panes is NEVER permanent (main parity).
  if (!collapsed) {
    restoreDismissedSidePanes(side)
  }
}

/**
 * Does the layout have a collapsible root side of `side`? ⌘J's normal target is
 * the right sidebar; a layout without one (e.g. a terminal-on-bottom preset)
 * lets callers fall back to the terminal so ⌘J is never a dead key. Semantic —
 * reuses `rootChildSide`, so it tracks a ⌘\ flip / drag like the toggles do.
 */
export function layoutHasRootSide(side: TreeSide): boolean {
  const row = rootRow()

  if (!row) {
    return false
  }

  const panes = registry.getArea('panes')

  return row.children.some(child => rootChildSide(child, id => panes.find(p => p.id === id)) === side)
}

/**
 * Un-dismiss + re-adopt every registered pane whose placement maps to `side`
 * (the same semantic mapping as `rootChildSide`: 'left' panes ⇔ ⌘B, everything
 * else non-main ⇔ ⌘J). Dismissal records for core chrome panes only exist as
 * legacy state (they all register closers now), but they must not strand the
 * pane where only a layout reset can recover it.
 */
function restoreDismissedSidePanes(side: TreeSide) {
  const dismissed = $dismissedPanes.get()

  if (dismissed.size === 0) {
    return
  }

  let changed = false

  for (const pane of registry.getArea('panes')) {
    if (!dismissed.has(pane.id)) {
      continue
    }

    const placement = (pane.data as { placement?: string } | undefined)?.placement
    const paneSide = placement === 'left' ? 'left' : placement === 'main' ? null : 'right'

    if (paneSide === side) {
      setDismissed(pane.id, false)
      changed = true
    }
  }

  if (changed) {
    adoptContributedPanes()
  }
}

/** Bind a side's visibility to an app store (mirror of bindPaneVisibility). */
export function bindTreeSideVisibility(
  side: TreeSide,
  $open: { get(): boolean; listen(fn: (open: boolean) => void): void },
  setOpen: (open: boolean) => void
) {
  sideOpeners[side] = setOpen
  setTreeSideCollapsed(side, !$open.get())
  $open.listen(open => setTreeSideCollapsed(side, !open))
}

/** The chrome toggle owning `paneId`'s root-row column — SEMANTIC, matching
 *  the renderer's `rootChildSide`: ⌘B ⇔ the sessions column (left-placement
 *  panes) wherever it sits, ⌘J ⇔ the other side columns. Null for the main
 *  column (never side-collapsed). */
export function treeSideOfPane(paneId: string): TreeSide | null {
  const row = rootRow()

  if (!row) {
    return null
  }

  const child = row.children.find(node => allPaneIds(node).includes(paneId))

  if (!child) {
    return null
  }

  const placementOf = (id: string) =>
    (registry.getArea('panes').find(c => c.id === id)?.data as { placement?: string } | undefined)?.placement

  const placements = allPaneIds(child).map(placementOf)

  if (placements.includes('main')) {
    return null
  }

  return placements.includes('left') ? 'left' : 'right'
}

/**
 * App intent "show pane X" (a preview target landed, ⌘G opened review, …):
 * open its side, unhide it, and bring it to the front of its group.
 */
export function revealTreePane(paneId: string) {
  // Reveal beats a Close: un-dismiss and let adoption put the pane back.
  if ($dismissedPanes.get().has(paneId)) {
    setDismissed(paneId, false)
    adoptContributedPanes()
  }

  const side = treeSideOfPane(paneId)

  if (side && $collapsedTreeSides.get().has(side)) {
    const open = sideOpeners[side]

    // Through the bound store when there is one, so the toggle stays truthful.
    if (open) {
      open(true)
    } else {
      setTreeSideCollapsed(side, false)
    }
  }

  const hiddenNow = $hiddenTreePanes.get()

  if (hiddenNow.has(paneId)) {
    setTreePaneHidden(paneId, false)

    return
  }

  const tree = $layoutTree.get()
  const group = tree ? findGroupOfPane(tree, paneId) : null

  if (tree && group) {
    // A minimized zone must be restored — "reveal" means show the pane, not
    // just front its tab behind a collapsed rail. Without this, a tool panel
    // (terminal/logs) in a shared zone stays minimized after its toggle opens
    // it: setPaneCollapsed's shared-zone branch calls revealTreePane instead
    // of toggleTreeGroupMinimized, so the zone never un-minimizes and the
    // pane appears to "close but not open" on ctrl-` / tab click.
    let next = tree

    if (group.minimized) {
      next = setGroupMinimized(next, group.id, false)
    }

    if (group.active !== paneId) {
      next = setActivePaneOp(next, group.id, paneId)
    }

    if (next !== tree) {
      commit(next)
    }
  }
}

/**
 * Narrow viewport (the app's sidebar-collapse breakpoint): panes whose
 * contribution declares `collapsible: true` leave the grid and become
 * edge overlays (see NarrowOverlays in renderer.tsx).
 */
// Optional-chained + `typeof window` guarded like every other matchMedia call
// site: this module is imported by non-DOM code paths (session actions) whose
// test env has no `window`/`matchMedia` — an unguarded call throws at load.
const narrowQuery = typeof window !== 'undefined' ? window.matchMedia?.(SIDEBAR_COLLAPSE_MEDIA_QUERY) : undefined

export const $narrowViewport = atom(Boolean(narrowQuery?.matches))

narrowQuery?.addEventListener('change', event => $narrowViewport.set(event.matches))

/** The titlebar flip toggle (⌘\): mirror the whole layout left↔right. */
export function mirrorLayoutTree() {
  const tree = $layoutTree.get()

  if (tree) {
    commit(mirrorTreeHorizontal(tree))
  }
}

export interface DropHint {
  kind: 'group'
  /** The zone a drop will land in (ClosestCenter among `groupIds`). */
  groupId?: string
  /** Full highlighted set (multi-zone when Shift extends the range). */
  groupIds?: string[]
  pos?: DropPosition
  /** Hovering the target's TAB STRIP: the drop stacks at a specific slot —
   *  before this pane id, or at the end (`before: null`). The strip renders
   *  the insertion divider; the zone sheet stands down. */
  stack?: { before: null | string }
}

/** Live drop target under the pointer while dragging. */
export const $dropHint = atom<DropHint | null>(null)

/**
 * Derived session-drag booleans for HEAVY subscribers (the chat surfaces).
 * `$dropHint` churns on every pointer-crossing during ANY drag; a chat surface
 * subscribing to it raw re-renders its whole thread per hint change. These
 * computeds collapse the churn to booleans that only notify on actual flips —
 * and stay `false` throughout pane/tab drags, which chat never cares about.
 */
export const $sessionTileDragging = computed($treeDragging, dragging => dragging === SESSION_TILE_DRAG)

/** True while a session drag aims at a zone EDGE (a tile split) or a tab
 *  strip (a stack) — the moments the chat surfaces' "link to chat" overlay
 *  must stand down. */
export const $sessionTileEdgeHover = computed(
  [$treeDragging, $dropHint],
  (dragging, hint) =>
    dragging === SESSION_TILE_DRAG && ((hint?.pos !== undefined && hint.pos !== 'center') || hint?.stack !== undefined)
)

/**
 * Adopt panes present in `source` but missing from `target`: each joins the
 * group its source siblings map to in the target (first group as a last
 * resort). Layout changes never lose panes.
 */
function adoptMissingPanes(target: LayoutNode, source: LayoutNode): LayoutNode {
  const have = new Set(allPaneIds(target))
  let next = target

  for (const paneId of allPaneIds(source)) {
    if (have.has(paneId)) {
      continue
    }

    const sibling = findGroupOfPane(source, paneId)?.panes.find(p => have.has(p))
    const targetId = (sibling ? findGroupOfPane(next, sibling)?.id : undefined) ?? groupLeafIds(next)[0]

    if (targetId) {
      // Silent adoption: don't steal the target zone's active tab (logs).
      next = insertAtGroup(next, targetId, paneId, 'center', null, false) ?? next
      have.add(paneId)
    }
  }

  return next
}

/**
 * Declare the app's default tree. Adopted immediately when the user has no
 * persisted customization; a persisted tree from an older default adopts any
 * panes it's missing.
 */
export function declareDefaultTree(tree: LayoutNode) {
  defaultTree = tree
  const current = $layoutTree.get()

  if (!current) {
    $layoutTree.set(tree)

    return
  }

  const next = adoptMissingPanes(current, tree)

  if (next !== current) {
    commit(next)
  }
}

/**
 * LIVE pane adoption — a `panes` contribution that isn't in the tree yet
 * (a plugin registered after boot, incl. runtime-loaded ones) joins the
 * tree via the SAME primitive a human drag/drop commits with
 * (`insertAtGroup`: anchor group + side). The pane's data supplies the
 * gesture:
 *
 *  - `dock: { pane, pos }` — "drop me on that edge of that pane". Any pane,
 *    any side, exactly what the drop chips do.
 *  - otherwise the semantic `placement` role infers the anchor: stack with
 *    a settled pane of the same placement, main zone as last resort.
 *
 * Happens once per pane lifetime (the committed tree remembers it across
 * boots), so user rearrangement wins from then on and plugin reloads keep
 * the pane where the user left it.
 */
interface PaneDockHint {
  pane: string
  pos: DropPosition
  /** Center docks: stack BEFORE this pane id (the strip divider's slot). */
  before?: null | string
}

function adoptContributedPanes(): void {
  const tree = $layoutTree.get()

  if (!tree) {
    return
  }

  const panes = registry.getArea('panes')

  const dataOf = (paneId: string) =>
    panes.find(c => c.id === paneId)?.data as { placement?: string; dock?: PaneDockHint } | undefined

  const placementOf = (paneId: string) => dataOf(paneId)?.placement
  const mainId = panes.find(c => placementOf(c.id) === 'main')?.id
  const inTree = new Set(allPaneIds(tree))

  // Plugin panes are never dismissed anymore (Close disables the plugin
  // instead) — drop stale entries so panes stranded by the old behavior
  // re-adopt on their own.
  for (const pane of panes) {
    if (pane.source?.startsWith('plugin:') && $dismissedPanes.get().has(pane.id)) {
      setDismissed(pane.id, false)
    }
  }

  const dismissed = $dismissedPanes.get()
  const missing = panes.filter(c => !inTree.has(c.id) && !dismissed.has(c.id))

  if (missing.length === 0) {
    return
  }

  let next = tree

  for (const pane of missing) {
    const dock = dataOf(pane.id)?.dock
    const placement = placementOf(pane.id) ?? 'right'

    const anchor =
      (dock && allPaneIds(next).includes(dock.pane) ? dock.pane : undefined) ??
      allPaneIds(next).find(id => id !== pane.id && placementOf(id) === placement) ??
      mainId

    const target = findGroupOfPane(next, anchor ?? '')?.id

    if (target) {
      // Silent adoption: don't front over the zone's active tab — a reveal does.
      next = insertAtGroup(next, target, pane.id, dock?.pos ?? 'center', dock?.before, false) ?? next

      // An adopted pane ARRIVES with its chip showing — a surprise zone with
      // zero chrome has no obvious handle to drag or close. (Explicit reveal;
      // the next structural op returns lone panes to the auto-hide default.)
      const landed = findGroupOfPane(next, pane.id)

      if (landed) {
        next = setGroupHeaderHiddenOp(next, landed.id, false)
      }
    }
  }

  if (next !== tree) {
    commit(next)
  }
}

/** Adopt now + on every registry change (call once from the app root). */
export function watchContributedPanes(): void {
  adoptContributedPanes()
  registry.subscribe(adoptContributedPanes)
}

function commit(next: LayoutNode | null) {
  if (!next) {
    return
  }

  $layoutTree.set(next)
  persist(next)
}

// ---------------------------------------------------------------------------
// USER-PLACED panes — "their spot wins". A pane the user has explicitly
// dragged (zone move / span / zone-menu split) keeps that placement; auto-
// docking (dockPaneBeside) only steers panes the user hasn't touched.
// Presets and resets hand placement back to the app.
// ---------------------------------------------------------------------------

const USER_PLACED_KEY = 'hermes.desktop.userPlacedPanes.v1'

export const $userPlacedPanes = atom<ReadonlySet<string>>(new Set(readJson<string[]>(USER_PLACED_KEY) ?? []))

function saveUserPlaced(next: ReadonlySet<string>) {
  $userPlacedPanes.set(next)
  writeJson(USER_PLACED_KEY, next.size === 0 ? null : [...next])
}

function markPaneUserPlaced(paneId: string) {
  const next = toggledSet($userPlacedPanes.get(), paneId, true)

  if (next) {
    saveUserPlaced(next)
  }
}

/**
 * Dock `paneId` directly beside `anchorPaneId` — the "preview opens NEXT TO
 * the file tree" contract, position-aware: wherever the anchor lives (default
 * rail, flipped via ⌘\, dragged into a stack, tabbed into main), the pane
 * lands adjacent to it. Side rule: an anchor sitting right of the main zone
 * gets the pane on its LEFT (the rail slides open toward the chat — main
 * parity); an anchor left of main, stacked with it, or anywhere else gets it
 * on the RIGHT. Skipped when the USER has placed the pane themselves, or the
 * anchor isn't visible. Idempotent — a pane already beside its anchor is a
 * shape no-op.
 */
export function dockPaneBeside(paneId: string, anchorPaneId: string) {
  const tree = $layoutTree.get()

  if (!tree || $userPlacedPanes.get().has(paneId)) {
    return
  }

  const panes = registry.getArea('panes')
  const anchor = findGroupOfPane(tree, anchorPaneId)

  // Anchor must be a live, shown pane — never dock beside a hidden file tree.
  if (!anchor || $hiddenTreePanes.get().has(anchorPaneId) || !panes.some(c => c.id === anchorPaneId)) {
    return
  }

  // The uncloseable main workspace (session tiles are placement:'main' too,
  // but closeable, so the uncloseable flag disambiguates).
  const mainId = panes.find(c => {
    const data = c.data as { placement?: string; uncloseable?: boolean } | undefined

    return data?.placement === 'main' && data.uncloseable
  })?.id

  const order = allPaneIds(tree)

  const anchorRightOfMain =
    !!mainId && !anchor.panes.includes(mainId) && order.indexOf(anchorPaneId) > order.indexOf(mainId)

  const pos: DropPosition = anchorRightOfMain ? 'left' : 'right'

  // A dismissed pane re-enters HERE (beside the anchor), not via adoption's
  // placement fallback — clear the record so the two never disagree.
  if ($dismissedPanes.get().has(paneId)) {
    setDismissed(paneId, false)
  }

  const next = findGroupOfPane(tree, paneId)
    ? movePaneOp(tree, paneId, { groupId: anchor.id, pos })
    : insertAtGroup(tree, anchor.id, paneId, pos)

  if (next && next !== tree) {
    commit(next)
  }
}

export function moveTreePane(paneId: string, target: { groupId: string; pos: DropPosition; before?: null | string }) {
  const tree = $layoutTree.get()

  if (!tree) {
    return
  }

  const next = movePaneOp(tree, paneId, target)

  // movePane returns the SAME root for no-op drops ("stays here") — only a
  // real move customizes the preset or pins the pane as user-placed.
  if (next !== tree) {
    commit(next)
    markActivePreset('custom')
    markPaneUserPlaced(paneId)
  }
}

/**
 * Replace the whole tree (preset application). Panes living in the CURRENT
 * tree that the preset doesn't know about (e.g. plugin panes vs a bundled
 * preset) are adopted into the group their current siblings land in, so
 * applying a preset never loses a pane.
 */
export function applyTree(tree: LayoutNode, presetId: string) {
  const previous = $layoutTree.get()

  // A preset defines the layout's SIZES too — stale drag overrides from the
  // previous arrangement would distort it. Same for user-placed pins: picking
  // a layout hands pane placement back to the app (auto-docking resumes).
  clearAllPaneSizeOverrides()
  saveUserPlaced(new Set())
  commit(previous ? adoptMissingPanes(tree, previous) : tree)
  markActivePreset(presetId)

  // Picking a named layout is an intent to SEE its panes. Toggle-gated panes
  // (the terminal, whose visibility a store owns) would otherwise stay
  // collapsed after the tree changes — so reveal the ones that opt in through
  // their owning store, keeping the ⌃`/toggle state truthful. Iterate the
  // preset's DECLARED panes (not the adopted result): logs is auto-adopted
  // hidden into every tree, so only a preset that explicitly places it (Quad)
  // should turn it on.
  const panes = registry.getArea('panes')

  for (const paneId of allPaneIds(tree)) {
    const data = panes.find(c => c.id === paneId)?.data as { revealOnPreset?: boolean } | undefined

    if (data?.revealOnPreset) {
      paneOpeners[paneId]?.()
    }
  }
}

/**
 * Shift-drag span: merge the highlighted zones into one holding `paneId`. Falls
 * back to a single-zone move at `fallbackGroupId` when the set can't merge
 * (non-rectangular selection).
 */
export function mergeTreeZones(groupIds: string[], paneId: string, fallbackGroupId: string | null) {
  const tree = $layoutTree.get()

  if (!tree) {
    return
  }

  const merged = mergeZonesWithPaneOp(tree, groupIds, paneId)

  if (merged) {
    commit(merged)
    markActivePreset('custom')
    markPaneUserPlaced(paneId)
  } else if (fallbackGroupId) {
    moveTreePane(paneId, { groupId: fallbackGroupId, pos: 'center' })
  }
}

export function activateTreePane(groupId: string, paneId: string) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(setActivePaneOp(tree, groupId, paneId))
  }
}

export function reorderTreePane(groupId: string, paneId: string, toIndex: number) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(reorderPaneInGroupOp(tree, groupId, paneId, toIndex))
    markActivePreset('custom')
  }
}

/** Split a zone on `side`, moving `movePaneId` out of its stack into the new
 *  zone (VS Code split-and-move — the zone menu's Split actions). */
export function splitTreeZone(groupId: string, side: RootEdge, movePaneId: string) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(splitGroupZoneOp(tree, groupId, side, movePaneId))
    markActivePreset('custom')
    markPaneUserPlaced(movePaneId)
  }
}

export function toggleTreeGroupMinimized(groupId: string, minimized: boolean) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(setGroupMinimized(tree, groupId, minimized))
  }
}

/** The group hosting `paneId`, or null. */
function paneGroup(paneId: string) {
  const tree = $layoutTree.get()

  return tree ? findGroupOfPane(tree, paneId) : null
}

/** Collapse/restore a pane's ZONE to a minimized rail — its tab stays visible.
 *  Store-driven (one-way): a tool panel's $open store mirrors here via
 *  bindPaneCollapse, so a toggle collapses rather than hides. */
export function setPaneCollapsed(paneId: string, collapsed: boolean) {
  const group = paneGroup(paneId)

  if (!group) {
    return
  }

  // SHARED zone (terminal + logs, or a tool panel stacked with the workspace):
  // one minimized flag but per-pane toggle stores — so "collapsed" is the
  // ZONE's. Open → reveal + front; close acts ONLY for the on-screen tab. An
  // inactive toggle folding its visible sibling is what re-collapsed the zone
  // on every boot (broke collapse persistence).
  if (group.panes.length > 1) {
    if (collapsed && group.active === paneId) {
      if (group.panes.some(isUncloseablePane)) {
        // Workspace can't minimize (strands the app) → tab-switch to a sibling
        // (guaranteed to exist by length > 1).
        const at = group.panes.indexOf(paneId)

        activateTreePane(group.id, group.panes[at - 1] ?? group.panes[at + 1])
      } else {
        toggleTreeGroupMinimized(group.id, true) // pure tool zone folds as a unit
      }
    } else if (!collapsed) {
      revealTreePane(paneId)
    }

    return
  }

  if (Boolean(group.minimized) !== collapsed) {
    toggleTreeGroupMinimized(group.id, collapsed)

    if (!collapsed) {
      revealTreePane(paneId)
    }
  }
}

/** Restore a minimized tool pane the truthful way — through its store opener
 *  when bound (keeps ⌃`/titlebar toggles in sync), else just un-minimize +
 *  front. Used by the rail (tab / whole-rail click) and the header chevron. */
export function restoreTreePane(paneId: string) {
  const open = paneOpeners[paneId]

  if (open) {
    open()

    // The opener may be a no-op — the store was already true (zone minimized
    // via the zone menu, not the toggle). nanostores don't fire listeners on
    // a same-value .set(), so the bindPaneCollapse listener never runs and
    // the zone stays minimized. Un-minimize directly when that happens.
    const group = paneGroup(paneId)

    if (group?.minimized) {
      toggleTreeGroupMinimized(group.id, false)
    }

    revealTreePane(paneId)

    return
  }

  const group = paneGroup(paneId)

  if (group) {
    toggleTreeGroupMinimized(group.id, false)
    activateTreePane(group.id, paneId)
  }
}

/** Collapse a tool pane through its store closer (truthful), else minimize the
 *  zone directly. Gated on isCollapsePane so a non-tool pane's closer (a tile's
 *  REMOVES it) is never mistaken for a collapse. */
export function collapseTreePane(paneId: string) {
  const close = paneClosers[paneId]

  if (isCollapsePane(paneId) && close) {
    close()

    return
  }

  const group = paneGroup(paneId)

  if (group) {
    toggleTreeGroupMinimized(group.id, true)
  }
}

/** Hide/show a zone's header entirely (double-click gesture). */
export function setTreeGroupHeaderHidden(groupId: string, headerHidden: boolean) {
  const tree = $layoutTree.get()

  if (tree) {
    commit(setGroupHeaderHiddenOp(tree, groupId, headerHidden))
  }
}

export function setTreeSplitWeights(splitId: string, weights: number[]) {
  const tree = $layoutTree.get()

  if (tree) {
    // Weight drags are high-frequency: update live, persist on the trailing edge.
    $layoutTree.set(setSplitWeightsOp(tree, splitId, weights))
  }
}

function findSplitWeights(node: LayoutNode, splitId: string): number[] | null {
  if (node.type !== 'split') {
    return null
  }

  if (node.id === splitId) {
    return node.weights
  }

  for (const child of node.children) {
    const hit = findSplitWeights(child, splitId)

    if (hit) {
      return hit
    }
  }

  return null
}

/**
 * The weights a layout preset declares for `splitId` — the ACTIVE preset
 * first, then any other preset that knows the id. (Rearranging panes marks
 * the active preset 'custom' but zone STRUCTURE — and so split ids — comes
 * from whichever preset was applied, so the original baseline stays
 * findable.) Null when no preset has a matching-shape split.
 */
export function presetSplitWeights(splitId: string, length: number): number[] | null {
  const activeId = $activePresetId.get()
  const presets = [...registry.getArea('layouts')].sort((a, b) => Number(b.id === activeId) - Number(a.id === activeId))

  for (const preset of presets) {
    const weights = preset.data && isLayoutNode(preset.data) ? findSplitWeights(preset.data, splitId) : null

    if (weights && weights.length === length) {
      return [...weights]
    }
  }

  return null
}

export function persistTree() {
  persist($layoutTree.get())
}

export function resetLayoutTree() {
  persist(null)
  clearAllPaneSizeOverrides()
  // Reset restores EVERYTHING — closed panes included — and hands pane
  // placement back to the app (user-placed pins cleared).
  saveDismissed(new Set())
  saveUserPlaced(new Set())
  $layoutTree.set(defaultTree)
  markActivePreset('default')
  // Owners PRE-PLACE their panes into the fresh default (session tiles stack
  // into main as tabs) FIRST, so generic adoption sees them already in-tree
  // and never scatters them to their old edges.
  resetHandlers.forEach(fn => fn())
  // Everything still missing (plugin panes) adopts by placement.
  adoptContributedPanes()

  // "Restore everything" includes collapsed SIDES: reopen every bound side
  // (through its store, so $sidebarOpen / the toggles stay truthful). Without
  // this a sidebar hidden before the reset silently survives it, flipping the
  // next ⌘B into a SHOW — so hiding never appears to persist.
  for (const side of Object.keys(sideOpeners) as TreeSide[]) {
    sideOpeners[side]?.(true)
  }
}

// Dev hook for automation.
if (import.meta.env.DEV && typeof window !== 'undefined') {
  ;(window as unknown as Record<string, unknown>).__HERMES_LAYOUT_TREE__ = {
    close: closeTreePane,
    dismissed: () => $dismissedPanes.get(),
    get: () => $layoutTree.get(),
    move: moveTreePane,
    registry,
    reset: resetLayoutTree,
    reveal: revealTreePane
  }
}
