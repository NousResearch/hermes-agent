/**
 * Split node renderer — a flex row/column whose 1px seams double as resize
 * sashes (the seam IS the boundary — junction-owned, never doubled). Sizing
 * is the TRACK MODEL (track-model.ts): fixed tracks keep their declared size,
 * flex tracks share the leftover by weight, and an all-fixed run lets its
 * last track absorb the slack (VS Code style).
 */

import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useCallback, useMemo, useRef, useSyncExternalStore } from 'react'

import { useContributions } from '@/contrib/react/use-contributions'
import { cn } from '@/lib/utils'
import { $paneStates, type PaneStateSnapshot, setPaneHeightOverride, setPaneWidthOverride } from '@/store/panes'

import { $layoutEditMode } from '../../edit-mode'
import type { LayoutNode, SplitNode } from '../model'
import { allPaneIds } from '../model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  $narrowViewport,
  persistTree,
  presetSplitWeights,
  setTreeSplitWeights
} from '../store'

import {
  computedPx,
  cssMax,
  edgeFixedZone,
  fixedTrackSize,
  MIN_PANE_PX,
  paneChrome,
  type PaneSizing,
  resolveCssPx,
  rootChildSide,
  shownPaneIds,
  subtreeGone,
  type TrackContext
} from './track-model'
import { TreeNode } from './tree-node'

/**
 * The size overrides for a fixed set of panes, referentially stable until one
 * of THEM changes. Sash drags churn `$paneStates` every frame; subscribing the
 * whole map would re-render every split — this narrows each split to its own
 * subtree via a signature-gated snapshot.
 */
function useSubtreeOverrides(paneIds: readonly string[]): TrackContext['overrides'] {
  const key = paneIds.join(',')
  const cache = useRef<{ sig: string; value: Record<string, PaneStateSnapshot> }>({ sig: '\0', value: {} })

  const snapshot = useCallback(() => {
    const all = $paneStates.get()
    const sig = paneIds.map(id => `${id}:${all[id]?.widthOverride ?? ''}:${all[id]?.heightOverride ?? ''}`).join('|')

    if (cache.current.sig !== sig) {
      cache.current = { sig, value: Object.fromEntries(paneIds.flatMap(id => (all[id] ? [[id, all[id]]] : []))) }
    }

    return cache.current.value
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return useSyncExternalStore(cb => $paneStates.listen(cb), snapshot, snapshot)
}

export function TreeSplit({ node, root }: { node: SplitNode; root?: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const panes = useContributions('panes')
  const hiddenPanes = useStore($hiddenTreePanes)
  const narrow = useStore($narrowViewport)
  // Scoped to THIS subtree's panes: a sash drag writes size overrides on every
  // pointermove, but only the splits whose subtree actually resized should
  // re-render — not every split in the tree.
  const overrides = useSubtreeOverrides(useMemo(() => allPaneIds(node), [node]))
  const editMode = useStore($layoutEditMode)
  const collapsedSides = useStore($collapsedTreeSides)
  const horizontal = node.orientation === 'row'
  const axis = node.orientation

  // A pane leaves the grid when its contribution isn't registered (yet) — a
  // runtime plugin's pane collapses until the plugin loads, then appears; no
  // placeholder flash — when a chrome toggle hides it, or when the viewport
  // is narrow and the pane is collapsible (edge overlay instead).
  const paneFor = (id: string) => panes.find(p => p.id === id)

  // Layout-edit mode forces toggle-hidden panes (terminal off, review/preview
  // closed) visible so they're rearrangeable — only truly-absent (unregistered)
  // or narrow-collapsed panes stay gone. Restores itself on exit (render-only).
  const paneGone = (id: string) =>
    !paneFor(id) || (!editMode && hiddenPanes.has(id)) || (narrow && Boolean(paneChrome(paneFor(id)).collapsible))

  const trackCtx: TrackContext = { paneFor, paneGone, overrides }

  // Chrome-toggle collapse: a subtree whose every pane is gone renders
  // display:none (content stays MOUNTED — toggling back is instant), and its
  // siblings absorb the space. Narrow-collapse UNMOUNTS instead, so the edge
  // overlay owns the single live instance of the pane's content.
  // EMPTY zones only exist in editor-authored trees (normalize prunes them on
  // every structural op) — they take space in edit mode as drop targets.
  const isEmptyZone = (child: LayoutNode) => child.type === 'group' && child.panes.length === 0
  const isCollapsed = (child: LayoutNode) => subtreeGone(child, trackCtx) || (isEmptyZone(child) && !editMode)

  // Min/max clamps come from a direct GROUP child's panes (the same clamps
  // the app's Pane props express) — but ONLY when they can speak for the
  // zone: a fixed track (pure sidebar stack) or a single-pane zone. A sidebar
  // pane fronted in a mixed flex stack must not cap it. A fixed STACK
  // aggregates its panes' clamps (largest-tenant semantics, mirroring the
  // max() track basis) — the active tab's caps must never resize the zone.
  const sizingFor = (child: LayoutNode, track: string | null): PaneSizing | null => {
    if (child.type !== 'group' || child.panes.length === 0) {
      return null
    }

    const shownIds = shownPaneIds(child, trackCtx)

    if (track === null && shownIds.length !== 1) {
      return null
    }

    if (shownIds.length <= 1) {
      return (paneFor(shownIds[0])?.data as PaneSizing | undefined) ?? null
    }

    // Fixed STACK: floors take the largest declared min; caps stay unbounded
    // unless EVERY pane declares one (a single uncapped tenant uncaps the
    // zone). Same largest-tenant basis as the track size — never per-tab.
    const all = shownIds.map(id => (paneFor(id)?.data ?? {}) as PaneSizing)

    const cap = (pick: (s: PaneSizing) => string | undefined) =>
      all.every(pick) ? cssMax(all.map(pick)) : undefined

    return {
      minWidth: cssMax(all.map(s => s.minWidth)),
      maxWidth: cap(s => s.maxWidth),
      minHeight: cssMax(all.map(s => s.minHeight)),
      maxHeight: cap(s => s.maxHeight)
    }
  }

  // Sashes pair each visible child with its nearest visible PREVIOUS sibling
  // (`aIndex`/`bIndex`), not blindly `i-1`/`i` — a collapsed zone in between
  // (e.g. the closed preview pane parked between main and the right rail)
  // must not swallow the seam its visible neighbors share.
  const startSash = useCallback(
    (aIndex: number, bIndex: number, e: ReactPointerEvent<HTMLDivElement>) => {
      const container = containerRef.current

      if (!container || e.button !== 0) {
        return
      }

      e.preventDefault()

      const handle = e.currentTarget
      const { pointerId } = e
      const rect = container.getBoundingClientRect()
      const totalPx = horizontal ? rect.width : rect.height
      const totalWeight = node.weights.reduce((a, b) => a + b, 0) || 1
      const pxPerWeight = totalPx / totalWeight
      const start = horizontal ? e.clientX : e.clientY
      const restoreCursor = document.body.style.cursor
      const restoreSelect = document.body.style.userSelect

      // Each side of the seam resolves to a RESIZE TARGET: a fixed zone (the
      // sash writes its px override — sidebar semantics) or the flex run
      // (the sash writes weights). Sizes/clamps read from the live DOM of
      // whichever element actually owns the boundary.
      const sizeOf = (el: HTMLElement) => {
        const r = el.getBoundingClientRect()

        return horizontal ? r.width : r.height
      }

      const sideFor = (child: LayoutNode, wrapper: HTMLElement, edge: 'start' | 'end') => {
        const fixed = fixedTrackSize(child, axis, trackCtx) !== null
        const zone = fixed ? edgeFixedZone(child, edge, axis, trackCtx) : null
        const zoneEl = zone ? container.querySelector<HTMLElement>(`[data-tree-group="${zone.id}"]`) : null
        // Clamps live on the zone's split-child WRAPPER (where we render them).
        const el = zoneEl?.parentElement ?? wrapper
        const cs = window.getComputedStyle(el)

        return {
          // EVERY shown pane of the zone: the zone's track is the max() of its
          // panes' sizes, so the sash writes the same px to all of them —
          // writing only the active pane would leave the zone pinned at a
          // larger sibling's width.
          paneIds: zone ? shownPaneIds(zone, trackCtx) : [],
          fixed: Boolean(zone),
          size: sizeOf(zoneEl ?? wrapper),
          min: Math.max(MIN_PANE_PX, computedPx(horizontal ? cs.minWidth : cs.minHeight, 0)),
          max: computedPx(horizontal ? cs.maxWidth : cs.maxHeight, Number.POSITIVE_INFINITY)
        }
      }

      const kidA = container.children[aIndex] as HTMLElement | undefined
      const kidB = container.children[bIndex] as HTMLElement | undefined

      if (!kidA || !kidB) {
        return
      }

      const a = sideFor(node.children[aIndex], kidA, 'end')
      const b = sideFor(node.children[bIndex], kidB, 'start')
      const a0px = a.fixed ? a.size : sizeOf(kidA)
      const b0px = b.fixed ? b.size : sizeOf(kidB)
      const lo = Math.max(a.min - a0px, b0px - b.max)
      const hi = Math.min(a.max - a0px, b0px - b.min)

      const setOverride = horizontal ? setPaneWidthOverride : setPaneHeightOverride

      try {
        handle.setPointerCapture?.(pointerId)
      } catch {
        // Synthetic events.
      }

      document.body.style.cursor = horizontal ? 'col-resize' : 'row-resize'
      document.body.style.userSelect = 'none'

      const onMove = (ev: PointerEvent) => {
        const shiftPx = Math.max(lo, Math.min(hi, (horizontal ? ev.clientX : ev.clientY) - start))

        if (a.fixed) {
          a.paneIds.forEach(id => setOverride(id, Math.round(a0px + shiftPx)))
        }

        if (b.fixed) {
          b.paneIds.forEach(id => setOverride(id, Math.round(b0px - shiftPx)))
        }

        if (!a.fixed && !b.fixed) {
          const weights = [...node.weights]
          // Convert the CLAMPED pixel sizes back to weights so the persisted
          // weights always agree with what's on screen.
          weights[aIndex] = (a0px + shiftPx) / pxPerWeight
          weights[bIndex] = (b0px - shiftPx) / pxPerWeight
          setTreeSplitWeights(node.id, weights)
        }
      }

      const cleanup = () => {
        document.body.style.cursor = restoreCursor
        document.body.style.userSelect = restoreSelect

        try {
          handle.releasePointerCapture?.(pointerId)
        } catch {
          // Mirror.
        }

        window.removeEventListener('pointermove', onMove, true)
        window.removeEventListener('pointerup', cleanup, true)
        window.removeEventListener('pointercancel', cleanup, true)
        persistTree()
      }

      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
    },
    // trackCtx is derived state rebuilt per render; the drag captures it once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [axis, editMode, horizontal, node.children, node.id, node.weights, hiddenPanes, narrow, overrides, panes]
  )

  // Double-click a sash: every neighbor returns to its DEFAULT size.
  //  - fixed zones (sidebar stacks): clear the drag override -> the declared
  //    width (237px etc.) comes back;
  //  - flex zones fronted by a size-declaring pane (a sidebar in a mixed
  //    stack): pin the weight so the zone lands EXACTLY on that size;
  //  - everything else: the preset's weights for this split (rearranging
  //    panes keeps the applied preset's split ids), else even distribution.
  const resetBoundary = useCallback(
    (aIndex: number, bIndex: number) => {
      const container = containerRef.current

      if (!container) {
        return
      }

      const setOverride = horizontal ? setPaneWidthOverride : setPaneHeightOverride

      for (const [child, edge] of [
        [node.children[aIndex], 'end'],
        [node.children[bIndex], 'start']
      ] as const) {
        const zone = edgeFixedZone(child, edge, axis, trackCtx)

        for (const paneId of zone ? shownPaneIds(zone, trackCtx) : []) {
          setOverride(paneId, undefined)
        }
      }

      const preset = presetSplitWeights(node.id, node.weights.length)
      const weights = preset ?? [...node.weights]

      const rect = container.getBoundingClientRect()
      const totalPx = horizontal ? rect.width : rect.height
      let pinned = false

      for (const i of [aIndex, bIndex]) {
        const child = node.children[i]

        // Fixed tracks size themselves from the declared width (override
        // cleared above) — weights only matter for FLEX zones.
        if (child.type !== 'group' || fixedTrackSize(child, axis, trackCtx) !== null) {
          continue
        }

        // The zone's natural default = the largest size any of its panes
        // declares along this axis (a sessions+terminal stack is still a
        // 237px sidebar at heart, whichever chip is fronted).
        let px: number | null = null

        for (const paneId of shownPaneIds(child, trackCtx)) {
          const sizing = (paneFor(paneId)?.data ?? {}) as PaneSizing
          const css = horizontal ? sizing.width : sizing.height
          const resolved = css ? resolveCssPx(container, css, horizontal) : null

          if (resolved !== null) {
            px = Math.max(px ?? 0, resolved)
          }
        }

        if (px === null || px <= 0 || px >= totalPx) {
          continue
        }

        const others = weights.reduce((sum, w, j) => (j === i ? sum : sum + w), 0)

        if (others > 0) {
          weights[i] = (px * others) / (totalPx - px)
          pinned = true
        }
      }

      setTreeSplitWeights(node.id, !preset && !pinned ? weights.map(() => 1) : weights)
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [axis, editMode, horizontal, node.children, node.id, node.weights, hiddenPanes, narrow, overrides, panes]
  )

  // A run of ONLY fixed tracks can't fill the container (grow-0 all around
  // leaves dead space — e.g. terminal + logs split into two 38vh zones with
  // the rail above them collapsed). The LAST visible track absorbs the
  // leftover, VS Code style.
  const isMinimized = (child: LayoutNode) => child.type === 'group' && Boolean(child.minimized)

  // SEMANTIC side collapse (titlebar toggles / ⌘B / ⌘J): at the ROOT row,
  // ⌘B owns the sessions column and ⌘J the other side columns — by pane
  // placement, NOT position, so a ⌘\ flip moves the columns without
  // rewiring the toggles (main parity). In edit mode sides stay visible.
  const semanticSides = root && horizontal && collapsedSides.size > 0 && !editMode

  const sideGone = (i: number) => {
    if (!semanticSides) {
      return false
    }

    const side = rootChildSide(node.children[i], paneFor)

    return side !== null && collapsedSides.has(side)
  }

  // One pass per child: collapse/minimize state, resolved fixed track, clamps,
  // and narrow-unmount flag. fixedTrackSize + subtreeGone each re-walk the
  // subtree, so resolve them ONCE here instead of per read below.
  const tracks = node.children.map((child, i) => {
    const minimized = isMinimized(child)
    const collapsed = isCollapsed(child) || sideGone(i)
    const track = minimized || collapsed ? null : fixedTrackSize(child, axis, trackCtx)
    const sizing = minimized || collapsed ? null : sizingFor(child, track)
    // Narrow-collapse UNMOUNTS (the edge overlay owns the live instance) — but
    // only for panes the breakpoint collapsed, not ones a chrome toggle hid.
    const narrowCollapsed = narrow && collapsed && allPaneIds(child).some(id => !hiddenPanes.has(id))

    return { child, collapsed, minimized, narrowCollapsed, sizing, track }
  })

  const growable = tracks.map((_, i) => i).filter(i => !tracks[i].collapsed && !tracks[i].minimized)
  const allFixed = growable.length > 0 && growable.every(i => tracks[i].track !== null)
  const absorberIndex = allFixed ? growable[growable.length - 1] : -1

  // Weights are RATIOS, but CSS flex-grow is absolute: a run whose grows sum
  // below 1 fills only that fraction of the leftover (normalize's flatten
  // scales weights into the parent slot — a dock-split nested into an
  // existing column can leave grow 0.5, i.e. dead space). Renormalize the
  // flex run so its grows always sum to 1.
  const flexTotal = growable.reduce((sum, i) => sum + (tracks[i].track === null ? node.weights[i] : 0), 0)
  const grow = (i: number) => node.weights[i] / (flexTotal || 1)

  // The seam partner for a visible child: the nearest VISIBLE previous
  // sibling. Collapsed zones (a hidden pane parked mid-row) are skipped, so
  // their visible neighbors keep a shared, draggable boundary.
  const seamPartner = (i: number): number => {
    for (let j = i - 1; j >= 0; j--) {
      if (!tracks[j].collapsed) {
        return j
      }
    }

    return -1
  }

  // Which half of this row a visible child sits in — a minimized zone's rail
  // hugs the app edge it collapsed toward, so its divider stroke must face
  // the content side (left rail → stroke right, right rail → stroke left).
  const visibleOrder = tracks.map((t, j) => (t.collapsed ? -1 : j)).filter(j => j >= 0)

  const railSideFor = (i: number): 'left' | 'right' => {
    const pos = visibleOrder.indexOf(i)

    return pos >= 0 && (pos + 0.5) / visibleOrder.length > 0.5 ? 'right' : 'left'
  }

  return (
    <div
      className={cn('flex min-h-0 min-w-0 flex-1', horizontal ? 'flex-row' : 'flex-col')}
      data-tree-split={node.id}
      ref={containerRef}
    >
      {tracks.map(({ child, collapsed, minimized, narrowCollapsed, sizing, track }, i) => {
        const partner = collapsed ? -1 : seamPartner(i)

        return (
          <div
            className="relative flex min-h-0 min-w-0"
            key={child.id}
            style={
              collapsed
                ? { display: 'none' }
                : minimized
                  ? { flex: '0 0 auto' }
                  : {
                      // One flexbox formula for everything: a sized zone is
                      // grow-0 shrink-1 from its preferred basis (it yields
                      // gracefully on tight windows, floored by min-width);
                      // everything else splits the leftover by weight. In an
                      // all-fixed run the last track grows into the leftover.
                      flex: track
                        ? `${i === absorberIndex ? 1 : 0} 1 ${track}`
                        : `${grow(i)} ${grow(i)} 0px`,
                      // Pane-declared clamps apply along THIS split's axis only
                      // (a rail's width clamp shouldn't constrain its height).
                      // The absorber drops its max clamp — it exists to fill
                      // the leftover, and clamping would recreate the gap.
                      minWidth: (horizontal && sizing?.minWidth) || 0,
                      maxWidth: horizontal && i !== absorberIndex ? sizing?.maxWidth : undefined,
                      minHeight: (!horizontal && sizing?.minHeight) || 0,
                      maxHeight: horizontal || i === absorberIndex ? undefined : sizing?.maxHeight
                    }
            }
          >
            {partner >= 0 && (
              <Sash
                disabled={minimized || tracks[partner].minimized}
                horizontal={horizontal}
                onDoubleClick={() => resetBoundary(partner, i)}
                onPointerDown={e => startSash(partner, i, e)}
              />
            )}
            {!narrowCollapsed && (
              <TreeNode node={child} parentAxis={axis} railSide={horizontal ? railSideFor(i) : undefined} />
            )}
          </div>
        )
      })}
    </div>
  )
}

function Sash({
  disabled,
  horizontal,
  onDoubleClick,
  onPointerDown
}: {
  disabled?: boolean
  horizontal: boolean
  onDoubleClick?: () => void
  onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => void
}) {
  return (
    <div
      className={cn(
        'group absolute z-20 [-webkit-app-region:no-drag]',
        horizontal ? 'inset-y-0 left-0 w-[9px] -translate-x-1/2' : 'inset-x-0 top-0 h-[9px] -translate-y-1/2',
        disabled ? 'pointer-events-none' : horizontal ? 'cursor-col-resize' : 'cursor-row-resize'
      )}
      onDoubleClick={disabled ? undefined : onDoubleClick}
      onPointerDown={disabled ? undefined : onPointerDown}
      role="separator"
    >
      {/* Persistent hairline: same token as PaneShell's divider sash
          (--ui-stroke-secondary) so every seam — vertical or horizontal —
          reads identically. */}
      <span
        className={cn(
          'absolute bg-(--ui-stroke-secondary)',
          horizontal ? 'inset-y-0 left-1/2 w-px -translate-x-1/2' : 'inset-x-0 top-1/2 h-px -translate-y-1/2'
        )}
      />
      {!disabled && (
        <span
          className={cn(
            'absolute bg-(--ui-sash-hover-border) opacity-0 transition-opacity duration-100 group-hover:opacity-100',
            horizontal
              ? 'inset-y-0 left-1/2 w-(--vscode-sash-hover-size,0.25rem) -translate-x-1/2'
              : 'inset-x-0 top-1/2 h-(--vscode-sash-hover-size,0.25rem) -translate-y-1/2'
          )}
        />
      )}
    </div>
  )
}
