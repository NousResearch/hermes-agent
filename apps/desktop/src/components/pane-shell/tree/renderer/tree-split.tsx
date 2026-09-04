/**
 * Split node renderer — a flex row/column whose 1px seams double as resize
 * sashes (the seam IS the boundary — junction-owned, never doubled). Sizing
 * is the TRACK MODEL (track-model.ts): fixed tracks keep their declared size,
 * flex tracks share the leftover by weight, and an all-fixed run lets an
 * UNCAPPED last track absorb the slack (capped sidebars stay put).
 */

import { useStore } from '@nanostores/react'
import { type PointerEvent as ReactPointerEvent, useCallback, useMemo, useRef, useSyncExternalStore } from 'react'

import { beginSashDrag, endSashDrag } from '@/components/pane-shell/geometry'
import { useContributions } from '@/contrib/react/use-contributions'
import { guardGuestPointers } from '@/lib/guest-pointer-guard'
import { rafCoalesce } from '@/lib/raf-coalesce'
import { cn } from '@/lib/utils'
import { $paneStates, type PaneStateSnapshot, setPaneHeightOverride, setPaneWidthOverride } from '@/store/panes'

import { $layoutEditMode } from '../../edit-mode'
import type { LayoutNode, SplitNode } from '../model'
import { allPaneIds } from '../model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  $narrowViewport,
  isCollapsePane,
  persistTree,
  presetSplitWeights,
  setTreeGroupMinimized,
  setTreeSplitWeights
} from '../store'

import {
  allFixedAbsorberIndex,
  COLLAPSED_ZONE_PX,
  computedPx,
  cssMax,
  edgeFixedZone,
  fixedTrackSize,
  MIN_PANE_PX,
  MINIMIZED_TRACK,
  paneChrome,
  type PaneSizing,
  resolveCssPx,
  rootChildSide,
  shownPaneIds,
  subtreeGone,
  type TrackContext
} from './track-model'
import { TreeNode } from './tree-node'

/** The single group id a subtree resolves to, or null when it holds several
 *  zones — the sash can only collapse a boundary that IS exactly one zone. */
function groupIdOf(node: LayoutNode): null | string {
  if (node.type === 'group') {
    return node.id
  }

  const ids = new Set(node.children.map(groupIdOf))

  return ids.size === 1 ? [...ids][0] : null
}

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

    const sig = paneIds
      .map(
        id =>
          `${id}:${all[id]?.widthOverride ?? ''}:${all[id]?.widthLocked ? 'locked' : ''}:${all[id]?.heightOverride ?? ''}:${all[id]?.heightLocked ? 'locked' : ''}`
      )
      .join('|')

    if (cache.current.sig !== sig) {
      cache.current = { sig, value: Object.fromEntries(paneIds.flatMap(id => (all[id] ? [[id, all[id]]] : []))) }
    }

    return cache.current.value
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return useSyncExternalStore(cb => $paneStates.listen(cb), snapshot, snapshot)
}

export function TreeSplit({ node, root, rootRow }: { node: SplitNode; root?: boolean; rootRow?: boolean }) {
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

  // When the root is a column (Terminal deck, Quad), the root ROW — the one
  // the side-collapse system operates on — is a row child containing main.
  // Propagate `rootRow` to that child so its `semanticSides` fires.
  const childRootRow = (child: LayoutNode): boolean => {
    if (!root || horizontal) {
      return false
    }

    if (child.type !== 'split' || child.orientation !== 'row') {
      return false
    }

    return allPaneIds(child).some(id => paneChrome(paneFor(id)).placement === 'main')
  }

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
  // the app's Pane props express). Floors apply to every zone; caps only when
  // they can speak for the whole zone: a fixed track (pure sidebar stack) or a
  // single-pane zone — a sidebar pane fronted in a mixed flex stack must not
  // cap it. Stacks aggregate clamps (largest-tenant semantics, mirroring the
  // max() track basis) — the active tab's caps must never resize the zone.
  const sizingFor = (child: LayoutNode, track: string | null): PaneSizing | null => {
    if (child.type !== 'group' || child.panes.length === 0) {
      return null
    }

    const shownIds = shownPaneIds(child, trackCtx)

    if (shownIds.length <= 1) {
      return (paneFor(shownIds[0])?.data as PaneSizing | undefined) ?? null
    }

    // STACKS aggregate floors with largest-tenant semantics (the zone's track
    // is the max() of its panes' sizes, so its floor is the max() of their
    // mins) — flex stacks included: the chat zone with session tabs stacked in
    // must keep the workspace's min width, or a browser sash can crush the
    // conversation down to the generic 80px floor. Caps only speak for a FIXED
    // zone; a sidebar pane fronted in a mixed flex stack must not cap it. In a
    // fixed stack caps stay unbounded unless EVERY pane declares one (a single
    // uncapped tenant uncaps the zone).
    const all = shownIds.map(id => (paneFor(id)?.data ?? {}) as PaneSizing)

    const cap = (pick: (s: PaneSizing) => string | undefined) =>
      track !== null && all.every(pick) ? cssMax(all.map(pick)) : undefined

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
        // A tool panel (terminal / logs) may be dragged down to its collapsed
        // header — the generic 80px floor is not its floor. Below that the
        // release minimizes the zone instead of leaving a useless sliver.
        const toolZone = allPaneIds(child).length > 0 && allPaneIds(child).every(isCollapsePane)
        const floor = toolZone ? COLLAPSED_ZONE_PX : MIN_PANE_PX
        const paneIds = zone ? shownPaneIds(zone, trackCtx) : allPaneIds(child).filter(id => !paneGone(id))
        const lockKey = horizontal ? 'widthLocked' : 'heightLocked'

        return {
          // EVERY shown pane of the zone: the zone's track is the max() of its
          // panes' sizes, so the sash writes the same px to all of them —
          // writing only the active pane would leave the zone pinned at a
          // larger sibling's width.
          paneIds,
          fixed: Boolean(zone),
          locked: paneIds.some(id => Boolean(overrides[id]?.[lockKey])),
          size: sizeOf(zoneEl ?? wrapper),
          min: toolZone ? floor : Math.max(floor, computedPx(horizontal ? cs.minWidth : cs.minHeight, 0)),
          max: computedPx(horizontal ? cs.maxWidth : cs.maxHeight, Number.POSITIVE_INFINITY),
          collapseId: toolZone ? (zone?.id ?? groupIdOf(child)) : null,
          floor
        }
      }

      const kidA = container.children[aIndex] as HTMLElement | undefined
      const kidB = container.children[bIndex] as HTMLElement | undefined

      if (!kidA || !kidB) {
        return
      }

      const tracks = node.children.map((child, index) => {
        const element = container.children[index] as HTMLElement | undefined

        if (!element) {
          return null
        }

        const side = sideFor(child, element, index <= aIndex ? 'end' : 'start')

        return {
          ...side,
          element,
          index,
          initial: side.fixed ? side.size : sizeOf(element),
          visible: !isCollapsed(child)
        }
      })

      if (tracks.some(track => !track)) {
        return
      }

      const sashTracks = tracks as Array<NonNullable<(typeof tracks)[number]>>
      const setOverride = horizontal ? setPaneWidthOverride : setPaneHeightOverride

      const planFor = (requestedShift: number, allowCascade: boolean) => {
        const targetIndex = requestedShift >= 0 ? aIndex : bIndex
        const donorStep = requestedShift >= 0 ? 1 : -1
        const target = sashTracks[targetIndex]
        const requested = Math.abs(requestedShift)
        const next = sashTracks.map(track => track.initial)
        const targetCapacity = target.locked ? 0 : Math.max(0, target.max - target.initial)
        let remaining = Math.min(requested, targetCapacity)
        let transferred = 0
        let cascaded = false
        const immediateDonorIndex = targetIndex + donorStep
        // A tool rail is locally resizable at its own seam, but must not make
        // every unrelated seam in the row local-only. It also remains the
        // terminal donor when reached only after another regular track.
        const cascade = allowCascade && !target.collapseId

        for (let donorIndex = immediateDonorIndex; donorIndex >= 0 && donorIndex < sashTracks.length; donorIndex += donorStep) {
          const donor = sashTracks[donorIndex]

          if (donor.collapseId && donorIndex !== immediateDonorIndex) {
            break
          }

          if (!donor.visible || donor.locked) {
            if (!cascade) {
              break
            }

            continue
          }

          const take = Math.min(remaining, Math.max(0, donor.initial - donor.min))
          next[donorIndex] -= take
          transferred += take
          remaining -= take

          if (take > 0 && donorIndex !== targetIndex + donorStep) {
            cascaded = true
          }

          if (remaining === 0 || !cascade) {
            break
          }
        }

        next[targetIndex] += transferred

        return { cascaded, moved: Math.sign(requestedShift) * transferred, sizes: next }
      }

      const commitPlan = (plan: ReturnType<typeof planFor>) => {
        const weights = [...node.weights]
        let weightsChanged = false
        sashTracks.forEach((track, index) => {
          const px = Math.round(plan.sizes[index])

          if (track.fixed) {
            if (!track.locked && px !== Math.round(track.initial)) {
              track.paneIds.forEach(id => setOverride(id, px))
            }
          } else if (track.visible) {
            // Flex weights are relative. Pixel values preserve the on-screen
            // ratio after fixed tracks have claimed their current widths.
            weights[index] = Math.max(0.01, plan.sizes[index] / pxPerWeight)
            weightsChanged = true
          }
        })

        if (weightsChanged) {
          setTreeSplitWeights(node.id, weights)
        }
      }

      const styleSnapshots = sashTracks.map(track => track.element.getAttribute('style'))

      const restoreStyles = () => {
        sashTracks.forEach((track, index) => {
          const style = styleSnapshots[index]

          if (style === null) {
            track.element.removeAttribute('style')
          } else {
            track.element.setAttribute('style', style)
          }
        })
      }

      const previewPlan = (plan: ReturnType<typeof planFor>) => {
        sashTracks.forEach((track, index) => {
          if (!track.visible) {
            return
          }

          const px = plan.sizes[index]

          // Outside a gesture a restored tool has the normal usable 80px CSS
          // floor. While its sash is actively dragging, temporarily release
          // that floor so it can reach the 28px collapse threshold; the next
          // render clears this preview-only inline value.
          if (track.collapseId) {
            if (horizontal) {
              track.element.style.minWidth = '0px'
            } else {
              track.element.style.minHeight = '0px'
            }
          }

          if (track.fixed) {
            track.element.style.flexBasis = `${px}px`
          } else {
            track.element.style.flex = `0 1 ${px}px`
          }
        })
      }

      try {
        handle.setPointerCapture?.(pointerId)
      } catch {
        // Synthetic events.
      }

      document.body.style.cursor = horizontal ? 'col-resize' : 'row-resize'
      document.body.style.userSelect = 'none'
      const releaseGuests = guardGuestPointers()
      beginSashDrag()

      const resize = rafCoalesce(previewPlan)
      // A return drag stays local only after this gesture has actually crossed
      // a second donor. A one-pixel opposite wobble must not poison the real
      // forward movement that follows.
      let cascadeDirection: -1 | 1 | null = null
      let lastPlan: null | ReturnType<typeof planFor> = null
      let done = false

      const onMove = (ev: PointerEvent) => {
        const shift = (horizontal ? ev.clientX : ev.clientY) - start
        const direction = Math.sign(shift) as -1 | 0 | 1

        const allowCascade = cascadeDirection === null || direction === cascadeDirection
        lastPlan = planFor(shift, allowCascade)

        if (cascadeDirection === null && direction !== 0 && lastPlan.cascaded) {
          cascadeDirection = direction
        }

        resize.push(lastPlan)
      }

      // Ends through several racing paths (pointerup, pointercancel, window
      // blur, lostpointercapture — releasePointerCapture below fires the
      // latter re-entrantly), so it must run exactly once.
      const cleanup = () => {
        if (done) {
          return
        }

        done = true
        resize.finish()

        if (lastPlan && lastPlan.moved !== 0) {
          const collapsedSide = sashTracks.find(
            (track, index) => track.collapseId && lastPlan!.sizes[index] <= track.floor
          )?.collapseId

          if (collapsedSide) {
            setTreeGroupMinimized(collapsedSide, true)
          } else {
            // One store commit; React rewrites the flex styles after release.
            commitPlan(lastPlan)
          }
        } else {
          // No usable movement means React will not rerender. Restore every
          // track, not merely the two immediate seam partners.
          restoreStyles()
        }

        // Geometry vars re-enable AFTER the final store commit above, so the
        // release publishes exactly one fresh measurement.
        endSashDrag()
        releaseGuests()
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
        window.removeEventListener('blur', cleanup)
        handle.removeEventListener('lostpointercapture', cleanup)
        persistTree()
      }

      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
      window.addEventListener('blur', cleanup)
      handle.addEventListener('lostpointercapture', cleanup)
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
  // the rail above them collapsed). An UNCAPPED last track absorbs the
  // leftover; capped sidebars (review/files) keep their max and stay put.
  const isMinimized = (child: LayoutNode) => child.type === 'group' && Boolean(child.minimized)

  // SEMANTIC side collapse (titlebar toggles / ⌘B / ⌘J): at the ROOT row,
  // ⌘B owns the sessions column and ⌘J the other side columns — by pane
  // placement, NOT position, so a ⌘\ flip moves the columns without
  // rewiring the toggles (main parity). In edit mode sides stay visible.
  // `rootRow` covers both a row root (Default, Focus) and a row nested inside
  // a column root (Terminal deck, Quad) — wherever the side columns live.
  const semanticSides = rootRow && horizontal && collapsedSides.size > 0 && !editMode

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
    const toolZone = allPaneIds(child).length > 0 && allPaneIds(child).every(isCollapsePane)
    const track = minimized || collapsed ? null : fixedTrackSize(child, axis, trackCtx)
    const sizing = minimized || collapsed ? null : sizingFor(child, track)
    // Narrow-collapse UNMOUNTS (the edge overlay owns the live instance) — but
    // only for panes the breakpoint collapsed, not ones a chrome toggle hid.
    const narrowCollapsed = narrow && collapsed && allPaneIds(child).some(id => !hiddenPanes.has(id))

    return { child, collapsed, minimized, narrowCollapsed, sizing, toolZone, track }
  })

  const growable = tracks.map((_, i) => i).filter(i => !tracks[i].collapsed && !tracks[i].minimized)
  const allFixed = growable.length > 0 && growable.every(i => tracks[i].track !== null)

  // Only an uncapped fixed track may absorb leftover. A maxWidth/maxHeight
  // sidebar (review, files, sessions) must keep that clamp — otherwise ⌘G
  // balloons the rail and sash-remembered sizes become a flex-basis that
  // grow still expands past.
  const absorberIndex = allFixed
    ? allFixedAbsorberIndex(growable, i => (horizontal ? tracks[i].sizing?.maxWidth : tracks[i].sizing?.maxHeight))
    : -1

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
      {tracks.map(({ child, collapsed, minimized, narrowCollapsed, sizing, toolZone, track }, i) => {
        const partner = collapsed ? -1 : seamPartner(i)
        const absorbs = i === absorberIndex

        return (
          <div
            className="relative flex min-h-0 min-w-0"
            key={child.id}
            style={
              collapsed
                ? { display: 'none' }
                : minimized
                  ? { flex: `0 0 ${MINIMIZED_TRACK}` }
                  : {
                      // One flexbox formula for everything: a sized zone is
                      // grow-0 shrink-1 from its preferred basis (it yields
                      // gracefully on tight windows, floored by min-width);
                      // everything else splits the leftover by weight. In an
                      // all-fixed run an UNCAPPED last track grows into the
                      // leftover; capped sidebars stay at their declared size.
                      flex: track ? `${absorbs ? 1 : 0} 1 ${track}` : `${grow(i)} ${grow(i)} 0px`,
                      // Pane-declared clamps apply along THIS split's axis only
                      // (a rail's width clamp shouldn't constrain its height).
                      // The absorber is uncapped by selection, so dropping its
                      // max is a no-op; capped tracks always keep theirs.
                      // A restored tool can remain a flex track, so its CSS
                      // floor must match the 80px floor used by the sash.
                      // Otherwise a tiny remembered weight redraws Terminal as
                      // a 0px edge sliver before the user can grab its divider.
                      minWidth: horizontal ? (sizing?.minWidth ?? (toolZone ? `${MIN_PANE_PX}px` : 0)) : 0,
                      maxWidth: horizontal && !absorbs ? sizing?.maxWidth : undefined,
                      minHeight: !horizontal ? (sizing?.minHeight ?? (toolZone ? `${MIN_PANE_PX}px` : 0)) : 0,
                      maxHeight: horizontal || absorbs ? undefined : sizing?.maxHeight
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
              <TreeNode
                node={child}
                parentAxis={axis}
                railSide={horizontal ? railSideFor(i) : undefined}
                rootRow={rootRow || childRootRow(child)}
              />
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
        // Asymmetric grab band: only 1px reaches into the leading pane so its
        // edge-hugging 4px scrollbar stays clickable (the old centered 9px band
        // swallowed it entirely — the pointer got col-resize instead of the
        // thumb). The trailing side keeps a generous 7px reach; total grab
        // width stays ~8px so the sash is no harder to hit.
        horizontal ? 'inset-y-0 left-0 w-[8px] -translate-x-[1px]' : 'inset-x-0 top-0 h-[8px] -translate-y-[1px]',
        disabled ? 'pointer-events-none' : horizontal ? 'cursor-col-resize' : 'cursor-row-resize'
      )}
      onDoubleClick={disabled ? undefined : onDoubleClick}
      onPointerDown={disabled ? undefined : onPointerDown}
      role="separator"
    >
      {/* Persistent hairline: same token as PaneShell's divider sash
          (--ui-stroke-secondary) so every seam — vertical or horizontal —
          reads identically. Sits at 0.1 so seams recede into the surface,
          and comes up to full on hover alongside the thicker grab band. */}
      <span
        className={cn(
          'absolute bg-(--ui-stroke-secondary) opacity-10 transition-opacity duration-100 group-hover:opacity-100',
          horizontal ? 'inset-y-0 left-[1px] w-px -translate-x-1/2' : 'inset-x-0 top-[1px] h-px -translate-y-1/2'
        )}
      />
      {!disabled && (
        <span
          className={cn(
            'absolute bg-(--ui-sash-hover-border) opacity-0 transition-opacity duration-100 group-hover:opacity-100',
            horizontal
              ? 'inset-y-0 left-[1px] w-(--vscode-sash-hover-size,0.25rem) -translate-x-1/2'
              : 'inset-x-0 top-[1px] h-(--vscode-sash-hover-size,0.25rem) -translate-y-1/2'
          )}
        />
      )}
    </div>
  )
}
