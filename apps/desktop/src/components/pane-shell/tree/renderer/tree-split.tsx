/**
 * Split node renderer — a flex row/column whose 1px seams double as resize
 * sashes (the seam IS the boundary — junction-owned, never doubled). Sizing
 * is the TRACK MODEL (track-model.ts): fixed tracks keep their declared size,
 * flex tracks share the leftover by weight, and an all-fixed run lets an
 * UNCAPPED last track absorb the slack (capped sidebars stay put).
 */

import { useStore } from '@nanostores/react'
import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore
} from 'react'

import { beginSashDrag, endSashDrag } from '@/components/pane-shell/geometry'
import { useContributions } from '@/contrib/react/use-contributions'
import { guardGuestPointers } from '@/lib/guest-pointer-guard'
import { rafCoalesce } from '@/lib/raf-coalesce'
import { cn } from '@/lib/utils'
import { $paneStates, type PaneStateSnapshot, setPaneHeightOverride, setPaneWidthOverride } from '@/store/panes'
import { usePortalShell } from '@/store/shell-variant'

import { $layoutEditMode, $layoutEditRevealsHidden } from '../../edit-mode'
import type { LayoutNode, SplitNode } from '../model'
import { allPaneIds } from '../model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  $narrowViewport,
  isCollapsePane,
  paneRootSide,
  persistTree,
  presetSplitWeights,
  setTreeGroupMinimized,
  setTreeSplitWeights
} from '../store'
import { $routeCollapsedPanes } from '../store-route-collapse'

import { useSashLabel } from './sash-copy'
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
  shownPaneIds,
  subtreeGone,
  type TrackContext
} from './track-model'
import { TreeNode } from './tree-node'

/** A key press moves the seam one step; Shift moves it a larger one. */
const SASH_KEY_STEP_PX = 16
const SASH_KEY_STEP_LARGE_PX = 64

/** The single group id a subtree resolves to, or null when it holds several
 *  zones — the sash can only collapse a boundary that IS exactly one zone. */
function groupIdOf(node: LayoutNode): null | string {
  if (node.type === 'group') {
    return node.id
  }

  const ids = new Set(node.children.map(groupIdOf))

  return ids.size === 1 ? [...ids][0] : null
}

/** A zone that folds away on a deep sash drag: every pane it shows declares `dragCollapseBelow`
 *  (the chat sidebar). `below` is the size under which the drag folds it; `collapse` hides its panes
 *  through their own toggles. Null when any pane opts out or the length resolves to nothing (the
 *  classic shell defines no fold width). */
function dragFoldOf(
  paneIds: readonly string[],
  ctx: TrackContext,
  container: HTMLElement,
  horizontal: boolean
): null | { below: number; collapse: () => void } {
  const chromes = paneIds.map(id => paneChrome(ctx.paneFor(id)))

  if (!horizontal || chromes.length === 0 || chromes.some(c => !c.dragCollapseBelow || !c.onDragCollapse)) {
    return null
  }

  const below = Math.max(...chromes.map(c => resolveCssPx(container, c.dragCollapseBelow!, true) ?? 0))
  const collapses = new Set(chromes.map(c => c.onDragCollapse!))

  return below > 0 ? { below, collapse: () => collapses.forEach(collapse => collapse()) } : null
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

    const sig = paneIds.map(id => `${id}:${all[id]?.widthOverride ?? ''}:${all[id]?.heightOverride ?? ''}`).join('|')

    if (cache.current.sig !== sig) {
      cache.current = { sig, value: Object.fromEntries(paneIds.flatMap(id => (all[id] ? [[id, all[id]]] : []))) }
    }

    return cache.current.value
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  return useSyncExternalStore(cb => $paneStates.listen(cb), snapshot, snapshot)
}

export function TreeSplit({
  node,
  root,
  rootRow,
  topEdge = false,
  leftEdge = false,
  rightEdge = false
}: {
  node: SplitNode
  root?: boolean
  rootRow?: boolean
  topEdge?: boolean
  leftEdge?: boolean
  rightEdge?: boolean
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const panes = useContributions('panes')
  const hiddenPanes = useStore($hiddenTreePanes)
  const routeCollapsed = useStore($routeCollapsedPanes)
  const narrow = useStore($narrowViewport)
  // Scoped to THIS subtree's panes: a sash drag writes size overrides on every
  // pointermove, but only the splits whose subtree actually resized should
  // re-render — not every split in the tree.
  const overrides = useSubtreeOverrides(useMemo(() => allPaneIds(node), [node]))
  const editMode = useStore($layoutEditMode)
  const revealsHidden = useStore($layoutEditRevealsHidden)
  const collapsedSides = useStore($collapsedTreeSides)
  const horizontal = node.orientation === 'row'
  // Classic keeps main's pointer-only seam; the redesign's is a focusable window splitter.
  const portal = usePortalShell()
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

  // Layout-edit mode (in Advanced) forces toggle-hidden panes (terminal off,
  // review/preview closed) visible so they're rearrangeable — only truly-absent
  // (unregistered) or narrow-collapsed panes stay gone. Restores itself on exit
  // (render-only).
  const paneGone = (id: string) =>
    !paneFor(id) ||
    (!revealsHidden && (hiddenPanes.has(id) || routeCollapsed.has(id))) ||
    (narrow && Boolean(paneChrome(paneFor(id)).collapsible))

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
  // One seam's resize model, measured from the live DOM: the tracks on either side, the plan for a
  // shift, the fold a deep shift asks for, and the one store commit. A pointer drag previews plans
  // per frame and commits on release; a key press commits one plan at once.
  const sashModel = (aIndex: number, bIndex: number) => {
    const container = containerRef.current

    if (!container) {
      return null
    }

    const rect = container.getBoundingClientRect()
    const totalPx = horizontal ? rect.width : rect.height
    const totalWeight = node.weights.reduce((a, b) => a + b, 0) || 1
    const pxPerWeight = totalPx / totalWeight

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
      // For a nested section this is the INNER flex item, not the seam partner.
      const zoneItem = zoneEl?.parentElement ?? wrapper
      const cs = window.getComputedStyle(zoneItem)
      // A tool panel (terminal / logs) may be dragged down to its collapsed
      // header — the generic 80px floor is not its floor. Below that the
      // release minimizes the zone instead of leaving a useless sliver.
      const toolZone = allPaneIds(child).length > 0 && allPaneIds(child).every(isCollapsePane)
      const floor = toolZone ? COLLAPSED_ZONE_PX : MIN_PANE_PX
      const paneIds = zone ? shownPaneIds(zone, trackCtx) : []

      return {
        // EVERY shown pane of the zone: the zone's track is the max() of its
        // panes' sizes, so the sash writes the same px to all of them —
        // writing only the active pane would leave the zone pinned at a
        // larger sibling's width.
        paneIds,
        fold: dragFoldOf(paneIds, trackCtx, container, horizontal),
        fixed: Boolean(zone),
        size: sizeOf(zoneEl ?? wrapper),
        min: toolZone ? floor : Math.max(floor, computedPx(horizontal ? cs.minWidth : cs.minHeight, 0)),
        max: computedPx(horizontal ? cs.maxWidth : cs.maxHeight, Number.POSITIVE_INFINITY),
        collapseId: toolZone ? (zone?.id ?? groupIdOf(child)) : null,
        floor,
        // The flex item the release commit resizes: the seam partner itself
        // for a direct group, the inner zone wrapper for a nested section.
        zoneItem
      }
    }

    const kidA = container.children[aIndex] as HTMLElement | undefined
    const kidB = container.children[bIndex] as HTMLElement | undefined

    if (!kidA || !kidB) {
      return null
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
        // Seam-partner width at pointerdown. A nested section is wider than
        // its edge zone, so the preview grows the wrapper from this width.
        wrapperSize: sizeOf(element),
        // A minimized rail is its 28px strip: it neither donates nor takes,
        // and its remembered weight must survive the gesture so restoring
        // it brings back the size it had before it was folded.
        minimized: child.type === 'group' && Boolean(child.minimized),
        visible: !isCollapsed(child)
      }
    })

    if (tracks.some(track => !track)) {
      return null
    }

    const sashTracks = tracks as Array<NonNullable<(typeof tracks)[number]>>
    const setOverride = horizontal ? setPaneWidthOverride : setPaneHeightOverride

    // The seam is ONE boundary of a whole run: the side the pointer moves
    // toward gives up space, its seam partner first. When the partner hits
    // its floor the drag keeps going and takes from the partner's next
    // visible sibling, and so on — a row behaves like one workspace instead
    // of a set of unrelated boxes. Plans are absolute (from the sizes at
    // pointerdown), so every frame is a pure function of the pointer.
    const planFor = (requestedShift: number, allowCascade: boolean) => {
      const toward = requestedShift >= 0 ? 1 : -1
      const targetIndex = toward > 0 ? aIndex : bIndex
      // The partner is the nearest VISIBLE sibling (`aIndex`/`bIndex`), not
      // `target ± 1` — a hidden zone parked between them (the closed preview
      // pane) is display:none and must neither donate nor block.
      const partnerIndex = toward > 0 ? bIndex : aIndex
      const target = sashTracks[targetIndex]
      const next = sashTracks.map(track => track.initial)
      let remaining = Math.min(Math.abs(requestedShift), Math.max(0, target.max - target.initial))
      let transferred = 0
      let cascaded = false
      // A tool rail is locally resizable at its own seam, but must not make
      // every unrelated seam in the row local-only. It also remains the
      // terminal donor when reached only after another regular track.
      const cascade = allowCascade && !target.collapseId

      for (let donorIndex = partnerIndex; donorIndex >= 0 && donorIndex < sashTracks.length; donorIndex += toward) {
        const donor = sashTracks[donorIndex]

        if (!donor.visible) {
          continue
        }

        if (donor.collapseId && donorIndex !== partnerIndex) {
          break
        }

        const take = Math.min(remaining, Math.max(0, donor.initial - donor.min))
        next[donorIndex] -= take
        transferred += take
        remaining -= take

        if (take > 0 && donorIndex !== partnerIndex) {
          cascaded = true
        }

        if (remaining === 0 || !cascade) {
          break
        }
      }

      next[targetIndex] += transferred

      return { cascaded, moved: Math.sign(requestedShift) * transferred, sizes: next }
    }

    // A side that folds on a deep drag (the chat sidebar) folds once the pointer asks it to be
    // narrower than its fold length: the seam's leading side follows the pointer, the trailing
    // side mirrors it. Folded, its whole size goes to the other side of the seam.
    const foldSides = [
      { index: aIndex, other: bIndex, sign: 1 },
      { index: bIndex, other: aIndex, sign: -1 }
    ]

    const foldFor = (shift: number) =>
      foldSides.find(({ index, sign }) => {
        const { fold, initial, visible, minimized } = sashTracks[index]

        return fold && visible && !minimized && initial + sign * shift < fold.below
      }) ?? null

    const foldPlan = ({ index, other }: (typeof foldSides)[number]): ReturnType<typeof planFor> => {
      const sizes = sashTracks.map(track => track.initial)
      sizes[other] += sizes[index]
      sizes[index] = 0

      return { cascaded: false, moved: sashTracks[index].initial, sizes }
    }

    // One store commit on release: fixed zones get their px override
    // (sidebar semantics), the flex run gets weights. Clamped px → weights
    // so persisted weights match what's on screen.
    const commitPlan = (plan: ReturnType<typeof planFor>) => {
      const weights = [...node.weights]
      let weightsChanged = false

      sashTracks.forEach((track, index) => {
        if (!track.visible || track.minimized) {
          return
        }

        const px = Math.round(plan.sizes[index])

        if (track.fixed) {
          if (px !== Math.round(track.initial)) {
            track.paneIds.forEach(id => setOverride(id, px))
          }
        } else {
          weights[index] = Math.max(0.01, plan.sizes[index] / pxPerWeight)
          weightsChanged = true
        }
      })

      if (weightsChanged) {
        setTreeSplitWeights(node.id, weights)
      }
    }

    // A tool panel taken down to its collapsed header folds to its rail instead of persisting a
    // sliver, and writes no size, so restoring brings back the size it had before. Only a track
    // THIS plan took to its floor counts: an unrelated rail already resting there must not cancel
    // the commit.
    const settlePlan = (plan: ReturnType<typeof planFor>) => {
      const collapsedSide = sashTracks.find(
        (track, index) => track.collapseId && track.initial > track.floor && plan.sizes[index] <= track.floor
      )?.collapseId

      if (collapsedSide) {
        setTreeGroupMinimized(collapsedSide, true)
      } else {
        commitPlan(plan)
      }
    }

    return { foldFor, foldPlan, foldSides, planFor, sashTracks, settlePlan }
  }

  const startSash = useCallback(
    (aIndex: number, bIndex: number, e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) {
        return
      }

      const model = sashModel(aIndex, bIndex)

      if (!model) {
        return
      }

      e.preventDefault()

      const { foldFor, foldPlan, foldSides, planFor, sashTracks, settlePlan } = model
      const handle = e.currentTarget
      const { pointerId } = e
      const start = horizontal ? e.clientX : e.clientY
      const restoreCursor = document.body.style.cursor
      const restoreSelect = document.body.style.userSelect

      // Nested sections also preview their inner zone wrapper, so snapshot it too.
      const styleSnapshots = [...new Set(sashTracks.flatMap(track => [track.element, track.zoneItem]))].map(
        el => [el, el.getAttribute('style')] as const
      )

      const restoreStyles = () => {
        styleSnapshots.forEach(([el, style]) => {
          if (style === null) {
            el.removeAttribute('style')
          } else {
            el.setAttribute('style', style)
          }
        })
      }

      // During the gesture the store is NOT written. setTreeSplitWeights /
      // setPaneWidthOverride each mint a new tree/pane-state object, and the
      // resulting commit walks every mounted pane — measured live on a real
      // 2-session layout: 31 commits across a 58-frame drag, 20.7fps, with
      // TreeNode at 490ms and Block/Ct re-parsing markdown for 620ms. The
      // store is written ONCE on release; during the drag the run is
      // previewed with inline styles on the same wrappers React sizes.
      //
      // Preview rules (learned the hard way — a wrong shape here left a
      // phantom gap where a hidden sidebar lived):
      //  - a FIXED track gets ONLY a flex-basis override. Its wrapper renders
      //    as `flex: 0 1 <track>`, so basis is the whole difference; grow and
      //    shrink stay React's.
      //  - every visible flex track is pinned to `0 1 <px>`. The plan keeps
      //    the run's total px constant, so no leftover gap can open.
      //  - hidden (display:none) and minimized tracks are left untouched.
      //  - cleanup: release restores the captured style attributes, then a
      //    real drag commits the store once and React re-renders from it.
      const previewPlan = (plan: ReturnType<typeof planFor>) => {
        sashTracks.forEach((track, index) => {
          if (!track.visible || track.minimized) {
            return
          }

          const px = plan.sizes[index]

          if (track.fixed) {
            // Fixed tracks plan in zone space. A nested section's wrapper moves
            // by the zone's delta from its own width. For a direct group both
            // are one element, and the second write leaves it at `px`.
            track.element.style.flexBasis = `${track.wrapperSize + px - track.initial}px`
            track.zoneItem.style.flexBasis = `${px}px`
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
      // Webview/iframe guests must not swallow the gesture — without this the
      // drag froze the moment the pointer entered the in-app browser, so the
      // seam could only shrink it a few px per press.
      const releaseGuests = guardGuestPointers()
      // Suppress :root geometry-var writes for the gesture (see geometry.ts —
      // each one restyles the whole document; they republish on release).
      beginSashDrag()

      // A folding side previews at nothing: its clamp (the wrapper's min size) stands down so the
      // other side takes the whole run. Unfolding restores the captured styles first, so the
      // ordinary preview starts from React's own clamps again.
      let previewedFold: null | (typeof foldSides)[number] = null

      const previewFrame = ({
        fold,
        plan
      }: {
        fold: null | (typeof foldSides)[number]
        plan: ReturnType<typeof planFor>
      }) => {
        if (fold !== previewedFold) {
          restoreStyles()
          previewedFold = fold
        }

        previewPlan(plan)

        if (fold) {
          const { element, zoneItem } = sashTracks[fold.index]
          element.style.minWidth = '0px'
          zoneItem.style.minWidth = '0px'
        }
      }

      // pointermove outpaces 60fps and each write relayouts the whole pane tree,
      // so coalesce to one apply per frame (rafCoalesce commits on cleanup).
      const resize = rafCoalesce(previewFrame)
      // A return drag stays local only after this gesture has actually crossed
      // a second donor. A one-pixel opposite wobble must not poison the real
      // forward movement that follows.
      let cascadeDirection: -1 | 1 | null = null
      let lastPlan: null | ReturnType<typeof planFor> = null
      let lastFold: null | (typeof foldSides)[number] = null
      let done = false

      const onMove = (ev: PointerEvent) => {
        const shift = (horizontal ? ev.clientX : ev.clientY) - start
        const direction = Math.sign(shift) as -1 | 0 | 1

        lastFold = foldFor(shift)

        if (lastFold) {
          resize.push({ fold: lastFold, plan: foldPlan(lastFold) })

          return
        }

        const allowCascade = cascadeDirection === null || direction === cascadeDirection
        lastPlan = planFor(shift, allowCascade)

        if (cascadeDirection === null && direction !== 0 && lastPlan.cascaded) {
          cascadeDirection = direction
        }

        resize.push({ fold: null, plan: lastPlan })
      }

      // Ends through several racing paths (pointerup, pointercancel, window
      // blur, lostpointercapture — releasePointerCapture below fires the
      // latter re-entrantly), so it must run exactly once.
      const cleanup = () => {
        if (done) {
          return
        }

        done = true

        try {
          try {
            resize.finish()
          } finally {
            restoreStyles()
          }

          if (lastFold) {
            // Released folded: hide the side through its own toggle and write no size, so it
            // returns at the width it had before this drag.
            sashTracks[lastFold.index].fold!.collapse()
          } else if (lastPlan && lastPlan.moved !== 0) {
            settlePlan(lastPlan)
          }
        } finally {
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
      }

      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
      window.addEventListener('blur', cleanup)
      handle.addEventListener('lostpointercapture', cleanup)
    },
    // trackCtx is derived state rebuilt per render; the drag captures it once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      axis,
      editMode,
      revealsHidden,
      horizontal,
      node.children,
      node.id,
      node.weights,
      hiddenPanes,
      narrow,
      overrides,
      panes
    ]
  )

  // The separator's keyboard contract (WAI-ARIA window splitter): the arrows along the axis move the
  // seam a step (Shift: a larger one), Home/End take the leading side to its smallest/largest, and
  // Enter folds a side that folds on a deep drag (the chat sidebar), else restores the defaults as
  // a double-click does. Each press is one plan committed at once, through the drag's own model.
  const keySash = (aIndex: number, bIndex: number, event: ReactKeyboardEvent<HTMLDivElement>) => {
    const key = event.key
    const step = event.shiftKey ? SASH_KEY_STEP_LARGE_PX : SASH_KEY_STEP_PX
    const back = horizontal ? 'ArrowLeft' : 'ArrowUp'
    const forward = horizontal ? 'ArrowRight' : 'ArrowDown'

    const shift = { [back]: -step, [forward]: step, End: Number.MAX_SAFE_INTEGER, Home: -Number.MAX_SAFE_INTEGER }[key]

    if (shift === undefined && key !== 'Enter') {
      return
    }

    const model = sashModel(aIndex, bIndex)

    if (!model) {
      return
    }

    event.preventDefault()

    if (key === 'Enter') {
      const folding = model.sashTracks.find(
        (track, index) => (index === aIndex || index === bIndex) && track.fold && track.visible && !track.minimized
      )

      if (folding) {
        folding.fold!.collapse()
      } else {
        resetBoundary(aIndex, bIndex)
      }
    } else {
      const plan = model.planFor(shift!, true)

      if (plan.moved === 0) {
        return
      }

      model.settlePlan(plan)
    }

    persistTree()
  }

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
    [
      axis,
      editMode,
      revealsHidden,
      horizontal,
      node.children,
      node.id,
      node.weights,
      hiddenPanes,
      narrow,
      overrides,
      panes
    ]
  )

  // A run of ONLY fixed tracks can't fill the container (grow-0 all around
  // leaves dead space — e.g. terminal + logs split into two 38vh zones with
  // the rail above them collapsed). An UNCAPPED last track absorbs the
  // leftover; capped sidebars (review/files) keep their max and stay put.
  const isMinimized = (child: LayoutNode) => child.type === 'group' && Boolean(child.minimized)

  // Side toggles own physical sides of the root row, including after a flip.
  // In edit mode sides stay visible.
  // `rootRow` covers both a row root (Default, Focus) and a row nested inside
  // a column root (Terminal deck, Quad) — wherever the side columns live.
  const semanticSides = rootRow && horizontal && collapsedSides.size > 0 && !editMode

  const sideGone = (i: number) => {
    if (!semanticSides) {
      return false
    }

    const side = paneRootSide(allPaneIds(node.children[i])[0])

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

  // Only an uncapped fixed track may absorb leftover. A maxWidth/maxHeight
  // sidebar (review, files, sessions) must keep that clamp — otherwise ⌘G
  // balloons the rail and sash-remembered sizes become a flex-basis that
  // grow still expands past.
  const absorberIndex = allFixed
    ? allFixedAbsorberIndex(growable, i => (horizontal ? tracks[i].sizing?.maxWidth : tracks[i].sizing?.maxHeight))
    : -1

  // A capped all-fixed run leaves slack. When every track left standing is
  // END-placed chrome (a bottom terminal whose column-mates ⌘J folded away),
  // the slack goes BEFORE it so the zone keeps hugging its edge — a terminal
  // deck belongs at the bottom of its column, not floating at the top.
  const endPlacement = horizontal ? 'right' : 'bottom'

  const anchorsEnd =
    allFixed &&
    absorberIndex < 0 &&
    growable.every(i => allPaneIds(tracks[i].child).every(id => paneChrome(paneFor(id)).placement === endPlacement))

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
      className={cn('flex min-h-0 min-w-0 flex-1', horizontal ? 'flex-row' : 'flex-col', anchorsEnd && 'justify-end')}
      data-tree-split={node.id}
      ref={containerRef}
    >
      {tracks.map(({ child, collapsed, minimized, narrowCollapsed, sizing, track }, i) => {
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
                  ? { flex: `0 0 ${horizontal ? MINIMIZED_TRACK : 'auto'}` }
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
                      minWidth: (horizontal && sizing?.minWidth) || 0,
                      maxWidth: horizontal && !absorbs ? sizing?.maxWidth : undefined,
                      minHeight: (!horizontal && sizing?.minHeight) || 0,
                      maxHeight: horizontal || absorbs ? undefined : sizing?.maxHeight
                    }
            }
          >
            {partner >= 0 && (
              <Sash
                disabled={minimized || tracks[partner].minimized}
                horizontal={horizontal}
                keyboard={portal}
                leadingIndex={partner}
                onDoubleClick={() => resetBoundary(partner, i)}
                onKeyDown={e => keySash(partner, i, e)}
                onPointerDown={e => startSash(partner, i, e)}
              />
            )}
            {!narrowCollapsed && (
              <TreeNode
                leftEdge={leftEdge && (!horizontal || i === visibleOrder[0])}
                node={child}
                parentAxis={axis}
                railSide={horizontal ? railSideFor(i) : undefined}
                rightEdge={rightEdge && (!horizontal || i === visibleOrder[visibleOrder.length - 1])}
                rootRow={rootRow || childRootRow(child)}
                topEdge={topEdge && (horizontal || i === visibleOrder[0])}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

/** The leading side's share of the split, in percent: the separator's value. Measured from the live
 *  wrappers and kept current by a ResizeObserver, so it follows drags, keys and window resizes. */
function useLeadingShare(
  ref: { current: HTMLDivElement | null },
  leadingIndex: number,
  horizontal: boolean,
  on: boolean
) {
  const [share, setShare] = useState<number | null>(null)

  useLayoutEffect(() => {
    const container = ref.current?.parentElement?.parentElement
    const leading = container?.children[leadingIndex]

    if (!on || !container || !leading) {
      return
    }

    const measure = () => {
      const box = container.getBoundingClientRect()
      const total = horizontal ? box.width : box.height
      const size = horizontal ? leading.getBoundingClientRect().width : leading.getBoundingClientRect().height
      setShare(total > 0 ? Math.round((100 * size) / total) : null)
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(container)
    observer.observe(leading)

    return () => observer.disconnect()
  }, [ref, leadingIndex, horizontal, on])

  return share
}

function Sash({
  disabled,
  horizontal,
  keyboard: keyboardSeam,
  leadingIndex,
  onDoubleClick,
  onKeyDown,
  onPointerDown
}: {
  disabled?: boolean
  horizontal: boolean
  /** The redesign's seam is focusable and keyboard-operable (a window splitter); classic's is not. */
  keyboard: boolean
  leadingIndex: number
  onDoubleClick?: () => void
  onKeyDown: (e: ReactKeyboardEvent<HTMLDivElement>) => void
  onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const label = useSashLabel()
  const keyboard = keyboardSeam && !disabled
  const share = useLeadingShare(ref, leadingIndex, horizontal, keyboard)

  const splitter = keyboard
    ? {
        'aria-label': label,
        // The seam between side-by-side panes is a vertical line, and the other way round.
        'aria-orientation': horizontal ? ('vertical' as const) : ('horizontal' as const),
        'aria-valuemax': 100,
        'aria-valuemin': 0,
        'aria-valuenow': share ?? undefined,
        onKeyDown,
        tabIndex: 0
      }
    : {}

  return (
    <div
      ref={ref}
      {...splitter}
      className={cn(
        'group absolute z-20 [-webkit-app-region:no-drag]',
        // Grab band lives entirely in the trailing pane. A 1px overlap into
        // the leading pane (the previous asymmetric band) still stole the
        // Windows overlay-scrollbar hit target on the chat — only ~3px of
        // thumb remained clickable (#99867). The sash stays 8px wide, all
        // on the sidebar/tool side of the seam.
        horizontal ? 'inset-y-0 left-0 w-[8px]' : 'inset-x-0 top-0 h-[8px]',
        disabled ? 'pointer-events-none' : horizontal ? 'cursor-col-resize' : 'cursor-row-resize',
        keyboard && 'outline-none'
      )}
      data-sash-overlap="trailing"
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
          'absolute bg-(--ui-stroke-secondary) opacity-10 transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100',
          horizontal ? 'inset-y-0 left-[1px] w-px -translate-x-1/2' : 'inset-x-0 top-[1px] h-px -translate-y-1/2'
        )}
      />
      {!disabled && (
        <span
          className={cn(
            'absolute bg-(--ui-sash-hover-border) opacity-0 transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100',
            horizontal
              ? 'inset-y-0 left-[1px] w-(--vscode-sash-hover-size,0.25rem) -translate-x-1/2'
              : 'inset-x-0 top-[1px] h-(--vscode-sash-hover-size,0.25rem) -translate-y-1/2'
          )}
        />
      )}
    </div>
  )
}
