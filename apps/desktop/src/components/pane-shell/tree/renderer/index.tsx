/**
 * Layout tree renderer (root).
 *
 * - `split` -> flex row/column; 1px seams between siblings double as resize
 *   sashes (the seam IS the boundary — junction-owned, never doubled). See
 *   tree-split.tsx.
 * - `group` -> a ZONE: header strip (tabs when stacked, minimize chevron) +
 *   the active pane's content, resolved from the contribution registry
 *   (`area: 'panes'`). Empty zones exist only in editor-authored trees. See
 *   tree-group.tsx.
 *
 * Dragging is FancyZones-style (drag-session.ts): the LAYOUT STAYS FIXED and
 * every zone lights up as a whole-region drop target; dropping moves the pane
 * into that zone (joining its tab stack). Structure changes (splitting/merging/
 * resizing zones) belong to the zone editor, not the drag.
 *
 * This file owns only the composition: the recursive tree, the narrow-viewport
 * overlays, the edit palette, and the zone editor. The pieces live in sibling
 * modules — track-model (sizing), drag-session (drag), tree-split / tree-group
 * (nodes), layout-picker + edit-bar (edit mode), narrow-overlays.
 */

import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'

import { useLayoutEditHotkey } from '../../edit-mode'
import { publishWorkspaceGeometry } from '../../geometry'
import { $layoutTree, trackActiveTreeGroup } from '../store'
import { ZoneEditor } from '../zone-editor'

import { TreeEditBar } from './edit-bar'
import { FloatingPanes } from './floating-panes'
import { NarrowOverlays } from './narrow-overlays'
import { TreeNode } from './tree-node'

export function LayoutTreeRoot({ children }: { children?: ReactNode }) {
  const tree = useStore($layoutTree)

  useLayoutEditHotkey(true)
  // Track the interacted zone so ⌘W closes the right tab even when nothing is
  // DOM-focused.
  useEffect(trackActiveTreeGroup, [])
  // Publish --workspace-left/right so chrome (titlebar title) aligns to the
  // main pane's geometry in plain CSS.
  useEffect(publishWorkspaceGeometry, [])

  if (!tree) {
    return null
  }

  return (
    <div
      className="relative mx-1.5 mb-1.5 flex min-h-0 min-w-0 flex-1 overflow-hidden rounded-[0.875rem] border border-[color-mix(in_srgb,var(--ui-accent,#6e9fc5)_18%,var(--ui-stroke-secondary))] bg-(--ui-editor-surface-background) shadow-[0_10px_32px_color-mix(in_srgb,var(--ui-bg-chrome)_72%,transparent),inset_0_1px_0_color-mix(in_srgb,var(--ui-text-primary)_5%,transparent)] max-[44rem]:mx-1 max-[44rem]:mb-1 max-[44rem]:rounded-[0.625rem]"
      data-hermes-workspace-frame=""
    >
      {/* ZonesOverlay::GetAnimationAlpha ramp: clamp(t / 200ms, 0.001, 1). */}
      <style>{`@keyframes hermes-zone-fade { from { opacity: 0.001 } to { opacity: 1 } }`}</style>
      {/* THE SEAM INVARIANT: boundaries are drawn by the tree (one sash
          hairline per seam) — content mounted in a zone must not paint its
          own edge chrome. App components (asides, the shadcn sidebar) carry
          edge borders + inset highlights for the OLD shell's geometry; this
          neutralizes all of them at the zone boundary, for every current and
          future pane, instead of per-pane class surgery. */}
      <style>{`
        [data-tree-group] :is(aside, [data-slot=sidebar]) {
          border-left-width: 0;
          border-right-width: 0;
          box-shadow: none;
        }
        /* Old-shell titlebar BANDS (chat's session header et al size to
           --titlebar-height, which is 0 inside zones): a zero-height band is
           non-functional but still paints its border-b — a stray hairline
           doubling the zone's top seam. Remove the band entirely. */
        [data-tree-group] header[class*="h-(--titlebar-height)"] {
          display: none;
        }
      `}</style>
      <TreeNode node={tree} root rootRow={tree.type === 'split' && tree.orientation === 'row'} />
      <NarrowOverlays />
      {/* Non-tiling panes: fixed cards above the tree, outside every zone. */}
      <FloatingPanes />
      <TreeEditBar />
      <ZoneEditor />
      {children}
    </div>
  )
}
