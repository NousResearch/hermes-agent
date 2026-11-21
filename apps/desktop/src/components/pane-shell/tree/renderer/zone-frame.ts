import type { CSSProperties } from 'react'

import type { Contribution } from '@/contrib/types'
import { useRootAttribute } from '@/lib/root-attribute'

import type { Rect } from '../../geometry'

import { paneChrome } from './track-model'

/** Which window edges a zone touches. The split passes left/right in DOM order (inline start/end). */
export interface ZoneEdges {
  left: boolean
  right: boolean
  top: boolean
}

interface ZoneFrameAttrsInput {
  active: Contribution | undefined
  activeId: string
  edges: ZoneEdges
  headerVisible: boolean
  isEmpty: boolean
  /** Only the redesign's chrome reads the frame beyond the top edge and the header. */
  portal: boolean
}

/** The zone root's data attributes: what shell chrome and panes read about this zone. */
export function zoneFrameAttrs({ active, activeId, edges, headerVisible, isEmpty, portal }: ZoneFrameAttrsInput) {
  const shared = {
    'data-window-top': edges.top || undefined,
    // Advertises the visible tab strip so panes can drop their own
    // self-naming labels (see [data-pane-self-label] in styles.css).
    'data-zone-header': headerVisible || undefined
  }

  if (!portal) {
    return shared
  }

  return {
    ...shared,
    // Window edges, so shell chrome can round only the outer corners of the panel row.
    'data-window-end': edges.right || undefined,
    'data-window-start': edges.left || undefined,
    // The active pane shows a full page (not a tab-able surface): shell chrome
    // frames it as a page rather than a chat panel.
    'data-zone-page': paneChrome(active).headerVeto || undefined,
    'data-zone-pane': isEmpty ? undefined : activeId
  }
}

/** `zoneFrameAttrs` for a mounted zone. It also names on the root that the redesign's chat sidebar is
 *  docked at the window's start: the artwork's veil follows it from the root (portal-artwork.css),
 *  where a :has() would restyle the whole document on every DOM change. */
export function useZoneFrameAttrs(input: ZoneFrameAttrsInput) {
  const { activeId, edges, isEmpty, portal } = input

  useRootAttribute('data-sessions-docked', portal && !isEmpty && activeId === 'sessions' && edges.left)

  return zoneFrameAttrs(input)
}

interface ZoneFrameStyleInput {
  portal: boolean
  titlebarHeight: number
  topEdge: boolean
  verticalCollapse: boolean
  wcOverlap: null | Rect
}

/** Clear the native window controls a zone sits under; a top zone's band height is exposed to its chrome. */
export function zoneFrameStyle({
  portal,
  titlebarHeight,
  topEdge,
  verticalCollapse,
  wcOverlap
}: ZoneFrameStyleInput): CSSProperties | undefined {
  if (wcOverlap) {
    return { paddingTop: wcOverlap.y + wcOverlap.height }
  }

  if (!topEdge) {
    return undefined
  }

  if (verticalCollapse) {
    return { paddingTop: titlebarHeight }
  }

  return portal ? ({ '--zone-band-height': `${titlebarHeight}px` } as CSSProperties) : undefined
}
