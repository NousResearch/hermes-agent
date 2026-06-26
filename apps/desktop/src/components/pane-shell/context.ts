import { createContext } from 'react'

export interface PaneSlot {
  side: 'left' | 'right'
  open: boolean
  /** Resolved CSS `grid-column` value (e.g. "3 / 4", or a full-side span for a bottom-row pane). */
  gridColumn: string
  /** Resolved CSS `grid-row` value ("1 / -1" full-height, "1 / 2" above a bottom row, "2 / 3" the row itself). */
  gridRow: string
  /** True when this pane lays out as a horizontal row beneath its rail instead of a vertical column. */
  bottomRow: boolean
}

export interface PaneShellContextValue {
  paneById: Map<string, PaneSlot>
  mainColumn: number
}

export const PaneShellContext = createContext<PaneShellContextValue | null>(null)
