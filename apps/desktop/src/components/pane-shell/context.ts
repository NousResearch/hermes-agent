import { createContext } from 'react'

export interface PaneSlot {
  column: number
  side: 'left' | 'right'
  open: boolean
}

export interface PaneShellContextValue {
  paneById: Map<string, PaneSlot>
  mainColumn: number
}

export const PaneShellContext = createContext<PaneShellContextValue | null>(null)
