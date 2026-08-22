import { useEffect, useState } from 'react'

export const KANBAN_DESKTOP_BREAKPOINT = 768
export const KANBAN_DRAWER_DESKTOP_WIDTH = 416
export const KANBAN_HORIZONTAL_GUTTER = 32

export interface KanbanViewportGeometry {
  desktop: boolean
  drawerWidth: number
  laneWidth: number
  viewportWidth: number
}

export function kanbanViewportGeometry(viewportWidth: number): KanbanViewportGeometry {
  const width = Math.max(0, viewportWidth)
  const desktop = width >= KANBAN_DESKTOP_BREAKPOINT

  return {
    desktop,
    drawerWidth: desktop ? Math.min(KANBAN_DRAWER_DESKTOP_WIDTH, width) : width,
    laneWidth: desktop ? 256 : Math.max(0, width - KANBAN_HORIZONTAL_GUTTER),
    viewportWidth: width
  }
}

export function useKanbanViewportGeometry(): KanbanViewportGeometry {
  const [geometry, setGeometry] = useState(() => kanbanViewportGeometry(window.innerWidth))

  useEffect(() => {
    const update = () => setGeometry(kanbanViewportGeometry(window.innerWidth))

    window.addEventListener('resize', update)

    return () => window.removeEventListener('resize', update)
  }, [])

  return geometry
}
