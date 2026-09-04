import type { ReactNode } from 'react'

export function StarmapContextMenuBoundary({ children }: { children: ReactNode }) {
  // The Star Map owns plain right-clicks so its canvas-level handler can open
  // NodeContextMenu. The app-wide capture handler still handles owned targets
  // (links, images, editables, selections) inside this boundary.
  return (
    <div className="contents" data-context-menu-skip="">
      {children}
    </div>
  )
}
