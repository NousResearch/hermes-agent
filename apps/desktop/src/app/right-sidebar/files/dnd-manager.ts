import { createDragDropManager, type DragDropManager } from 'dnd-core'
import { HTML5Backend } from 'react-dnd-html5-backend'

let manager: DragDropManager | null = null

/**
 * A single, app-lifetime react-dnd manager for the file tree.
 *
 * react-arborist mounts its own react-dnd `DndProvider` with `HTML5Backend`
 * inside every `<Tree>`. react-dnd v14 stores that provider's manager on a
 * global, ref-counted singleton context and nulls it when the count hits 0.
 * On a keyed remount (cwd / collapse changes force a fresh `<Tree>`), the
 * singleton can be torn down and recreated while the previous `HTML5Backend`
 * still owns the `window.__isReactDndHtml5Backend` setup flag — so the new
 * backend's `setup()` throws "Cannot have two HTML5 backends at the same
 * time." and trips the file-tree error boundary (it never recovers, because
 * "Try again" just remounts into the same race).
 *
 * Passing arborist a stable `dndManager` makes it skip the global-singleton
 * path entirely and reuse one backend for the lifetime of the app, so the
 * window flag is never double-claimed.
 */
export function getFileTreeDndManager(): DragDropManager {
  manager ??= createDragDropManager(HTML5Backend)

  return manager
}
