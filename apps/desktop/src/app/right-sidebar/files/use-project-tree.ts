import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo } from 'react'

import { normalizeWorkspacePath } from '@/lib/workspace-key'

import { clearProjectDirCache, readProjectDir } from './ipc'

export interface TreeNode {
  /** Absolute filesystem path. Doubles as react-arborist node id. */
  id: string
  name: string
  /** Drives arborist's leaf-vs-expandable decision via childrenAccessor. */
  isDirectory: boolean
  /** `undefined` = directory, children not yet loaded. `[]` = loaded empty. */
  children?: TreeNode[]
  /** True while a readDir for this folder is in flight. */
  loading?: boolean
  /** Last error code from readDir (e.g. EACCES). Cleared on next successful load. */
  error?: string
}

const PLACEHOLDER_ID = '__loading__'

function makeNode(path: string, name: string, isDirectory: boolean): TreeNode {
  return { id: path, isDirectory, name }
}

function patchNode(nodes: TreeNode[] | undefined | null, id: string, patch: (n: TreeNode) => TreeNode): TreeNode[] {
  if (!nodes) {
    return []
  }

  return nodes.map(n => {
    if (n.id === id) {
      return patch(n)
    }

    if (n.children && n.children.length > 0) {
      return { ...n, children: patchNode(n.children, id, patch) }
    }

    return n
  })
}

function placeholderChild(parentId: string): TreeNode {
  return { id: `${parentId}::${PLACEHOLDER_ID}`, isDirectory: false, name: 'Loading…' }
}

export interface UseProjectTreeResult {
  /** Bumped by collapseAll so callers can remount the tree fully collapsed. */
  collapseNonce: number
  data: TreeNode[]
  openState: Record<string, boolean>
  rootError: string | null
  rootLoading: boolean
  collapseAll: () => void
  loadChildren: (id: string) => Promise<void>
  refreshRoot: () => Promise<void>
  setNodeOpen: (id: string, open: boolean) => void
}

interface ProjectTreeState {
  collapseNonce: number
  cwd: string
  data: TreeNode[]
  loaded: boolean
  openState: Record<string, boolean>
  requestId: number
  rootError: string | null
  rootLoading: boolean
}

const initialState: ProjectTreeState = {
  collapseNonce: 0,
  cwd: '',
  data: [],
  loaded: false,
  openState: {},
  requestId: 0,
  rootError: null,
  rootLoading: false
}

const inflight = new Set<string>()
const $projectTree = atom<ProjectTreeState>(initialState)
const projectTrees = new Map<string, ProjectTreeState>()
let nextRootRequestId = 0

function setProjectTree(updater: (current: ProjectTreeState) => ProjectTreeState) {
  const next = updater($projectTree.get())

  $projectTree.set(next)

  if (next.cwd) {
    projectTrees.set(next.cwd, next)
  }
}

function setWorkspaceTree(cwd: string, updater: (current: ProjectTreeState) => ProjectTreeState) {
  const current = projectTrees.get(cwd)

  if (!current) {
    return
  }

  const next = updater(current)

  projectTrees.set(cwd, next)

  if ($projectTree.get().cwd === cwd) {
    $projectTree.set(next)
  }
}

function clearProjectTree() {
  nextRootRequestId += 1
  inflight.clear()
  projectTrees.clear()
  $projectTree.set({ ...initialState, requestId: nextRootRequestId })
}

async function loadRoot(cwd: string, { force = false }: { force?: boolean } = {}) {
  cwd = normalizeWorkspacePath(cwd)

  if (!cwd) {
    $projectTree.set({ ...initialState, requestId: nextRootRequestId })

    return
  }

  const current = $projectTree.get()

  if (!force && current.cwd === cwd && (current.loaded || current.rootLoading)) {
    return
  }

  const cached = projectTrees.get(cwd)

  if (!force && cached && (cached.loaded || cached.rootLoading)) {
    $projectTree.set(cached)

    return
  }

  const requestId = nextRootRequestId + 1
  nextRootRequestId = requestId

  if (force) {
    clearProjectDirCache(cwd)
  }

  const next: ProjectTreeState = {
    collapseNonce: cached?.collapseNonce ?? 0,
    cwd,
    data: [],
    loaded: false,
    openState: cached?.openState ?? {},
    requestId,
    rootError: null,
    rootLoading: true
  }

  projectTrees.set(cwd, next)
  $projectTree.set(next)

  const { entries, error } = await readProjectDir(cwd, cwd)

  setWorkspaceTree(cwd, latest => {
    if (latest.requestId !== requestId) {
      return latest
    }

    return {
      ...latest,
      data: error ? [] : entries.map(e => makeNode(e.path, e.name, e.isDirectory)),
      loaded: true,
      rootError: error || null,
      rootLoading: false
    }
  })
}

export function resetProjectTreeState() {
  clearProjectTree()
  clearProjectDirCache()
}

/**
 * Lazy-loads a directory tree rooted at `cwd`. Children are fetched on first
 * expand and cached in this feature-owned atom so unrelated chat rerenders or
 * remounts cannot reset the browser. A placeholder leaf renders so the
 * disclosure caret shows for unloaded folders. `refreshRoot` invalidates the
 * whole tree (used after cwd change or manual refresh).
 */
export function useProjectTree(cwd: string): UseProjectTreeResult {
  const workspaceCwd = normalizeWorkspacePath(cwd)
  const state = useStore($projectTree)

  const refreshRoot = useCallback(() => loadRoot(workspaceCwd, { force: true }), [workspaceCwd])

  const setNodeOpen = useCallback(
    (id: string, open: boolean) => {
      setProjectTree(current => {
        if (current.cwd !== workspaceCwd || current.openState[id] === open) {
          return current
        }

        return {
          ...current,
          openState: {
            ...current.openState,
            [id]: open
          }
        }
      })
    },
    [workspaceCwd]
  )

  // Clears the recorded open state and bumps the nonce; the tree is keyed on
  // the nonce so it remounts with everything collapsed (loaded children stay
  // cached in `data`, just hidden).
  const collapseAll = useCallback(() => {
    setProjectTree(current => {
      if (current.cwd !== workspaceCwd) {
        return current
      }

      return { ...current, collapseNonce: current.collapseNonce + 1, openState: {} }
    })
  }, [workspaceCwd])

  const loadChildren = useCallback(
    async (id: string) => {
      const inflightKey = `${workspaceCwd}\0${id}`

      if (!workspaceCwd || inflight.has(inflightKey)) {
        return
      }

      inflight.add(inflightKey)

      setWorkspaceTree(workspaceCwd, current => {
        return {
          ...current,
          data: patchNode(current.data, id, n => ({ ...n, loading: true, children: [placeholderChild(n.id)] }))
        }
      })

      const { entries, error } = await readProjectDir(id, workspaceCwd)

      inflight.delete(inflightKey)

      setWorkspaceTree(workspaceCwd, current => {
        return {
          ...current,
          data: patchNode(current.data, id, n => ({
            ...n,
            loading: false,
            error: error || undefined,
            children: error ? [] : entries.map(e => makeNode(e.path, e.name, e.isDirectory))
          }))
        }
      })
    },
    [workspaceCwd]
  )

  useEffect(() => {
    void loadRoot(workspaceCwd)
  }, [workspaceCwd])

  return useMemo(
    () => ({
      collapseAll,
      collapseNonce: state.cwd === workspaceCwd ? state.collapseNonce : 0,
      data: state.cwd === workspaceCwd ? state.data : [],
      loadChildren,
      openState: state.cwd === workspaceCwd ? state.openState : {},
      refreshRoot,
      rootError: state.cwd === workspaceCwd ? state.rootError : null,
      rootLoading: state.cwd === workspaceCwd ? state.rootLoading : Boolean(workspaceCwd),
      setNodeOpen
    }),
    [
      collapseAll,
      loadChildren,
      refreshRoot,
      setNodeOpen,
      state.collapseNonce,
      state.cwd,
      state.data,
      state.openState,
      state.rootError,
      state.rootLoading,
      workspaceCwd
    ]
  )
}
