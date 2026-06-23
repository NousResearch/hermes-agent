import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo } from 'react'

import { normalizeWorkspacePath } from '@/lib/workspace-key'
import { $connection } from '@/store/session'

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
  /** Synthetic loading/error rows are not real filesystem entries. */
  placeholder?: 'error' | 'loading'
  /** Last error code from readDir (e.g. EACCES). Cleared on next successful load. */
  error?: string
}

const PLACEHOLDER_ID = '__loading__'
const ERROR_PLACEHOLDER_ID = '__error__'

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
  return { id: `${parentId}::${PLACEHOLDER_ID}`, isDirectory: false, name: 'Loading…', placeholder: 'loading' }
}

function errorChild(parentId: string, error: string | undefined): TreeNode {
  return {
    id: `${parentId}::${ERROR_PLACEHOLDER_ID}`,
    isDirectory: false,
    name: `Unable to read (${error || 'read-error'})`,
    placeholder: 'error'
  }
}

export interface UseProjectTreeResult {
  /** Bumped by collapseAll so callers can remount the tree fully collapsed. */
  collapseNonce: number
  data: TreeNode[]
  /** Directory actually displayed — differs from the requested cwd when the
   *  session's recorded cwd no longer exists and we fell back to the default
   *  workspace dir. */
  effectiveCwd: string
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
  /** Directory the displayed entries were read from ('' until first load). */
  resolvedCwd: string
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
  resolvedCwd: '',
  rootError: null,
  rootLoading: false
}

const inflight = new Set<string>()
const $projectTree = atom<ProjectTreeState>(initialState)
const projectTrees = new Map<string, ProjectTreeState>()
let nextRootRequestId = 0
let lastConnectionKey = ''

// While the root is errored (ENOENT during a session's cwd race, a folder that
// reappears after a checkout, a remote that wasn't ready), keep retrying on a
// slow cadence so the tree self-heals instead of staying "UNREADABLE" forever.
const ROOT_ERROR_RETRY_MS = 3_000

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

/** Sessions record their launch cwd; deleted worktrees and remote-backend
 *  paths arrive here as directories that don't exist on this machine. Rather
 *  than bricking the tree, display the sanitized workspace fallback (main
 *  prefers the configured default project dir). Local connections only —
 *  remote trees are read through the remote bridge. */
async function fallbackRootFor(cwd: string): Promise<string | null> {
  if ($connection.get()?.mode === 'remote') {
    return null
  }

  const sanitize = window.hermesDesktop?.sanitizeWorkspaceCwd

  if (!sanitize) {
    return null
  }

  try {
    const { cwd: fallback, sanitized } = await sanitize(cwd)

    return sanitized && fallback && fallback !== cwd ? fallback : null
  } catch {
    return null
  }
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
    resolvedCwd: '',
    rootError: null,
    rootLoading: true
  }

  projectTrees.set(cwd, next)
  $projectTree.set(next)

  let resolvedCwd = cwd
  let { entries, error } = await readProjectDir(cwd, cwd)

  if (error) {
    const fallback = await fallbackRootFor(cwd)

    if (fallback) {
      const retry = await readProjectDir(fallback, fallback)

      if (!retry.error) {
        resolvedCwd = fallback
        entries = retry.entries
        error = undefined
      }
    }
  }

  setWorkspaceTree(cwd, latest => {
    if (latest.requestId !== requestId) {
      return latest
    }

    return {
      ...latest,
      data: error ? [] : entries.map(e => makeNode(e.path, e.name, e.isDirectory)),
      loaded: true,
      resolvedCwd,
      rootError: error || null,
      rootLoading: false
    }
  })
}

export function resetProjectTreeState() {
  lastConnectionKey = ''
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
  const connection = useStore($connection)
  const connectionKey = `${connection?.mode || 'local'}:${connection?.profile || ''}:${connection?.baseUrl || ''}`

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

      // Use the resolved fallback cwd (if any) so children are read relative to
      // the directory actually displayed, while keeping workspace normalization.
      const rootPath = normalizeWorkspacePath($projectTree.get().resolvedCwd || workspaceCwd)
      const { entries, error } = await readProjectDir(id, rootPath)

      inflight.delete(inflightKey)

      setWorkspaceTree(workspaceCwd, current => {
        return {
          ...current,
          data: patchNode(current.data, id, n => ({
            ...n,
            loading: false,
            error: error || undefined,
            children: error ? [errorChild(n.id, error)] : entries.map(e => makeNode(e.path, e.name, e.isDirectory))
          }))
        }
      })
    },
    [workspaceCwd]
  )

  useEffect(() => {
    const connectionChanged = lastConnectionKey !== '' && lastConnectionKey !== connectionKey
    lastConnectionKey = connectionKey

    if (connectionChanged) {
      clearProjectDirCache()
      void loadRoot(workspaceCwd, { force: true })

      return
    }

    void loadRoot(workspaceCwd)
  }, [connectionKey, workspaceCwd])

  // Self-heal: an errored root re-probes every few seconds while the tree is
  // mounted. Each attempt bumps requestId, so a persistent error re-arms the
  // timer; a success clears rootError and stops it.
  useEffect(() => {
    if (!workspaceCwd || state.cwd !== workspaceCwd || !state.rootError) {
      return
    }

    const timer = window.setTimeout(() => void loadRoot(workspaceCwd, { force: true }), ROOT_ERROR_RETRY_MS)

    return () => window.clearTimeout(timer)
  }, [workspaceCwd, state.cwd, state.requestId, state.rootError])

  // While showing the fallback root, quietly re-probe the session's real cwd
  // (a worktree re-created, a checkout restored) and switch back when it
  // reappears. The probe never touches state, so there's no flicker.
  const usingFallback = state.cwd === workspaceCwd && Boolean(state.resolvedCwd) && state.resolvedCwd !== workspaceCwd

  useEffect(() => {
    if (!workspaceCwd || !usingFallback) {
      return
    }

    let cancelled = false

    const timer = window.setInterval(() => {
      void readProjectDir(workspaceCwd, workspaceCwd).then(({ error }) => {
        if (!cancelled && !error) {
          void loadRoot(workspaceCwd, { force: true })
        }
      })
    }, ROOT_ERROR_RETRY_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [workspaceCwd, usingFallback])

  return useMemo(
    () => ({
      collapseAll,
      collapseNonce: state.cwd === workspaceCwd ? state.collapseNonce : 0,
      data: state.cwd === workspaceCwd ? state.data : [],
      effectiveCwd: state.cwd === workspaceCwd && state.resolvedCwd ? state.resolvedCwd : workspaceCwd,
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
      state.resolvedCwd,
      state.rootError,
      state.rootLoading,
      workspaceCwd
    ]
  )
}
