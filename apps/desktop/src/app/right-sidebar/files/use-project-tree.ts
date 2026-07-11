import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo } from 'react'

import { $connection } from '@/store/session'
import { $workspaceChangeTick } from '@/store/workspace-events'

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
const snapshots = new Map<string, ProjectTreeState>()
const SNAPSHOT_LIMIT = 24
let nextRootRequestId = 0
let lastConnectionKey = ''
let activeConnectionKey = 'local:'
let revalidation: Promise<void> | null = null
let revalidateQueued = false

// While the root is errored (ENOENT during a session's cwd race, a folder that
// reappears after a checkout, a remote that wasn't ready), keep retrying on a
// slow cadence so the tree self-heals instead of staying "UNREADABLE" forever.
const ROOT_ERROR_RETRY_MS = 3_000

function rememberSnapshot(key: string, state: ProjectTreeState): void {
  snapshots.delete(key)
  snapshots.set(key, state)

  while (snapshots.size > SNAPSHOT_LIMIT) {
    const oldest = snapshots.keys().next().value

    if (oldest === undefined) {
      break
    }

    snapshots.delete(oldest)
  }
}

function readSnapshot(key: string): ProjectTreeState | undefined {
  const snapshot = snapshots.get(key)

  if (snapshot) {
    snapshots.delete(key)
    snapshots.set(key, snapshot)
  }

  return snapshot
}

function setProjectTree(updater: (current: ProjectTreeState) => ProjectTreeState) {
  const next = updater($projectTree.get())
  $projectTree.set(next)

  if (next.cwd && next.loaded) {
    rememberSnapshot(`${activeConnectionKey}:${next.cwd}`, next)
  }
}

function clearProjectTree() {
  nextRootRequestId += 1
  inflight.clear()
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
  if (!cwd) {
    clearProjectTree()

    return
  }

  let current = $projectTree.get()

  if (!force && current.cwd === cwd && (current.loaded || current.rootLoading)) {
    return
  }

  const cached = readSnapshot(`${activeConnectionKey}:${cwd}`)

  if (!force && current.cwd !== cwd && cached) {
    current = { ...cached, rootLoading: true }
    $projectTree.set(current)
  }

  const requestId = nextRootRequestId + 1
  nextRootRequestId = requestId
  inflight.clear()

  if (force || current.cwd !== cwd) {
    clearProjectDirCache(cwd)
  }

  $projectTree.set({
    collapseNonce: current.collapseNonce,
    cwd,
    data: current.cwd === cwd ? current.data : [],
    loaded: current.cwd === cwd && current.loaded,
    openState: current.cwd === cwd ? current.openState : {},
    requestId,
    resolvedCwd: current.cwd === cwd ? current.resolvedCwd : '',
    rootError: null,
    rootLoading: true
  })

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

  setProjectTree(latest => {
    if (latest.cwd !== cwd || latest.requestId !== requestId) {
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
  snapshots.clear()
  revalidation = null
  revalidateQueued = false
  clearProjectTree()
  clearProjectDirCache()
}

// Non-destructive refresh: re-read every currently-loaded directory and merge
// entries (add new files/folders, drop deleted ones) while preserving expansion
// and already-loaded subtrees. Unlike `loadRoot({force})` this never collapses
// the tree, so it's safe to run live as the agent edits — and because node ids
// (absolute paths) stay stable across merges, rows can animate in/out.
async function runRevalidateTree(cwd: string): Promise<void> {
  const state = $projectTree.get()

  if (!cwd || state.cwd !== cwd || !state.loaded) {
    return
  }

  const rootPath = state.resolvedCwd || cwd
  const requestId = state.requestId
  clearProjectDirCache()

  const reconcile = async (dirPath: string, existing: TreeNode[]): Promise<TreeNode[]> => {
    const { entries, error } = await readProjectDir(dirPath, rootPath)

    if (error) {
      return existing
    }

    const byId = new Map(existing.filter(node => !node.placeholder).map(node => [node.id, node]))

    return Promise.all(
      entries.map(async entry => {
        const prev = byId.get(entry.path)

        if (prev?.isDirectory && prev.children) {
          return { ...prev, children: await reconcile(prev.id, prev.children) }
        }

        return prev || makeNode(entry.path, entry.name, entry.isDirectory)
      })
    )
  }

  const nextData = await reconcile(rootPath, state.data)

  setProjectTree(latest =>
    latest.cwd === cwd && latest.requestId === requestId && latest.loaded ? { ...latest, data: nextData } : latest
  )
}

async function revalidateTree(cwd: string): Promise<void> {
  if (revalidation) {
    revalidateQueued = true

    return revalidation
  }

  revalidation = runRevalidateTree(cwd).finally(() => {
    revalidation = null

    if (revalidateQueued) {
      revalidateQueued = false
      void revalidateTree($projectTree.get().cwd)
    }
  })

  return revalidation
}

/**
 * Lazy-loads a directory tree rooted at `cwd`. Children are fetched on first
 * expand and cached in this feature-owned atom so unrelated chat rerenders or
 * remounts cannot reset the browser. A placeholder leaf renders so the
 * disclosure caret shows for unloaded folders. `refreshRoot` invalidates the
 * whole tree (used after cwd change or manual refresh).
 */
export function useProjectTree(cwd: string, sessionKey = 'detached'): UseProjectTreeResult {
  const state = useStore($projectTree)
  const connection = useStore($connection)
  const workspaceTick = useStore($workspaceChangeTick)
  const connectionKey = `${sessionKey}:${connection?.mode || 'local'}:${connection?.profile || ''}:${connection?.baseUrl || ''}`

  const refreshRoot = useCallback(() => loadRoot(cwd, { force: true }), [cwd])

  const setNodeOpen = useCallback(
    (id: string, open: boolean) => {
      setProjectTree(current => {
        if (current.cwd !== cwd || current.openState[id] === open) {
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
    [cwd]
  )

  // Clears the recorded open state and bumps the nonce; the tree is keyed on
  // the nonce so it remounts with everything collapsed (loaded children stay
  // cached in `data`, just hidden).
  const collapseAll = useCallback(() => {
    setProjectTree(current => {
      if (current.cwd !== cwd) {
        return current
      }

      return { ...current, collapseNonce: current.collapseNonce + 1, openState: {} }
    })
  }, [cwd])

  const loadChildren = useCallback(
    async (id: string) => {
      if (!cwd || inflight.has(id)) {
        return
      }

      inflight.add(id)

      setProjectTree(current => {
        if (current.cwd !== cwd) {
          return current
        }

        return {
          ...current,
          data: patchNode(current.data, id, n => ({ ...n, loading: true, children: [placeholderChild(n.id)] }))
        }
      })

      const rootPath = $projectTree.get().resolvedCwd || cwd
      const { entries, error } = await readProjectDir(id, rootPath)

      inflight.delete(id)

      setProjectTree(current => {
        if (current.cwd !== cwd) {
          return current
        }

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
    [cwd]
  )

  // Live, non-destructive refresh when the agent touches the tree (skip the
  // very first render: tick 0 is the initial value, not a real change).
  useEffect(() => {
    if (workspaceTick > 0) {
      void revalidateTree(cwd)
    }
  }, [workspaceTick, cwd])

  useEffect(() => {
    activeConnectionKey = connectionKey
    const connectionChanged = lastConnectionKey !== '' && lastConnectionKey !== connectionKey
    lastConnectionKey = connectionKey

    if (connectionChanged) {
      clearProjectDirCache()
      void loadRoot(cwd, { force: true })

      return
    }

    void loadRoot(cwd)
  }, [connectionKey, cwd])

  // Self-heal: an errored root re-probes every few seconds while the tree is
  // mounted. Each attempt bumps requestId, so a persistent error re-arms the
  // timer; a success clears rootError and stops it.
  useEffect(() => {
    if (!cwd || state.cwd !== cwd || !state.rootError) {
      return
    }

    const timer = window.setTimeout(() => void loadRoot(cwd, { force: true }), ROOT_ERROR_RETRY_MS)

    return () => window.clearTimeout(timer)
  }, [cwd, state.cwd, state.requestId, state.rootError])

  // While showing the fallback root, quietly re-probe the session's real cwd
  // (a worktree re-created, a checkout restored) and switch back when it
  // reappears. The probe never touches state, so there's no flicker.
  const usingFallback = state.cwd === cwd && Boolean(state.resolvedCwd) && state.resolvedCwd !== cwd

  useEffect(() => {
    if (!cwd || !usingFallback) {
      return
    }

    let cancelled = false

    const timer = window.setInterval(() => {
      void readProjectDir(cwd, cwd).then(({ error }) => {
        if (!cancelled && !error) {
          void loadRoot(cwd, { force: true })
        }
      })
    }, ROOT_ERROR_RETRY_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [cwd, usingFallback])

  return useMemo(
    () => ({
      collapseAll,
      collapseNonce: state.cwd === cwd ? state.collapseNonce : 0,
      data: state.cwd === cwd ? state.data : [],
      effectiveCwd: state.cwd === cwd && state.resolvedCwd ? state.resolvedCwd : cwd,
      loadChildren,
      openState: state.cwd === cwd ? state.openState : {},
      refreshRoot,
      rootError: state.cwd === cwd ? state.rootError : null,
      rootLoading: state.cwd === cwd ? state.rootLoading : Boolean(cwd),
      setNodeOpen
    }),
    [
      collapseAll,
      cwd,
      loadChildren,
      refreshRoot,
      setNodeOpen,
      state.collapseNonce,
      state.cwd,
      state.data,
      state.openState,
      state.resolvedCwd,
      state.rootError,
      state.rootLoading
    ]
  )
}
