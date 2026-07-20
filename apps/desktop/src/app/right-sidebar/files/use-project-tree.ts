import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo } from 'react'

import { $connection } from '@/store/session'
import { $workspaceChangeTick, consumeWorkspaceChange } from '@/store/workspace-events'

import { clearProjectDirCache, type ProjectTreeEntry, readProjectDir } from './ipc'

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

function findNode(nodes: TreeNode[], id: string): null | TreeNode {
  for (const node of nodes) {
    if (node.id === id) {
      return node
    }

    if (node.children?.length) {
      const hit = findNode(node.children, id)

      if (hit) {
        return hit
      }
    }
  }

  return null
}

// Merge a freshly-read dir's entries into its existing children: keep surviving
// nodes (subtrees intact), add new, drop deleted. Non-recursive — a grandchild
// dir only re-reads when it's itself in the change set.
function mergeChildren(existing: TreeNode[], entries: ProjectTreeEntry[]): TreeNode[] {
  const byId = new Map(existing.filter(node => !node.placeholder).map(node => [node.id, node]))

  return entries.map(entry => byId.get(entry.path) ?? makeNode(entry.path, entry.name, entry.isDirectory))
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
let nextRootRequestId = 0
let lastConnectionKey = ''

// While the root is errored (ENOENT during a session's cwd race, a folder that
// reappears after a checkout, a remote that wasn't ready), keep retrying on a
// slow cadence so the tree self-heals instead of staying "UNREADABLE" forever.
const ROOT_ERROR_RETRY_MS = 3_000

function setProjectTree(updater: (current: ProjectTreeState) => ProjectTreeState) {
  $projectTree.set(updater($projectTree.get()))
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

  const current = $projectTree.get()

  if (!force && current.cwd === cwd && (current.loaded || current.rootLoading)) {
    return
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
    data: [],
    loaded: false,
    openState: current.cwd === cwd ? current.openState : {},
    requestId,
    resolvedCwd: '',
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
  clearProjectTree()
  clearProjectDirCache()
}

// Non-destructive live refresh as the agent edits: preserves expansion + loaded
// subtrees (stable absolute-path ids let rows animate in/out), never collapses.
// Targeted by default — re-reads only the changed dirs in `change`; the root and
// untouched folders never touch the filesystem or re-render. Falls back to
// re-reading every loaded dir only when the mutation is opaque (a terminal
// command / a path we couldn't resolve) — see store/workspace-events.
async function revalidateTree(cwd: string, change: { dirs: string[]; full: boolean }): Promise<void> {
  const state = $projectTree.get()

  if (!cwd || state.cwd !== cwd || !state.loaded) {
    return
  }

  const rootPath = state.resolvedCwd || cwd

  if (!change.full && change.dirs.length) {
    // Only re-read changed dirs that are actually loaded (root, or an expanded
    // folder); a change inside a collapsed/absent dir isn't visible → skip.
    const targets = change.dirs.filter(dir => dir === rootPath || findNode(state.data, dir)?.children)

    if (!targets.length) {
      return
    }

    const reads = await Promise.all(targets.map(async dir => ({ dir, ...(await readProjectDir(dir, rootPath)) })))

    setProjectTree(latest => {
      if (latest.cwd !== cwd || !latest.loaded) {
        return latest
      }

      let data = latest.data

      for (const { dir, entries, error } of reads) {
        if (error) {
          continue // keep last-known children on a transient read error
        }

        data =
          dir === rootPath
            ? mergeChildren(data, entries)
            : patchNode(data, dir, node =>
                node.children ? { ...node, children: mergeChildren(node.children, entries) } : node
              )
      }

      return data === latest.data ? latest : { ...latest, data }
    })

    return
  }

  // Opaque fallback: reconcile every loaded dir. Siblings read concurrently
  // (Promise.all keeps order); loaded subfolders recurse.
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

        return prev ?? makeNode(entry.path, entry.name, entry.isDirectory)
      })
    )
  }

  const nextData = await reconcile(rootPath, state.data)

  setProjectTree(latest => (latest.cwd === cwd && latest.loaded ? { ...latest, data: nextData } : latest))
}

/**
 * Lazy-loads a directory tree rooted at `cwd`. Children are fetched on first
 * expand and cached in this feature-owned atom so unrelated chat rerenders or
 * remounts cannot reset the browser. A placeholder leaf renders so the
 * disclosure caret shows for unloaded folders. `refreshRoot` invalidates the
 * whole tree (used after cwd change or manual refresh).
 */
export function useProjectTree(cwd: string): UseProjectTreeResult {
  const state = useStore($projectTree)
  const connection = useStore($connection)
  const workspaceTick = useStore($workspaceChangeTick)
  const connectionKey = `${connection?.mode || 'local'}:${connection?.profile || ''}:${connection?.baseUrl || ''}`

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
      void revalidateTree(cwd, consumeWorkspaceChange())
    }
  }, [workspaceTick, cwd])

  useEffect(() => {
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
