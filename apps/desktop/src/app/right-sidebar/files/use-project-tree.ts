import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback, useEffect, useMemo } from 'react'

import { desktopFsCacheKey } from '@/lib/desktop-fs'
import { $connection } from '@/store/session'
import { $workspaceChangeTick, consumeWorkspaceChange } from '@/store/workspace-events'

import { clearProjectDirCache, type ProjectTreeEntry, readProjectDir } from './ipc'
import { $showIgnoredRoots, setShowIgnoredFiles, showsIgnoredFiles } from './prefs'

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

function readError(cause: unknown): string {
  return cause instanceof Error && cause.message ? cause.message : 'read-error'
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
  /** True while this project's gitignored entries are displayed. */
  showIgnored: boolean
  collapseAll: () => void
  loadChildren: (id: string) => Promise<void>
  refreshRoot: () => Promise<void>
  setNodeOpen: (id: string, open: boolean) => void
  setShowIgnored: (show: boolean) => void
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
async function fallbackRootFor(cwd: string, sourceIsRemote: boolean): Promise<string | null> {
  if (sourceIsRemote) {
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

async function loadRoot(
  cwd: string,
  {
    connectionKey = desktopFsCacheKey(),
    force = false,
    reset = false
  }: { connectionKey?: string; force?: boolean; reset?: boolean } = {}
) {
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

  // Re-reading the SAME root — the error retry below, the refresh button, a
  // workspace revalidation — must probe underneath what is on screen rather
  // than blanking it first. Clearing here meant an unreadable root strobed
  // "unreadable" → blank → "unreadable" every retry, forever, while the app
  // just sat there; the header's project name flickered with it as
  // `resolvedCwd` dropped back to the raw cwd and returned. Only a different
  // root (or a different backend, via `reset`) may clear, where the previous
  // project's tree would otherwise linger under the new one.
  const keepVisible = current.cwd === cwd && !reset

  $projectTree.set({
    collapseNonce: current.collapseNonce,
    cwd,
    data: keepVisible ? current.data : [],
    loaded: false,
    openState: keepVisible ? current.openState : {},
    requestId,
    resolvedCwd: keepVisible ? current.resolvedCwd : '',
    rootError: keepVisible ? current.rootError : null,
    rootLoading: true
  })

  const sourceIsRemote = $connection.get()?.mode === 'remote'
  let resolvedCwd = cwd
  let entries: ProjectTreeEntry[] = []
  let error: string | undefined

  try {
    ;({ entries, error } = await readProjectDir(cwd, cwd))

    if (error && desktopFsCacheKey() === connectionKey) {
      const fallback = await fallbackRootFor(cwd, sourceIsRemote)

      if (fallback) {
        const retry = await readProjectDir(fallback, fallback)

        if (!retry.error) {
          resolvedCwd = fallback
          entries = retry.entries
          error = undefined
        }
      }
    }
  } catch (cause) {
    error = readError(cause)
  }

  setProjectTree(latest => {
    if (latest.cwd !== cwd || latest.requestId !== requestId || desktopFsCacheKey() !== connectionKey) {
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
async function revalidateTree(
  cwd: string,
  change: { dirs: string[]; full: boolean },
  connectionKey: string
): Promise<void> {
  const state = $projectTree.get()

  if (!cwd || state.cwd !== cwd || !state.loaded || desktopFsCacheKey() !== connectionKey) {
    return
  }

  const rootPath = state.resolvedCwd || cwd
  // The gitignore filter is user-switchable, so a listing read under the old
  // preference must not be committed after a toggle flipped it — it would
  // re-hide entries the toggle just revealed. Snapshot the policy, re-check it
  // on commit, same shape as the requestId / connectionKey guards.
  const filterAtRead = showsIgnoredFiles(rootPath)

  if (!change.full && change.dirs.length) {
    // Only re-read changed dirs that are actually loaded (root, or an expanded
    // folder); a change inside a collapsed/absent dir isn't visible → skip.
    const targets = change.dirs.filter(dir => dir === rootPath || findNode(state.data, dir)?.children)

    if (!targets.length) {
      return
    }

    const reads = await Promise.all(targets.map(async dir => ({ dir, ...(await readProjectDir(dir, rootPath)) })))

    setProjectTree(latest => {
      if (
        latest.cwd !== cwd ||
        !latest.loaded ||
        desktopFsCacheKey() !== connectionKey ||
        showsIgnoredFiles(rootPath) !== filterAtRead
      ) {
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

  setProjectTree(latest =>
    latest.cwd === cwd &&
    latest.loaded &&
    desktopFsCacheKey() === connectionKey &&
    showsIgnoredFiles(rootPath) === filterAtRead
      ? { ...latest, data: nextData }
      : latest
  )
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
  const connectionKey = desktopFsCacheKey(connection)
  // Subscribed so a toggle re-renders the header; the roots list itself is read
  // through showsIgnoredFiles so every caller shares one resolver.
  useStore($showIgnoredRoots)

  const effectiveCwd = state.cwd === cwd && state.resolvedCwd ? state.resolvedCwd : cwd
  const showIgnored = showsIgnoredFiles(effectiveCwd)

  const refreshRoot = useCallback(() => loadRoot(cwd, { connectionKey, force: true }), [connectionKey, cwd])

  // Flipping the filter changes what every already-read directory contains, so
  // every loaded listing is stale by construction. `loadRoot(force)` is the
  // wrong instrument: it rebuilds `data` from the root listing alone, so an
  // expanded subtree loses its children while arborist keeps the row open —
  // the folder renders empty and nothing re-fetches it. The full reconcile
  // re-reads every loaded dir recursively and merges by path id, so expansion
  // and the subtrees on screen both survive.
  const setShowIgnored = useCallback(
    (show: boolean) => {
      if (!setShowIgnoredFiles(effectiveCwd, show)) {
        return
      }

      void revalidateTree(cwd, { dirs: [], full: true }, connectionKey)
    },
    [connectionKey, cwd, effectiveCwd]
  )

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
      const inflightKey = `${connectionKey}:${id}`

      if (!cwd || inflight.has(inflightKey)) {
        return
      }

      inflight.add(inflightKey)

      setProjectTree(current => {
        if (current.cwd !== cwd || desktopFsCacheKey() !== connectionKey) {
          return current
        }

        return {
          ...current,
          data: patchNode(current.data, id, n => ({ ...n, loading: true, children: [placeholderChild(n.id)] }))
        }
      })

      const rootPath = $projectTree.get().resolvedCwd || cwd
      const filterAtRead = showsIgnoredFiles(rootPath)
      let entries: ProjectTreeEntry[] = []
      let error: string | undefined

      try {
        ;({ entries, error } = await readProjectDir(id, rootPath))
      } catch (cause) {
        error = readError(cause)
      } finally {
        inflight.delete(inflightKey)
      }

      setProjectTree(current => {
        // Same filter guard as revalidateTree: a child listing read before a
        // show-ignored toggle must not land after it.
        if (
          current.cwd !== cwd ||
          desktopFsCacheKey() !== connectionKey ||
          showsIgnoredFiles(rootPath) !== filterAtRead
        ) {
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
    [connectionKey, cwd]
  )

  // Live, non-destructive refresh when the agent touches the tree (skip the
  // very first render: tick 0 is the initial value, not a real change).
  useEffect(() => {
    if (workspaceTick > 0) {
      void revalidateTree(cwd, consumeWorkspaceChange(), connectionKey)
    }
  }, [connectionKey, cwd, workspaceTick])

  useEffect(() => {
    const connectionChanged = lastConnectionKey !== '' && lastConnectionKey !== connectionKey
    lastConnectionKey = connectionKey

    if (connectionChanged) {
      clearProjectDirCache()
      // Same path, different machine: what is on screen was read from the old
      // backend, so it has to go even though the cwd string is unchanged.
      void loadRoot(cwd, { connectionKey, force: true, reset: true })

      return
    }

    void loadRoot(cwd, { connectionKey })
  }, [connectionKey, cwd])

  // Self-heal: an errored root re-probes every few seconds while the tree is
  // mounted. Each attempt bumps requestId, so a persistent error re-arms the
  // timer; a success clears rootError and stops it.
  useEffect(() => {
    if (!cwd || state.cwd !== cwd || !state.rootError) {
      return
    }

    const timer = window.setTimeout(() => void loadRoot(cwd, { connectionKey, force: true }), ROOT_ERROR_RETRY_MS)

    return () => window.clearTimeout(timer)
  }, [connectionKey, cwd, state.cwd, state.requestId, state.rootError])

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
      void readProjectDir(cwd, cwd)
        .then(({ error }) => {
          if (!cancelled && !error && desktopFsCacheKey() === connectionKey) {
            void loadRoot(cwd, { connectionKey, force: true })
          }
        })
        .catch(() => undefined)
    }, ROOT_ERROR_RETRY_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [connectionKey, cwd, usingFallback])

  return useMemo(
    () => ({
      collapseAll,
      collapseNonce: state.cwd === cwd ? state.collapseNonce : 0,
      data: state.cwd === cwd ? state.data : [],
      effectiveCwd,
      loadChildren,
      openState: state.cwd === cwd ? state.openState : {},
      refreshRoot,
      rootError: state.cwd === cwd ? state.rootError : null,
      rootLoading: state.cwd === cwd ? state.rootLoading : Boolean(cwd),
      setNodeOpen,
      setShowIgnored,
      showIgnored
    }),
    [
      collapseAll,
      cwd,
      effectiveCwd,
      loadChildren,
      refreshRoot,
      setNodeOpen,
      setShowIgnored,
      showIgnored,
      state.collapseNonce,
      state.cwd,
      state.data,
      state.openState,
      state.rootError,
      state.rootLoading
    ]
  )
}
