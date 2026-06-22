import type { HermesWorktreeInfo } from '@/global'
import type { SessionInfo } from '@/hermes'

export interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
  // Profile color for the ALL-profiles view; absent for workspace groups.
  color?: null | string
  loadingMore?: boolean
  mode?: 'profile' | 'source' | 'workspace'
  onLoadMore?: () => void
  sourceId?: string
  totalCount?: number
}

const NO_WORKSPACE_ID = '__no_workspace__'

/** Path split into segments, ignoring trailing slashes and mixed separators. */
const segments = (path: string): string[] => path.replace(/[/\\]+$/, '').split(/[/\\]/).filter(Boolean)

/** Last path segment. */
export const baseName = (path: string): string | undefined => segments(path).pop()

/** The segments above the basename. */
const parentSegments = (path: string): string[] => segments(path).slice(0, -1)

interface Labelable {
  id: string
  label: string
  path: null | string
}

/**
 * Disambiguate groups whose basename collides (worktrees all end in the same
 * `apps/desktop`, sibling repos share a folder name, etc.) by walking up the
 * path and prepending parent segments until each colliding label is unique —
 * e.g. `hermes-agent/desktop` vs `hermes-agent-wt-rtl/desktop`. Groups with a
 * unique basename keep their short label untouched.
 */
function disambiguateLabels(groups: Labelable[]): void {
  const byLabel = new Map<string, Labelable[]>()

  for (const group of groups) {
    const bucket = byLabel.get(group.label)

    if (bucket) {
      bucket.push(group)
    } else {
      byLabel.set(group.label, [group])
    }
  }

  for (const bucket of byLabel.values()) {
    if (bucket.length < 2) {
      continue
    }

    // Only groups backed by a real path can grow a prefix; the synthetic
    // "No workspace" group has no path and stays as-is.
    const pathed = bucket.filter(group => group.path)

    if (pathed.length < 2) {
      continue
    }

    const parents = new Map(pathed.map(group => [group.id, parentSegments(group.path!)]))
    let depth = 1

    // Grow the prefix one parent segment at a time until every label in the
    // bucket is distinct, or we run out of parent segments to add.
    while (depth <= Math.max(...pathed.map(g => parents.get(g.id)!.length))) {
      const labels = new Map<string, number>()

      for (const group of pathed) {
        const segs = parents.get(group.id)!
        const prefix = segs.slice(-depth).join('/')
        const base = baseName(group.path!) ?? group.path!
        group.label = prefix ? `${prefix}/${base}` : base
        labels.set(group.label, (labels.get(group.label) ?? 0) + 1)
      }

      if ([...labels.values()].every(count => count === 1)) {
        break
      }

      depth += 1
    }
  }
}

export function workspaceGroupsFor(
  sessions: SessionInfo[],
  noWorkspaceLabel: string,
  options: { preserveSessionOrder?: boolean } = {}
): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const path = session.cwd?.trim() || ''
    const id = path || NO_WORKSPACE_ID
    const label = baseName(path) || path || noWorkspaceLabel

    const group = groups.get(id) ?? { id, label, path: path || null, sessions: [] }
    group.sessions.push(session)
    groups.set(id, group)
  }

  if (!options.preserveSessionOrder) {
    // Groups keep recency order (Map insertion = first-seen in the recency-sorted
    // input, so an active project floats up), but rows *within* a group sort by
    // creation time so they don't reshuffle every time a message lands — keeps
    // muscle memory intact.
    for (const group of groups.values()) {
      group.sessions.sort((a, b) => b.started_at - a.started_at)
    }
  }

  const result = [...groups.values()]
  disambiguateLabels(result)

  return result
}

/**
 * A worktree's main repo and all its linked worktrees collapse into ONE parent
 * (keyed by the repo root); each worktree is a child group; sessions hang off
 * the worktree they ran in. `parent → worktree → sessions`.
 */
export interface SidebarWorkspaceTree {
  id: string
  label: string
  path: null | string
  groups: SidebarSessionGroup[]
  sessionCount: number
}

/** Resolves a session cwd to git-worktree identity (from the local fs probe). */
export type WorktreeResolver = (cwd: string) => HermesWorktreeInfo | null | undefined

interface WorkspacePlacement {
  parentKey: string
  parentLabel: string
  parentPath: string
  worktreeKey: string
  worktreeLabel: string
  worktreePath: string
}

/** Replace a path's final segment, preserving its prefix + separators. */
const withBaseName = (path: string, name: string): string =>
  path.replace(/[/\\]+$/, '').replace(/[^/\\]+$/, name)

/**
 * Path-only fallback for when git metadata is unavailable (remote backends,
 * unreadable paths). Mirrors the git layout: a `<repo>-wt-<branch>` directory
 * nests under its sibling `<repo>`; any other directory is its own repo root.
 */
function placeByHeuristic(path: string): WorkspacePlacement | null {
  const base = baseName(path)

  if (!base) {
    return null
  }

  const worktreeMatch = base.match(/^(.+)-wt-(.+)$/)

  if (worktreeMatch) {
    const repo = worktreeMatch[1]
    const repoPath = withBaseName(path, repo)

    return {
      parentKey: repoPath,
      parentLabel: repo,
      parentPath: repoPath,
      worktreeKey: path,
      worktreeLabel: worktreeMatch[2],
      worktreePath: path
    }
  }

  return {
    parentKey: path,
    parentLabel: base,
    parentPath: path,
    worktreeKey: path,
    worktreeLabel: base,
    worktreePath: path
  }
}

function placeWorkspace(path: string, resolver?: WorktreeResolver): WorkspacePlacement | null {
  const info = resolver?.(path)

  if (info?.repoRoot && info.worktreeRoot) {
    const dirLabel = baseName(info.worktreeRoot) || info.worktreeRoot

    return {
      parentKey: info.repoRoot,
      parentLabel: baseName(info.repoRoot) ?? info.repoRoot,
      parentPath: info.repoRoot,
      worktreeKey: info.worktreeRoot,
      // The main checkout's branch is transient — it changes as you work, so a
      // branch label would misattribute every past session to whatever branch
      // is checked out *now*. Label it by directory. Linked worktrees are
      // per-branch by construction, so branch is the clearest label there.
      worktreeLabel: info.isMainWorktree ? dirLabel : info.branch || dirLabel,
      worktreePath: info.worktreeRoot
    }
  }

  return placeByHeuristic(path)
}

/** Unique, non-empty session cwds — the batch to probe for worktree info. */
export function uniqueCwds(sessions: SessionInfo[]): string[] {
  const seen = new Set<string>()

  for (const session of sessions) {
    const path = session.cwd?.trim()

    if (path) {
      seen.add(path)
    }
  }

  return [...seen]
}

/**
 * Build the `parent → worktree → sessions` tree. Parents keep recency order
 * (first-seen in the recency-sorted input); worktree groups within a parent do
 * too, while rows inside a worktree sort by creation time (stable muscle memory,
 * matching `workspaceGroupsFor`).
 */
export function workspaceTreeFor(
  sessions: SessionInfo[],
  noWorkspaceLabel: string,
  resolver?: WorktreeResolver,
  options: { preserveSessionOrder?: boolean } = {}
): SidebarWorkspaceTree[] {
  interface WorktreeEntry {
    group: SidebarSessionGroup
    parentKey: string
    parentLabel: string
    parentPath: string
  }

  const worktrees = new Map<string, WorktreeEntry>()
  const noWorkspace: SessionInfo[] = []

  for (const session of sessions) {
    const path = session.cwd?.trim() || ''

    if (!path) {
      noWorkspace.push(session)

      continue
    }

    const placement = placeWorkspace(path, resolver)

    if (!placement) {
      noWorkspace.push(session)

      continue
    }

    let entry = worktrees.get(placement.worktreeKey)

    if (!entry) {
      entry = {
        group: { id: placement.worktreeKey, label: placement.worktreeLabel, path: placement.worktreePath, sessions: [] },
        parentKey: placement.parentKey,
        parentLabel: placement.parentLabel,
        parentPath: placement.parentPath
      }
      worktrees.set(placement.worktreeKey, entry)
    }

    entry.group.sessions.push(session)
  }

  if (!options.preserveSessionOrder) {
    for (const entry of worktrees.values()) {
      entry.group.sessions.sort((a, b) => b.started_at - a.started_at)
    }
  }

  const parents = new Map<string, SidebarWorkspaceTree>()

  for (const entry of worktrees.values()) {
    let parent = parents.get(entry.parentKey)

    if (!parent) {
      parent = { id: entry.parentKey, label: entry.parentLabel, path: entry.parentPath, groups: [], sessionCount: 0 }
      parents.set(entry.parentKey, parent)
    }

    parent.groups.push(entry.group)
    parent.sessionCount += entry.group.sessions.length
  }

  const result = [...parents.values()]

  if (noWorkspace.length) {
    result.push({
      id: NO_WORKSPACE_ID,
      label: noWorkspaceLabel,
      path: null,
      groups: [{ id: NO_WORKSPACE_ID, label: noWorkspaceLabel, path: null, sessions: noWorkspace }],
      sessionCount: noWorkspace.length
    })
  }

  // Parents that collide on basename grow a path prefix; worktree labels that
  // collide inside a parent do the same.
  disambiguateLabels(result)

  for (const parent of result) {
    disambiguateLabels(parent.groups)
  }

  return result
}
