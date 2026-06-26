import type { HermesGitWorktree } from '@/global'
import type { ProjectInfo, SessionInfo } from '@/hermes'

// Session grouping is now computed authoritatively on the backend
// (`tui_gateway/project_tree.py`, exposed via `projects.tree` /
// `projects.project_sessions`). The desktop is a thin renderer: this module
// only holds the render contract (the three tree interfaces) plus a couple of
// pure helpers and the VISUAL-ONLY worktree enhancer that injects empty lanes
// from `git worktree list`. It never decides session membership.

export interface SidebarSessionGroup {
  id: string
  label: string
  path: null | string
  sessions: SessionInfo[]
  // Profile color for the ALL-profiles view; absent for workspace groups.
  color?: null | string
  // True when this group is a repo's main checkout (vs a linked worktree).
  isMain?: boolean
  // True for the repo's primary ("home") checkout lane — the single lane that
  // collapses all main-checkout sessions, labeled by the worktree's LIVE branch
  // (defaulting to `main`). Renders a home glyph and pins to the top.
  isHome?: boolean
  // True for the synthetic lane that collapses all of a repo's kanban task
  // worktrees (`<repo>/.worktrees/t_*`) into one row, so a heavy board doesn't
  // spray hundreds of throwaway branch lanes across the sidebar.
  isKanban?: boolean
  loadingMore?: boolean
  mode?: 'profile' | 'source' | 'workspace'
  onLoadMore?: () => void
  sourceId?: string
  totalCount?: number
}

/** A repo node: holds its branch/worktree lanes (`repo -> lane -> sessions`). */
export interface SidebarWorkspaceTree {
  id: string
  label: string
  path: null | string
  groups: SidebarSessionGroup[]
  sessionCount: number
}

/** A project node: human-named (or repo-derived), holds its repo subtree. */
export interface SidebarProjectTree {
  id: string
  label: string
  path: null | string
  color?: null | string
  icon?: null | string
  archived?: boolean
  // A git repo root promoted automatically (not a user-created projects.db row).
  // Deletable = dismissable.
  isAuto?: boolean
  // The synthetic "No project" bucket for cwd-less sessions.
  isNoProject?: boolean
  repos: SidebarWorkspaceTree[]
  sessionCount: number
  // Max activity timestamp across the project's sessions (overview sort key).
  lastActive?: number
  // Up to N most-recent sessions for the overview preview (set by `projects.tree`).
  previewSessions?: SessionInfo[]
}

/** Path split into segments, ignoring trailing slashes and mixed separators. */
const segments = (path: string): string[] =>
  path
    .replace(/[/\\]+$/, '')
    .split(/[/\\]/)
    .filter(Boolean)

/** A path with trailing separators stripped, for stable equality checks. */
const normalizePath = (path: null | string | undefined): string => (path ?? '').replace(/[/\\]+$/, '')

/** Last path segment. */
export const baseName = (path: string): string | undefined => segments(path).pop()

// The `.worktrees` dir for a KANBAN-TASK worktree path, else null. Only matches
// task worktrees (`<repo>/.worktrees/t_<hex>`, the `t_…` id kanban_db mints) so
// the many ephemeral task worktrees collapse into one lane — while user-named
// "New worktree" dirs (`<repo>/.worktrees/<slug>`) stay as their own lanes.
const KANBAN_DIR_RE = /^(.*[/\\]\.worktrees)[/\\]t_[0-9a-f]+[/\\]?$/

export function kanbanWorktreeDir(path: string): null | string {
  return path.match(KANBAN_DIR_RE)?.[1] ?? null
}

/** Label for a main-checkout lane whose session recorded no branch. */
export const DEFAULT_BRANCH_LABEL = 'main'

/** The one definition of a main-checkout lane id (must match the backend tree). */
export const branchLaneId = (repoRoot: string, branch?: string): string =>
  `${repoRoot}::branch::${(branch ?? '').trim()}`

/** A session's recency stamp (last activity, falling back to creation). */
export const sessionRecency = (session: SessionInfo): number => session.last_active || session.started_at || 0

/** Default-branch names that pin to the top and read as the repo's trunk. */
const TRUNK_BRANCHES = new Set(['main', 'master', 'trunk', 'develop'])

const isTrunkLane = (group: SidebarSessionGroup): boolean =>
  Boolean(group.isMain) && TRUNK_BRANCHES.has(group.label.toLowerCase())

/** A lane's recency = its most-recently-active session (empty lanes sink). */
const laneActivity = (group: SidebarSessionGroup): number =>
  group.sessions.reduce((max, session) => Math.max(max, sessionRecency(session)), 0)

// Lane tiers (low sorts first): the repo's primary ("home") checkout pins above
// everything (it's "where you are", labeled by its live branch), then trunk,
// then ordinary branches/worktrees, then the kanban aggregate.
const laneRank = (group: SidebarSessionGroup): number =>
  group.isHome ? 0 : isTrunkLane(group) ? 1 : group.isKanban ? 3 : 2

/**
 * Sort by tier (home → trunk → branches/worktrees → kanban); within a tier, by
 * most-recent activity (empty lanes fall last), label as the tiebreak.
 */
function compareWorktreeGroups(a: SidebarSessionGroup, b: SidebarSessionGroup): number {
  const byRank = laneRank(a) - laneRank(b)

  if (byRank !== 0) {
    return byRank
  }

  const byActivity = laneActivity(b) - laneActivity(a)

  return byActivity || a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
}

export function sortWorktreeGroups(groups: SidebarSessionGroup[]): SidebarSessionGroup[] {
  return [...groups].sort(compareWorktreeGroups)
}

/**
 * VISUAL enhancer only: inject empty lanes from a live `git worktree list` so a
 * repo shows its branches/worktrees even when they have no Hermes sessions yet.
 * The repo's real session lanes already come fully built from the backend
 * (`projects.project_sessions`); this never adds or moves session rows, and it
 * degrades to a no-op on remote backends (where the Electron probe returns
 * nothing). Lanes already present (by id/path) are left untouched.
 */
export function mergeRepoWorktreeGroups(
  repo: Pick<SidebarWorkspaceTree, 'groups' | 'id' | 'path'>,
  discoveredWorktrees?: HermesGitWorktree[]
): SidebarSessionGroup[] {
  // Branch-primary labels: a linked worktree's identity in every git UI (VS
  // Code, JetBrains, lazygit, …) is its CHECKED-OUT BRANCH, not the directory it
  // happens to live in. The backend labels these lanes by dir/slug; relabel them
  // to the live branch from `git worktree list` so the sidebar matches the
  // composer's branch strip. Detached worktrees (no branch) keep their dir label.
  const liveBranchByPath = new Map<string, string>()
  // Inverse: branch → its ONE live worktree path. git guarantees a branch is
  // checked out in at most one worktree, so this mapping is a function and can
  // re-anchor a lane whose stored path has drifted from git truth.
  const livePathByBranch = new Map<string, string>()

  for (const worktree of discoveredWorktrees ?? []) {
    const wtPath = normalizePath(worktree.path)
    const branch = worktree.branch?.trim()

    if (wtPath && branch && !worktree.detached) {
      liveBranchByPath.set(wtPath, branch)
      livePathByBranch.set(branch.toLowerCase(), worktree.path.trim())
    }
  }

  // The primary ("home") checkout's LIVE branch. A repo dir is only ever on ONE
  // branch, so every main-checkout session lane (historical branches over the
  // same root path) collapses into a single home lane labeled by this live
  // branch, defaulting to `main`. Known only when the local git probe ran;
  // remote backends keep the backend's recorded-branch main lane untouched.
  const mainWorktree = (discoveredWorktrees ?? []).find(w => w.isMain)
  const homeBranch = mainWorktree && !mainWorktree.detached ? mainWorktree.branch?.trim() || DEFAULT_BRANCH_LABEL : ''

  // Reconcile a LINKED worktree lane against git truth so its label AND path
  // describe the SAME worktree. Two repair directions:
  //  1. Path git knows → relabel to that path's live branch (git UIs identify a
  //     worktree by its checked-out branch, not the dir it lives in).
  //  2. Path git DOESN'T know but the label IS a live branch → the lane's path
  //     has gone stale; re-anchor it to that branch's real path, else "reveal"
  //     opens a different, stale checkout. The home checkout is folded
  //     separately (below), never here.
  const reconcile = (group: SidebarSessionGroup): SidebarSessionGroup => {
    if (group.isMain || group.isKanban) {
      return group
    }

    const branchForPath = liveBranchByPath.get(normalizePath(group.path))

    if (branchForPath) {
      return branchForPath !== group.label ? { ...group, label: branchForPath } : group
    }

    const livePath = livePathByBranch.get(group.label.trim().toLowerCase())

    if (livePath && normalizePath(livePath) !== normalizePath(group.path)) {
      return { ...group, id: livePath, path: livePath }
    }

    return group
  }

  const dedupeById = (sessions: SessionInfo[]): SessionInfo[] => {
    const byId = new Map<string, SessionInfo>()

    for (const session of sessions) {
      byId.set(session.id, byId.get(session.id) ?? session)
    }

    return [...byId.values()]
  }

  // Fold every main-checkout lane into one home lane labeled by the live branch
  // (the root dir is only ever on one branch); reconcile the linked worktrees.
  // Always shown, even with no sessions on the current branch yet. Remote
  // backends (no probe → no homeBranch) keep their main lanes untouched.
  const mainGroups = repo.groups.filter(group => group.isMain)
  const reconciled = repo.groups.filter(group => !group.isMain).map(reconcile)

  if (homeBranch) {
    reconciled.push({
      id: branchLaneId(repo.id, homeBranch),
      label: homeBranch,
      path: repo.path,
      isMain: true,
      isHome: true,
      sessions: dedupeById(mainGroups.flatMap(group => group.sessions))
    })
  } else {
    reconciled.push(...mainGroups)
  }

  // Collapse any duplicate a re-anchor produced (a stale lane re-pointed onto a
  // path a real lane already holds) — keep the richer (more sessions) lane.
  const byPath = new Map<string, SidebarSessionGroup>()
  const merged: SidebarSessionGroup[] = []

  for (const group of reconciled) {
    const key = !group.isMain && group.path ? normalizePath(group.path) : ''
    const existing = key ? byPath.get(key) : undefined

    if (existing) {
      if (group.sessions.length > existing.sessions.length) {
        merged[merged.indexOf(existing)] = group
        byPath.set(key, group)
      }

      continue
    }

    if (key) {
      byPath.set(key, group)
    }

    merged.push(group)
  }

  const seenIds = new Set(merged.map(group => group.id))
  const seenPaths = new Set(merged.map(group => group.path).filter((path): path is string => Boolean(path)))
  // Dedupe by branch label too: a branch shows once even if it's checked out in
  // a linked worktree AND already has a session lane.
  const seenLabels = new Set(merged.map(group => group.label.toLowerCase()))

  for (const worktree of discoveredWorktrees ?? []) {
    const wtPath = worktree.path?.trim()

    if (!wtPath) {
      continue
    }

    // The home checkout is already the collapsed home lane (above).
    if (worktree.isMain && homeBranch) {
      continue
    }

    // Kanban task worktrees never get their own lane — they fold into the
    // session-derived `::kanban` bucket. Listing every `git worktree list` entry
    // here is exactly what blew the sidebar up to hundreds of empty rows.
    if (!worktree.isMain && kanbanWorktreeDir(wtPath)) {
      continue
    }

    const label =
      (worktree.isMain ? worktree.branch?.trim() || DEFAULT_BRANCH_LABEL : worktree.branch?.trim()) ||
      baseName(wtPath) ||
      wtPath

    const id = worktree.isMain ? branchLaneId(repo.id, label) : wtPath

    const alreadySeen =
      seenIds.has(id) || seenLabels.has(label.toLowerCase()) || (!worktree.isMain && seenPaths.has(wtPath))

    if (alreadySeen) {
      continue
    }

    merged.push({ id, isMain: worktree.isMain, label, path: wtPath, sessions: [] })
    seenIds.add(id)
    seenPaths.add(wtPath)
    seenLabels.add(label.toLowerCase())
  }

  return sortWorktreeGroups(merged)
}

// ── Live session overlay ─────────────────────────────────────────────────────
// The backend tree is a snapshot (sessions with >=1 message, refreshed on a
// turn boundary). For parity with the flat Recents list — instant insertion of
// a freshly-created session and the live "working" arc — we overlay the live
// `$sessions` store onto the tree at render time. This is ADDITIVE only: the
// backend still owns membership, structure, counts, and history. The overlay
// just places rows already present in `$sessions` into the project/lane the
// backend would put them in, using the same id scheme. Worktree/kanban folding
// needs the backend common-root probe, so those rows are left for the next
// tree refresh; the common case (a new main-checkout session) overlays here.

/** True when `target` equals `folder` or is nested under it (segment-wise). */
function isPathUnder(folder: string, target: string): boolean {
  const f = segments(folder)
  const t = segments(target)

  if (!f.length || f.length > t.length) {
    return false
  }

  return f.every((seg, i) => seg === t[i])
}

/**
 * The project a live session belongs to (overview membership) — explicit project
 * by longest-prefix folder, else the repo root (the auto-project id). An IN-TREE
 * linked worktree (`<repoRoot>/.worktrees/<slug>`) belongs to the SAME project as
 * its repo root (the root is right there in the path), so a freshly-created
 * worktree session — e.g. from "convert a branch" / "new worktree" — surfaces in
 * the overview at once instead of waiting for the next backend refresh. Returns
 * null only for sessions we genuinely can't place from the row alone: cwd-less,
 * kanban-task worktrees (they fold into the kanban bucket), or a worktree that
 * lives OUTSIDE the repo root (a sibling dir whose project can't be derived).
 */
export function liveSessionProjectId(session: SessionInfo, explicitProjects: ProjectInfo[]): null | string {
  const cwd = (session.cwd || '').trim()

  if (!cwd || kanbanWorktreeDir(cwd)) {
    return null
  }

  // No persisted repo root yet (brand-new session) → the cwd is the root.
  const repoRoot = (session.git_repo_root || '').trim() || cwd
  const underRepo = cwd === repoRoot || cwd.startsWith(`${repoRoot}/`) || cwd.startsWith(`${repoRoot}\\`)

  if (!underRepo) {
    return null
  }

  let projectId = ''
  let bestLen = -1

  for (const project of explicitProjects) {
    if (project.archived) {
      continue
    }

    for (const folder of project.folders) {
      if (isPathUnder(folder.path, cwd) || isPathUnder(folder.path, repoRoot)) {
        const len = segments(folder.path).length

        if (len > bestLen) {
          bestLen = len
          projectId = project.id
        }
      }
    }
  }

  return projectId || repoRoot
}

const upsertSession = (rows: SessionInfo[], session: SessionInfo): SessionInfo[] =>
  [session, ...rows.filter(row => row.id !== session.id)].sort((a, b) => b.started_at - a.started_at)

/**
 * The lane a live session belongs to WITHIN a known repo root, by path — the
 * entered project already knows its repo roots, so we don't need the session's
 * (often-unset, on a fresh row) git_repo_root. Mirrors the backend's lane ids:
 * main checkout -> branch lane, `.worktrees/t_<hex>` -> kanban, any other
 * `.worktrees/<slug>` -> that worktree's own lane.
 */
function liveLaneForRepo(repoRoot: string, session: SessionInfo): null | SidebarSessionGroup {
  const cwd = (session.cwd || '').trim()

  if (!cwd || !isPathUnder(repoRoot, cwd)) {
    return null
  }

  const wt = cwd.match(/^(.*[/\\]\.worktrees)[/\\]([^/\\]+)/)

  if (wt) {
    const [worktreeRoot, worktreesDir, slug] = [wt[0], wt[1], wt[2]]

    return /^t_[0-9a-f]+$/.test(slug)
      ? { id: `${repoRoot}::kanban`, isKanban: true, isMain: false, label: 'kanban', path: worktreesDir, sessions: [] }
      : { id: worktreeRoot, isMain: false, label: slug, path: worktreeRoot, sessions: [] }
  }

  const branch = (session.git_branch || '').trim() || DEFAULT_BRANCH_LABEL

  return { id: branchLaneId(repoRoot, branch), isMain: true, label: branch, path: repoRoot, sessions: [] }
}

const NO_REMOVED: ReadonlySet<string> = new Set()

/**
 * Reconcile ONE repo's lanes against the live `$sessions` cache: evict
 * deleted/archived rows (`removed`) and inject freshly-created ones, so a lane
 * mutates exactly like the flat Recents list. The backend snapshot stays the
 * datasource for structure and off-page history; this is the optimistic layer
 * on top (Apollo-style), reconciled away on the next snapshot refresh. Returns
 * the same repo ref when nothing changes (memo-stable).
 */
export function overlayRepoLanes(
  repo: SidebarWorkspaceTree,
  live: SessionInfo[],
  removed: ReadonlySet<string> = NO_REMOVED
): SidebarWorkspaceTree {
  const repoRoot = normalizePath(repo.path)
  let changed = false

  // Snapshot lanes minus anything the user just deleted/archived.
  const lanes = repo.groups.map(g => {
    if (!removed.size) {
      return { ...g, sessions: [...g.sessions] }
    }

    const kept = g.sessions.filter(s => !removed.has(s.id))

    changed ||= kept.length !== g.sessions.length

    return { ...g, sessions: kept }
  })

  for (const session of live) {
    const cwd = (session.cwd || '').trim()

    if (removed.has(session.id) || !cwd) {
      continue
    }

    // (1) Join an EXISTING worktree lane by its own path. A linked worktree can
    // live anywhere on disk (often a repo sibling, e.g. `repo-ci`), so nesting
    // under the repo root isn't reliable — but the lane carries its real dir.
    // Longest match wins; skip the root lane so an in-tree `.worktrees/<slug>`
    // session isn't swallowed by main.
    let lane: SidebarSessionGroup | undefined
    let bestLen = -1

    for (const g of lanes) {
      const lanePath = normalizePath(g.path)

      if (!lanePath || lanePath === repoRoot || !isPathUnder(lanePath, cwd)) {
        continue
      }

      const len = segments(lanePath).length

      if (len > bestLen) {
        bestLen = len
        lane = g
      }
    }

    // (2) Else place under the repo root via a computed lane (main / branch /
    // in-tree `.worktrees` / kanban). Match by id, then path (the backend may
    // key a worktree lane off the git-probed root OR a branch-style id), then
    // the main-lane label; create it when the snapshot lacked it.
    if (!lane) {
      const placed = repo.path ? liveLaneForRepo(repo.path, session) : null

      if (!placed) {
        continue
      }

      const placedPath = normalizePath(placed.path)

      lane =
        lanes.find(g => g.id === placed.id) ??
        (placed.isMain
          ? lanes.find(g => g.isMain && g.label.toLowerCase() === placed.label.toLowerCase())
          : undefined) ??
        (!placed.isMain && placedPath ? lanes.find(g => normalizePath(g.path) === placedPath) : undefined)

      if (!lane) {
        lane = { ...placed, sessions: [] }
        lanes.push(lane)
      }
    }

    lane.sessions = upsertSession(lane.sessions, session)
    changed = true
  }

  if (!changed) {
    return repo
  }

  // Drop lanes emptied by eviction (the server only emits non-empty lanes; the
  // git-worktree enhancer re-adds any still-real worktree as an empty lane).
  const groups = sortWorktreeGroups(lanes.filter(g => g.sessions.length > 0))

  return { ...repo, groups, sessionCount: groups.reduce((n, g) => n + g.sessions.length, 0) }
}

/** Project-level overlay: {@link overlayRepoLanes} across every repo subtree. */
export function overlayLiveLanes(
  project: SidebarProjectTree,
  live: SessionInfo[],
  removed: ReadonlySet<string> = NO_REMOVED
): SidebarProjectTree {
  let changed = false

  const repos = project.repos.map(repo => {
    const next = overlayRepoLanes(repo, live, removed)

    changed ||= next !== repo

    return next
  })

  if (!changed) {
    return project
  }

  return { ...project, repos, sessionCount: repos.reduce((n, repo) => n + repo.sessionCount, 0) }
}

/** Merge live sessions into per-project overview previews, keyed by project path. */
export function overlayLivePreviews(
  projects: SidebarProjectTree[],
  live: SessionInfo[],
  explicitProjects: ProjectInfo[],
  limit: number,
  removed: ReadonlySet<string> = new Set()
): Record<string, SessionInfo[]> {
  const byProject = new Map<string, SessionInfo[]>()

  for (const session of live) {
    if (removed.has(session.id)) {
      continue
    }

    const projectId = liveSessionProjectId(session, explicitProjects)

    if (!projectId) {
      continue
    }

    const arr = byProject.get(projectId) ?? []
    arr.push(session)
    byProject.set(projectId, arr)
  }

  const out: Record<string, SessionInfo[]> = {}

  for (const node of projects) {
    if (!node.path) {
      continue
    }

    const liveRows = byProject.get(node.id) ?? []
    const base = (node.previewSessions ?? []).filter(session => !removed.has(session.id))

    if (!liveRows.length && !base.length) {
      continue
    }

    // Live rows take precedence (fresher title/activity/working state).
    const map = new Map<string, SessionInfo>()

    for (const session of [...liveRows, ...base]) {
      if (!map.has(session.id)) {
        map.set(session.id, session)
      }
    }

    out[node.path] = [...map.values()].sort((a, b) => sessionRecency(b) - sessionRecency(a)).slice(0, limit)
  }

  return out
}
