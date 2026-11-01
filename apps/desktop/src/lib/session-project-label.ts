import { liveSessionProjectId } from '@/app/chat/sidebar/projects/workspace-groups'
import { pathLeaf } from '@/lib/display-path'
import { $projectOwnerBySessionId, $projectTree } from '@/store/projects'
import type { ProjectInfo, SessionInfo } from '@/types/hermes'

/**
 * The PROJECT a session belongs to, as a label for the sidebar card.
 *
 * Resolution order:
 *   1. `liveSessionProjectId` — the same resolver the session COLOR reads, so
 *      an explicitly-projected row's name and its inherited tint always agree.
 *      Returns an explicit project id, or a repo root for an auto-project.
 *   2. the recorded `git_repo_root` leaf.
 *   3. null (a chat with no workspace at all).
 *
 * Why step 2 exists: `liveSessionProjectId` deliberately refuses to place a
 * worktree that lives OUTSIDE its repo root (`bb/<slug>` beside the checkout
 * rather than in-tree `.worktrees/`), because for LANE MEMBERSHIP a sibling
 * directory genuinely can't be assigned from the row alone. A label has no
 * such ambiguity: the backend already recorded which repo the session belongs
 * to, so we can name it. Without this, most of a sibling-worktree workflow's
 * rows would paint a blank line — and naming the repo is exactly the point of
 * showing the project instead of the cwd (`hermes-agent`, not
 * `hermes-agent-cwd-copy`).
 *
 * This never widens membership — it only names a row that placement leaves
 * unplaced.
 */
export function sessionProjectLabel(session: SessionInfo, projects: ProjectInfo[]): null | string {
  const projectId = liveSessionProjectId(session, projects, $projectOwnerBySessionId.get())

  if (projectId) {
    const explicit = projects.find(project => project.id === projectId)

    if (explicit) {
      // An explicit project shows its user-given name; an auto-project's id IS
      // its repo root path, so it shows that folder's leaf.
      return (explicit.name.trim() ? explicit.name.trim() : pathLeaf(projectId)) || null
    }

    // The owner id can name a project the current `projects.list` answer does
    // not describe: the all-profiles view fans `projects.tree` out over every
    // profile while `projects.list` stays on the active one, so a foreign
    // profile's assignment resolves here with no ProjectInfo row. The tree's
    // own label is the authoritative name — falling through to `pathLeaf`
    // would leak the raw id ("p_8953c3d5") into the UI.
    const fromTree = $projectTree.get().find(project => project.id === projectId)

    if (fromTree) {
      return fromTree.label.trim() || null
    }
  }

  return pathLeaf(session.git_repo_root) || null
}
