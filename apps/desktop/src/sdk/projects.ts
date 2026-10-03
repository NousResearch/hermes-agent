import { computed } from 'nanostores'

import { $activeGatewayProfile, $profileScope } from '@/store/profile'
import { $projectTree, projectProfile, projectRootCwd, requestStartWorkSession } from '@/store/projects'

/** A project row a plugin picker can show — the smallest stable subset of the sidebar tree. */
export interface PluginProject {
  id: string
  label: string
  path: string
  sessionCount: number
}

const toPluginProject = (node: {
  id: string
  label: string
  path: null | string
  sessionCount: number
}): PluginProject => ({
  id: node.id,
  label: node.label,
  path: (node.path ?? '').trim(),
  sessionCount: node.sessionCount ?? 0,
})

const isPickable = (node: { isNoProject?: boolean; path: null | string }): boolean =>
  !node.isNoProject && (node.path ?? '').trim() !== ''

/** Project verbs a plugin may use on the user's behalf. Reads and starts use the SAME stores and the SAME draft door as the app's own project controls, so a plugin pick and a hand click can never disagree. Active profile only: every verb throws while viewing all profiles, mirroring the store. */
export const projectsHost = {
  /** List pickable projects for the active profile (the path-less Home bucket is skipped — it is not a folder). */
  list: (): PluginProject[] => {
    if (!projectProfile()) {
      throw new Error('Projects are unavailable while viewing all profiles')
    }

    return $projectTree.get().filter(isPickable).map(toPluginProject)
  },

  /** The same rows as `list()`, as a reactive atom — the tree arrives over an async gateway call after the composer mounts, so a picker subscribes to this (via the SDK's `useValue`) instead of reading once. Empty while viewing all profiles (the stores are per-profile). */
  $list: computed([$projectTree, $profileScope, $activeGatewayProfile], tree =>
    // projectProfile() reads the scope atoms; they are explicit deps so a
    // profile-scope switch re-emits (to [] or back to the rows).
    !projectProfile() ? [] : tree.filter(isPickable).map(toPluginProject)
  ),

  /** Start a fresh chat draft anchored at a project root or an explicit folder. Same door as the sidebar's project “new session” — a stacked tab when main is occupied, never a mutation of an existing conversation. */
  openNewSession: (options: { path?: string; projectId?: string } = {}): void => {
    if (!projectProfile()) {
      throw new Error('Projects are unavailable while viewing all profiles')
    }

    const explicitPath = (options.path ?? '').trim()

    if (explicitPath) {
      requestStartWorkSession(explicitPath, undefined, { openTab: true })

      return
    }

    const projectId = (options.projectId ?? '').trim()

    if (!projectId) {
      throw new Error('Pass a projectId or a folder path')
    }

    const node = $projectTree.get().find(entry => entry.id === projectId)

    if (!node) {
      throw new Error('Unknown project')
    }

    const cwd = projectRootCwd(node)

    if (!cwd) {
      throw new Error('Project has no folder')
    }

    requestStartWorkSession(cwd, undefined, { openTab: true })
  },
}
