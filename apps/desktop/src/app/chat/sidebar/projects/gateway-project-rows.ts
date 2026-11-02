import type { SessionInfo } from '@/types/hermes'

import {
  branchLaneId,
  NO_PROJECT_ID,
  sessionRecency,
  type SidebarProjectTree,
  type SidebarWorkspaceTree
} from './workspace-groups'

/** The folder a row from another gateway files under: its repo root, else its cwd, else none (Home). */
const rowAnchor = (session: SessionInfo): string => (session.git_repo_root || '').trim() || (session.cwd || '').trim()

const basename = (path: string): string =>
  path
    .replace(/[\\/]+$/, '')
    .split(/[\\/]/)
    .pop() || path

/** Id of a project that exists only on another gateway: [gateway, folder], never a local path,
 *  so a same-named folder on this machine stays a separate project. */
export const gatewayProjectId = (connectionId: string, path: string): string =>
  JSON.stringify(['gateway-project', connectionId, path])

function addToNode(node: SidebarProjectTree, rows: SessionInfo[]): SidebarProjectTree {
  const known = new Set(node.sessionIds ?? [])
  const fresh = rows.filter(session => !known.has(session.id))

  if (!fresh.length) {
    return node
  }

  return {
    ...node,
    lastActive: Math.max(node.lastActive ?? 0, ...fresh.map(sessionRecency)),
    previewSessions: [...(node.previewSessions ?? []), ...fresh],
    sessionCount: node.sessionCount + fresh.length,
    sessionIds: [...known, ...fresh.map(session => session.id)]
  }
}

function gatewayProjectNode(connectionId: string, path: string, rows: SessionInfo[]): SidebarProjectTree {
  const id = gatewayProjectId(connectionId, path)
  const label = basename(path)
  const sessions = [...rows].sort((a, b) => sessionRecency(b) - sessionRecency(a))

  const repo: SidebarWorkspaceTree = {
    id,
    label,
    path,
    groups: [{ id: branchLaneId(path), label, path, isHome: true, isMain: true, sessions }],
    sessionCount: sessions.length
  }

  return {
    id,
    label,
    path,
    isAuto: true,
    connectionId,
    repos: [repo],
    sessionCount: sessions.length,
    lastActive: Math.max(0, ...sessions.map(sessionRecency)),
    previewSessions: sessions,
    sessionIds: sessions.map(session => session.id)
  }
}

/**
 * The project tree is served by ONE backend (the window's gateway), so it only knows that gateway's
 * rows. Rows another gateway owns (All profiles across gateways) would otherwise match no project and
 * vanish from Project grouping. File them the way that backend files its own: a folder-less row goes
 * to Home, a row with a folder to a project for that folder on its gateway. A project name two
 * sources share gets the gateway in its label.
 */
export function withGatewayProjectRows(
  tree: SidebarProjectTree[],
  live: SessionInfo[],
  foreignOwner: (session: SessionInfo) => null | string,
  gatewayLabel: (connectionId: string) => string
): SidebarProjectTree[] {
  const home: SessionInfo[] = []
  const byFolder = new Map<string, { connectionId: string; path: string; rows: SessionInfo[] }>()

  for (const session of live) {
    const connectionId = foreignOwner(session)

    if (!connectionId) {
      continue
    }

    const path = rowAnchor(session)

    if (!path) {
      home.push(session)

      continue
    }

    const id = gatewayProjectId(connectionId, path)
    const bucket = byFolder.get(id) ?? { connectionId, path, rows: [] }

    bucket.rows.push(session)
    byFolder.set(id, bucket)
  }

  if (!home.length && !byFolder.size) {
    return tree
  }

  let next = tree

  if (home.length) {
    const homeNode = tree.find(node => node.isNoProject)

    next = homeNode
      ? tree.map(node => (node === homeNode ? addToNode(node, home) : node))
      : [
          addToNode({ id: NO_PROJECT_ID, isNoProject: true, label: '', path: null, repos: [], sessionCount: 0 }, home),
          ...tree
        ]
  }

  const added = [...byFolder.values()].map(({ connectionId, path, rows }) =>
    gatewayProjectNode(connectionId, path, rows)
  )

  const labelCount = new Map<string, number>()

  for (const node of [...next, ...added]) {
    if (!node.isNoProject) {
      labelCount.set(node.label, (labelCount.get(node.label) ?? 0) + 1)
    }
  }

  return [
    ...next,
    ...added.map(node =>
      (labelCount.get(node.label) ?? 0) > 1
        ? { ...node, label: `${node.label} · ${gatewayLabel(node.connectionId!)}` }
        : node
    )
  ]
}
