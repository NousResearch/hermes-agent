import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { AgentProfileRoute } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

import {
  branchLaneId,
  NO_PROJECT_ID,
  reconcileEnteredProjectSessions,
  type SessionOwnerKey,
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

const profileOf = (session: SessionInfo): string => (session.profile || '').trim() || 'default'

/** A row's identity across gateways: two profiles or gateways can each hold the same durable id. */
export const sessionIdentityKey = (owner: string, session: SessionInfo): string =>
  JSON.stringify([owner, profileOf(session), session.id])

/** Add another gateway's rows to a window-gateway node. They never repeat the node's own rows
 *  (those belong to the window's gateway, even under the same id), only each other. */
function addToNode(
  node: SidebarProjectTree,
  rows: SessionInfo[],
  ownerOf: (s: SessionInfo) => string
): SidebarProjectTree {
  const seen = new Set<string>()

  const fresh = rows.filter(session => {
    const key = sessionIdentityKey(ownerOf(session), session)

    return !seen.has(key) && Boolean(seen.add(key))
  })

  if (!fresh.length) {
    return node
  }

  const known = node.sessionIds ?? []

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
  // New chats start on the profile last used in this folder on that gateway.
  const profile = profileOf(sessions[0])

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
    profile,
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

  const ownerOf = (session: SessionInfo): string => foreignOwner(session) ?? ''

  let next = tree

  if (home.length) {
    const homeNode = tree.find(node => node.isNoProject)

    next = homeNode
      ? tree.map(node => (node === homeNode ? addToNode(node, home, ownerOf) : node))
      : [
          addToNode(
            { id: NO_PROJECT_ID, isNoProject: true, label: '', path: null, repos: [], sessionCount: 0 },
            home,
            ownerOf
          ),
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

/** The route a new chat in `project` is created on: the owning gateway profile of a project that
 *  exists only on another gateway, else none (the window's own project tree decides). */
export function projectOwnerRoute(project: SidebarProjectTree): AgentProfileRoute | undefined {
  return project.connectionId
    ? { connectionId: project.connectionId, profile: project.profile || 'default' }
    : undefined
}

/** A project's "new session" handlers, carrying its owner route when another gateway owns it. */
export function ownedNewSessionHandlers(
  project: SidebarProjectTree,
  onNewSession?: (path: null | string, owner?: AgentProfileRoute) => void,
  onNewSessionSplit?: NewSessionSplitHandler
): { onNewSession?: (path: null | string) => void; onNewSessionSplit?: NewSessionSplitHandler } {
  const route = projectOwnerRoute(project)

  if (!route) {
    return { onNewSession, onNewSessionSplit }
  }

  return {
    onNewSession: onNewSession && (path => onNewSession(path, route)),
    onNewSessionSplit:
      onNewSessionSplit && ((dir, opts) => onNewSessionSplit(dir, { ...opts, profile: route.profile, route }))
  }
}

/**
 * The live rows an entered project may overlay: a project another gateway owns takes only that
 * gateway's rows, a window-gateway project only the window's, so a chat in the same folder on a
 * different machine never lists in both. Home holds every gateway's folder-less rows.
 */
export function rowsForProjectOwner(
  project: SidebarProjectTree,
  rows: SessionInfo[],
  foreignOwner: (session: SessionInfo) => null | string
): SessionInfo[] {
  if (project.isNoProject) {
    return rows
  }

  const owner = project.connectionId ?? null

  return rows.filter(session => foreignOwner(session) === owner)
}

/** The live rows the entered project overlays: its owner's rows plus its overview previews. */
export function enteredProjectLiveRows(
  project: SidebarProjectTree | undefined,
  live: SessionInfo[],
  owners: { foreignOwner: (session: SessionInfo) => null | string; ownerKey: SessionOwnerKey }
): SessionInfo[] {
  if (!project) {
    return live
  }

  return reconcileEnteredProjectSessions(
    rowsForProjectOwner(project, live, owners.foreignOwner),
    project.previewSessions,
    owners.ownerKey
  )
}
