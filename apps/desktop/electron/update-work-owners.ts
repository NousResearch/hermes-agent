// Which running work does a local in-app update actually interrupt?
//
// The hand-off restarts the local backend and stops every backend the app
// manages over SSH; a remote URL or Hermes Cloud backend keeps running. So
// beside its active-work report each window sends the owner (connection and/or
// profile) of every working chat, and the main process counts only the chats
// whose backend goes down with the update. A report without owners (an older
// renderer, or owners out of step with the count) and an owner the main
// process cannot resolve still count: holding an update is the safe side.

import { type ActiveWork, mergeActiveWork, NO_ACTIVE_WORK } from './quit-guard'

/** The backend a working chat runs on, as the renderer knows it. */
export interface WorkOwner {
  connectionId?: string
  profile?: string
}

export interface OwnedChat {
  title: string
  owner: null | WorkOwner
}

export interface OwnedActiveWork extends ActiveWork {
  /** One entry per working chat. Absent from older renderers. */
  sessions?: OwnedChat[]
}

export type UpdateBackendKind = 'cloud' | 'local' | 'remote' | 'ssh'

// Served from somewhere this app neither restarts nor stops.
const SURVIVES_UPDATE: ReadonlySet<UpdateBackendKind | null> = new Set(['cloud', 'remote'])

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function normalizeOwner(raw: unknown): null | WorkOwner {
  if (!raw || typeof raw !== 'object') {
    return null
  }

  const { connectionId, profile } = raw as { connectionId?: unknown; profile?: unknown }
  const owner: WorkOwner = { connectionId: text(connectionId), profile: text(profile) }

  return owner.connectionId || owner.profile ? owner : null
}

/** Coerce an untrusted renderer's per-chat owner list. */
export function normalizeWorkOwners(payload: unknown): OwnedChat[] {
  return (Array.isArray(payload) ? payload : []).map(entry => ({
    title: text((entry as { title?: unknown } | null)?.title) ?? '',
    owner: normalizeOwner((entry as { owner?: unknown } | null)?.owner)
  }))
}

/**
 * The live window reports narrowed to chats whose backend the update takes
 * down. `kindOf` resolves an owner to its backend kind (null = unknown, which
 * counts). A window that closed has no report here: its chats are covered by
 * the backends' own idle proofs.
 */
export function workInterruptedByUpdate(
  reports: Iterable<OwnedActiveWork>,
  kindOf: (owner: WorkOwner) => null | UpdateBackendKind
): ActiveWork {
  const narrowed: ActiveWork[] = []

  for (const report of reports) {
    const chats = report.sessions

    if (!chats || chats.length !== report.count) {
      narrowed.push(report)

      continue
    }

    const interrupted = chats.filter(chat => !chat.owner || !SURVIVES_UPDATE.has(kindOf(chat.owner)))

    narrowed.push({ count: interrupted.length, titles: interrupted.map(chat => chat.title).filter(Boolean) })
  }

  return mergeActiveWork(narrowed)
}

/** A connection the main process holds; SSH-managed ones are `mode: 'remote'` with `remoteKind: 'ssh'`. */
export interface UpdateBackendConnection {
  baseUrl?: string
  mode?: string
  remoteKind?: string
}

/**
 * The backends whose work the update interrupts, deduplicated by base URL:
 * the local backend (restarted by the update) and every SSH-managed one
 * (stopped when the app quits for the hand-off). URL/cloud remotes outlive it.
 */
export function backendsInterruptedByUpdate<C extends UpdateBackendConnection>(
  connections: ReadonlyArray<C | null>
): C[] {
  const seen = new Set<string>()

  return connections.filter((connection): connection is C => {
    const baseUrl = connection?.baseUrl

    if (!baseUrl || seen.has(baseUrl) || (connection.mode === 'remote' && connection.remoteKind !== 'ssh')) {
      return false
    }

    seen.add(baseUrl)

    return true
  })
}

/** What the main process knows; every read is live (nothing cached between update checks). */
export interface UpdateWorkHost {
  /** Every backend connection attempt the app holds (primary + pool); reading never starts one. */
  connections: () => ReadonlyArray<Promise<unknown> | null | undefined>
  /** Each open window's report with the owners it sent. A closed window has none. */
  reports: () => Iterable<OwnedActiveWork>
  /** One remote serves the whole app and quitting stops no backend the app manages. */
  remoteOnly: () => boolean
  registryConnections: () => ReadonlyArray<{ id: string; kind: UpdateBackendKind }>
  /** The route a profile resolves to; null/undefined is the local backend. Throws when unreadable. */
  profileRouteKind: (profile?: string) => null | undefined | UpdateBackendKind
  /** How long a backend still dialling may take before it counts as having no turn yet. */
  dialBoundMs?: number
}

/** The update gate's two inputs: the interrupted window work and the backends to probe and fence. */
export function updateWorkSources(host: UpdateWorkHost) {
  function ownerKind(owner: WorkOwner): null | UpdateBackendKind {
    try {
      if (owner.connectionId) {
        return host.registryConnections().find(connection => connection.id === owner.connectionId)?.kind ?? null
      }

      return host.profileRouteKind(owner.profile) ?? 'local'
    } catch {
      return null
    }
  }

  return {
    // Live window reports only: the quit-time "last seen" fallback stays busy after the window
    // that reported a turn closes; the backends' idle proofs cover work without a window.
    activeWork: (): ActiveWork =>
      host.remoteOnly() ? NO_ACTIVE_WORK : workInterruptedByUpdate(host.reports(), ownerKind),
    localBackends: async <C extends UpdateBackendConnection>(): Promise<C[]> => {
      const bound = new Promise<null>(resolve => setTimeout(resolve, host.dialBoundMs ?? 5000, null))
      const pending = host.connections().filter((promise): promise is Promise<unknown> => Boolean(promise))
      const settled = await Promise.all(pending.map(promise => Promise.race([promise.catch(() => null), bound])))

      return backendsInterruptedByUpdate(settled as Array<C | null>)
    }
  }
}
