import { map } from 'nanostores'

import type { HermesGateway } from '@/hermes'
import { gatewayRouteState, requestGatewayForAgent } from '@/store/gateway'
import type { AgentProfileRoute } from '@/store/profile'
import type { ProjectInfo, ProjectsPayload } from '@/types/hermes'

// A project another gateway owns is drawn from that gateway's session rows; its
// look (color / icon) lives in THAT gateway profile's projects.db, the same
// record its own window themes. Reads and writes go to the owner, so theming
// "app · Studio" never mints a project on this window's gateway. The sidebar's
// reads are passive (only a socket already open to that owner); only a colour
// pick, a user gesture, may dial it.

export interface GatewayProjectLook {
  color: null | string
  icon: null | string
  /** The owner's project record for the folder, once it has one. */
  id: null | string
}

/** route key → folder → look, for the folders each owner has a project record for. */
export const $gatewayProjectLooks = map<Record<string, Record<string, GatewayProjectLook>>>({})

export const gatewayRouteKey = (route: AgentProfileRoute): string => JSON.stringify([route.connectionId, route.profile])

export const folderKey = (path: string): string => path.replace(/[\\/]+$/, '') || path

const basename = (path: string): string => folderKey(path).split(/[\\/]/).pop() || path

/** Each folder's live project: a primary path wins over a folder another project merely includes. */
function looksByFolder(projects: ProjectInfo[]): Record<string, GatewayProjectLook> {
  const looks: Record<string, GatewayProjectLook> = {}
  const live = projects.filter(project => !project.archived)
  const paths = (project: ProjectInfo) => [project.primary_path, ...(project.folders ?? []).map(folder => folder.path)]

  for (const project of [...live].sort((a, b) => Number(!a.primary_path) - Number(!b.primary_path))) {
    for (const path of paths(project)) {
      const key = path && folderKey(path)

      if (key && !looks[key]) {
        looks[key] = { color: project.color ?? null, icon: project.icon ?? null, id: project.id }
      }
    }
  }

  return looks
}

async function readOwnerLooks(route: AgentProfileRoute): Promise<Record<string, GatewayProjectLook>> {
  const payload = await requestGatewayForAgent<ProjectsPayload>(route.connectionId, route.profile, 'projects.list', {
    profile: route.profile
  })

  return looksByFolder(payload.projects ?? [])
}

// Every write stamps its folder when it starts and when it settles. A read issued before the
// latest stamp may predate that write's commit, so it never replaces that folder's look.
let writeSeq = 0
const written = new Map<string, Map<string, number>>()

function stampWrite(route: AgentProfileRoute, path: string): void {
  const key = gatewayRouteKey(route)
  const folders = written.get(key) ?? new Map<string, number>()

  folders.set(folderKey(path), ++writeSeq)
  written.set(key, folders)
}

/** An owner read replaces that owner's looks, except folders written since the read was sent. */
function publishOwnerLooks(route: AgentProfileRoute, looks: Record<string, GatewayProjectLook>, sentAt: number): void {
  const key = gatewayRouteKey(route)
  const current = $gatewayProjectLooks.get()[key] ?? {}
  const next = { ...looks }

  for (const [folder, seq] of written.get(key) ?? []) {
    if (seq <= sentAt) {
      continue
    }

    if (current[folder]) {
      next[folder] = current[folder]
    } else {
      delete next[folder]
    }
  }

  $gatewayProjectLooks.setKey(key, next)
}

interface OwnerRead {
  /** Failed reads on the watched socket; at the cap the owner waits for that socket to reconnect. */
  failures: number
  inFlight: boolean
  /** The socket reconnected while a read was in flight: that read may predate it, so read again. */
  rereadQueued: boolean
  /** The socket the published looks came over; until it matches the route's socket, a render reads. */
  readOver: HermesGateway | null
  stopWatching: () => void
  watching: HermesGateway | null
}

const MAX_FAILED_READS = 3
const reads = new Map<string, OwnerRead>()

async function readOwner(route: AgentProfileRoute, read: OwnerRead, gateway: HermesGateway): Promise<void> {
  const sentAt = writeSeq

  read.inFlight = true

  try {
    const payload = await gateway.request<ProjectsPayload>('projects.list', { profile: route.profile })

    publishOwnerLooks(route, looksByFolder(payload.projects ?? []), sentAt)
    read.failures = 0
    read.readOver = gateway
  } catch {
    read.failures += 1
  } finally {
    read.inFlight = false
  }

  if (read.rereadQueued) {
    read.rereadQueued = false
    read.failures = 0
    read.readOver = null

    if (read.watching === gateway && gateway.connectionState === 'open') {
      await readOwner(route, read, gateway)
    }
  }
}

/** A socket that drops and reopens re-reads: the owner's window may have recoloured meanwhile. */
function watchSocket(route: AgentProfileRoute, read: OwnerRead, gateway: HermesGateway): void {
  let dropped = false

  read.stopWatching()
  read.watching = gateway
  read.failures = 0
  read.stopWatching = gateway.onState(state => {
    if (state !== 'open') {
      dropped = true
    } else if (dropped) {
      dropped = false
      read.failures = 0
      read.readOver = null

      if (read.inFlight) {
        read.rereadQueued = true
      } else {
        void readOwner(route, read, gateway)
      }
    }
  })
}

/** Read each owner's looks on a socket already open to it: a sidebar render never dials or spawns
 *  a backend (an idle-evicted local profile, a new pool entry per profile on a shared host). An
 *  owner with no open socket, or whose read failed, is read again on a later render (failures
 *  capped per socket); a reconnect of that socket re-reads on its own. */
export function loadGatewayProjectLooks(routes: AgentProfileRoute[]): void {
  for (const route of routes) {
    const key = gatewayRouteKey(route)
    const { gateway } = gatewayRouteState(route.connectionId, route.profile)

    if (!gateway) {
      continue
    }

    const read = reads.get(key) ?? {
      failures: 0,
      inFlight: false,
      readOver: null,
      rereadQueued: false,
      stopWatching: () => {},
      watching: null
    }

    reads.set(key, read)

    if (read.watching !== gateway) {
      watchSocket(route, read, gateway)
    }

    if (!read.inFlight && read.readOver !== gateway && read.failures < MAX_FAILED_READS) {
      void readOwner(route, read, gateway)
    }
  }
}

export function gatewayProjectLook(route: AgentProfileRoute, path: string): GatewayProjectLook | undefined {
  return $gatewayProjectLooks.get()[gatewayRouteKey(route)]?.[folderKey(path)]
}

function setLook(route: AgentProfileRoute, path: string, look: GatewayProjectLook | undefined): void {
  const key = gatewayRouteKey(route)
  const next = { ...$gatewayProjectLooks.get()[key] }

  if (look) {
    next[folderKey(path)] = look
  } else {
    delete next[folderKey(path)]
  }

  $gatewayProjectLooks.setKey(key, next)
}

// One write per owner folder at a time: the first color on a folder creates the
// owner's record, and a second pick racing it must patch that record, not
// create a second one.
const writes = new Map<string, Promise<unknown>>()

/**
 * Theme a folder on the gateway that owns it: patch the owner's project for that
 * folder, or create one there (named for the folder, not this window's
 * gateway-suffixed label) carrying the look. Optimistic, rolled back on failure.
 */
export function setGatewayProjectAppearance(
  route: AgentProfileRoute,
  path: string,
  patch: { color?: null | string; icon?: null | string }
): Promise<void> {
  const key = JSON.stringify([gatewayRouteKey(route), folderKey(path)])
  const run = (writes.get(key) ?? Promise.resolve()).catch(() => undefined).then(() => writeLook(route, path, patch))

  writes.set(key, run)

  return run
}

async function writeLook(
  route: AgentProfileRoute,
  path: string,
  patch: { color?: null | string; icon?: null | string }
): Promise<void> {
  const before = gatewayProjectLook(route, path)

  const patched = (current: GatewayProjectLook | undefined): GatewayProjectLook => ({
    color: patch.color !== undefined ? patch.color : (current?.color ?? null),
    icon: patch.icon !== undefined ? patch.icon : (current?.icon ?? null),
    id: current?.id ?? null
  })

  const request = (method: string, params: Record<string, unknown>) =>
    requestGatewayForAgent<{ project: null | ProjectInfo }>(route.connectionId, route.profile, method, {
      ...params,
      profile: route.profile
    })

  stampWrite(route, path)
  setLook(route, path, patched(before))

  try {
    // The owner may hold a record for the folder this window never read (its own window themed it).
    // Only this folder is taken from the read: a read racing another folder's write is stale there.
    const current = (await readOwnerLooks(route))[folderKey(path)]
    const look = patched(current)

    setLook(route, path, look)

    const { project } = current?.id
      ? // Backend treats null as "leave unchanged"; "" clears.
        await request('projects.update', {
          id: current.id,
          ...(patch.color !== undefined && { color: patch.color ?? '' }),
          ...(patch.icon !== undefined && { icon: patch.icon ?? '' })
        })
      : await request('projects.create', {
          name: basename(path),
          folders: [path],
          primary_path: path,
          color: look.color || undefined,
          icon: look.icon || undefined
        })

    if (project) {
      setLook(route, path, { color: project.color ?? null, icon: project.icon ?? null, id: project.id })
    }
  } catch (err) {
    setLook(route, path, before)

    throw err
  } finally {
    stampWrite(route, path)
  }
}
