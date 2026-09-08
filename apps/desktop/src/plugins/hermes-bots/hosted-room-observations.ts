import type { GroupChat } from './types'

const MAX_KNOWN_IDS = 2000

type Observation = { connectionId: string; token: symbol; capabilityToken?: symbol }
type Inventory = { token: symbol; capabilityToken?: symbol; known: Set<string>; complete: boolean; saturated: boolean }

function roomConnections(room: GroupChat) {
  return new Set(
    [room.hostedConnectionId, ...(room.members || []).map(member => member.route?.connectionId || member.connectionId)]
      .filter(Boolean)
      .map(String)
  )
}

/** Volatile evidence, scoped to retained connections, never a durable ownership ledger. */
export class HostedRoomObservations {
  private readonly inventories = new Map<string, Inventory>()

  capture(connectionId: string): Observation {
    let inventory = this.inventories.get(connectionId)

    if (!inventory) {
      inventory = { token: Symbol(), known: new Set(), complete: false, saturated: false }
      this.inventories.set(connectionId, inventory)
    }

    return { connectionId, token: inventory.token }
  }

  current(observation: Observation) {
    const inventory = this.inventories.get(observation.connectionId)

    return (
      inventory?.token === observation.token &&
      (!observation.capabilityToken || inventory.capabilityToken === observation.capabilityToken)
    )
  }

  /** Order actual probes without invalidating warm inventory or history reads. */
  captureCapability(connectionId: string): Observation {
    const observation = this.capture(connectionId)
    const capabilityToken = Symbol()
    this.inventories.get(connectionId)!.capabilityToken = capabilityToken

    return { ...observation, capabilityToken }
  }

  invalidate(connectionId: string) {
    this.capture(connectionId)
    const inventory = this.inventories.get(connectionId)!
    // A fresh symbol cannot be reused after ABA, pruning, or runtime restart.
    inventory.token = Symbol()
    inventory.complete = false
  }

  invalidateAll() {
    for (const connectionId of this.inventories.keys()) {
      this.invalidate(connectionId)
    }
  }

  retain(connections: Iterable<string>, rooms: GroupChat[]) {
    const retained = new Set([...connections, ...rooms.flatMap(room => [...roomConnections(room)])])

    for (const connectionId of this.inventories.keys()) {
      if (!retained.has(connectionId)) {
        this.inventories.delete(connectionId)
      }
    }
  }

  observe(observation: Observation, ids: ReadonlySet<string>) {
    if (!this.current(observation)) {
      return false
    }

    const inventory = this.inventories.get(observation.connectionId)!

    for (const id of ids) {
      if (inventory.known.has(id)) {
        continue
      }

      if (inventory.known.size === MAX_KNOWN_IDS) {
        // Never evict positive ownership to grant absence. Saturate fail-closed.
        inventory.saturated = true

        break
      }

      inventory.known.add(id)
    }

    return true
  }

  publish(observation: Observation, ids: ReadonlySet<string>, complete: boolean) {
    if (!this.observe(observation, ids)) {
      return false
    }

    this.inventories.get(observation.connectionId)!.complete = complete

    return true
  }

  async read<T>(observation: Observation, request: () => Promise<T>): Promise<T> {
    if (!this.current(observation)) {
      throw new Error('Stale Group Chat observation')
    }

    const result = await request()

    if (!this.current(observation)) {
      throw new Error('Stale Group Chat observation')
    }

    return result
  }

  classicReady(room: GroupChat) {
    if (!room.roomId) {
      return true
    }

    const connections = [...roomConnections(room)]

    if (
      connections.some(id => {
        const inventory = this.inventories.get(id)

        return inventory?.saturated || inventory?.known.has(room.roomId!)
      })
    ) {
      return false
    }

    const members = room.members || []

    if (!members.length || members.some(member => member.remoteSource !== true)) {
      return true
    }

    return connections.every(id => this.inventories.get(id)?.complete)
  }
}
