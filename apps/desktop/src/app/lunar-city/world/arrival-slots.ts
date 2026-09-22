import type { DestinationId, EntityKey, LunarCitySnapshot, Vec3, WorldManifestV2 } from '../model'

/** Stable presentation slots along the authored approach, within its walkable width. */
export function createArrivalSlots(manifest: { navigation: Pick<WorldManifestV2['navigation'], 'links'> }) {
  const slots = new Map<DestinationId, Map<EntityKey, number>>()

  return {
    update(snapshot: LunarCitySnapshot) {
      for (const [destination, assignments] of slots) {for (const key of assignments.keys()) {
        if (snapshot.entities.get(key)?.destination !== destination) {assignments.delete(key)}
      }}
    },
    target(key: EntityKey, destination: DestinationId, origin: Vec3): Vec3 {
      const approach = manifest.navigation.links.find(link =>
        Math.hypot(link.to.x-origin.x,link.to.z-origin.z)<.05 || Math.hypot(link.from.x-origin.x,link.from.z-origin.z)<.05)

      if (!approach) {return origin}
      const other = Math.hypot(approach.to.x-origin.x,approach.to.z-origin.z)<.05 ? approach.from : approach.to
      const length = Math.hypot(other.x-origin.x,other.z-origin.z)

      if (length < 2) {return origin}
      const assignments=slots.get(destination) ?? new Map<EntityKey,number>()
      slots.set(destination,assignments)
      const capacity = Math.min(24, 2 * (1 + Math.floor((length - 1) / 1.8)))

      if (!assignments.has(key)) {
        const occupied=new Set(assignments.values());let slot=0

        while(occupied.has(slot) && slot<capacity){slot++}

        if(slot>=capacity){return origin}
        assignments.set(key,slot)
      }

      const slot=assignments.get(key)!

      if (slot >= capacity) {return origin}
      const distance=.5+Math.floor(slot/2)*1.8
      const dx=(other.x-origin.x)/length,dz=(other.z-origin.z)/length,side=slot%2===0?-.85:.85

      return {x:origin.x+dx*distance-dz*side,y:origin.y+(other.y-origin.y)*distance/length,z:origin.z+dz*distance+dx*side}
    }
  }
}
