import { atom, Button, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'
import type { ShippedGroupAdoption } from './types'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})

export function bindAdoptedCanonicalGroup(
  group: string,
  route: CanonicalGroupRoute,
  room: Pick<CanonicalRoom, 'room_id'>,
  adoption: ShippedGroupAdoption,
  lifecycleGeneration: number,
  adoptionCurrent: () => boolean,
  routeOwner: NonNullable<CanonicalGroupBinding['routeOwner']>
): string {
  const key = String(group || '').trim()

  if (!key || !route.connectionId?.trim() || !route.profile?.trim() || !room.room_id?.trim()) {
    throw new Error('Adopted canonical group requires its retained group key and exact owner route')
  }

  if (
    adoption.state !== 'adopted' ||
    adoption.roomId !== room.room_id ||
    adoption.route?.connectionId !== route.connectionId ||
    adoption.route?.profile !== route.profile
  ) {
    throw new Error('Adopted canonical group binding does not match its durable checkpoint')
  }

  const binding: CanonicalGroupBinding = {
    ...route,
    roomId: room.room_id,
    routeOwner,
    adoptionOwner: {
      authorityGatewayId: adoption.route.authorityGatewayId,
      lifecycleGeneration,
      requestHash: adoption.requestHash,
      sourceId: adoption.sourceId
    }
  }

  binding.isCurrent = () => {
    if ($canonicalGroupBindings.get()[key] !== binding || !adoptionCurrent()) {
      return false
    }

    try {
      routeOwner.assertCurrent()

      return true
    } catch {
      return false
    }
  }

  const previous = $canonicalGroupBindings.get()[key]?.routeOwner

  if (previous && previous !== routeOwner) {
    previous.release()
  }
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: binding })

  return key
}

export function revokeCanonicalGroupBinding(group: string): void {
  const key = String(group || '').trim()
  const current = $canonicalGroupBindings.get()

  if (!key || !current[key]) {
    return
  }

  const next = { ...current }
  current[key].routeOwner?.release()
  delete next[key]
  $canonicalGroupBindings.set(next)
}

export function revokeAdoptedCanonicalGroupsForConnection(connectionId?: string): void {
  const id = String(connectionId || '').trim()
  const current = $canonicalGroupBindings.get()

  const retained = Object.entries(current).filter(
    ([, binding]) => !binding.adoptionOwner || (id && binding.connectionId !== id)
  )

  const keep = new Set(retained.map(([key]) => key))

  for (const [key, binding] of Object.entries(current)) {
    if (!keep.has(key)) {
      binding.routeOwner?.release()
    }
  }

  const next = Object.fromEntries(retained)

  if (Object.keys(next).length !== Object.keys(current).length) {
    $canonicalGroupBindings.set(next)
  }
}

export function revokeStaleAdoptedCanonicalGroups(): void {
  const current = $canonicalGroupBindings.get()
  const retained: Array<[string, CanonicalGroupBinding]> = []

  for (const [key, binding] of Object.entries(current)) {
    if (binding.adoptionOwner && binding.isCurrent?.() === false) {
      binding.routeOwner?.release()
    } else {
      retained.push([key, binding])
    }
  }

  if (retained.length !== Object.keys(current).length) {
    $canonicalGroupBindings.set(Object.fromEntries(retained))
  }
}

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${room.room_id}`
  $canonicalGroupBindings.get()[key]?.routeOwner?.release()
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: { ...route, roomId: room.room_id } })

  return key
}

export function CanonicalGroupList({ onOpen }: { onOpen: (key: string) => void }) {
  const labels = useCanonicalGroupLabels()
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const [rooms, setRooms] = useState<Array<{ key: string; name: string }>>([])
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let cancelled = false
    setRooms([])
    setError('')
    void (async () => {
      const route = captureCanonicalGroupRoute()
      const result = await discoverCanonicalGroups(route)

      if (!cancelled) {
        setRooms(result.rooms.map(room => ({ key: registerCanonicalGroup(route, room), name: room.name })))
      }
    })().catch(e => {
      if (!cancelled) {
        setError(e instanceof Error ? e.message : String(e))
      }
    })

    return () => {
      cancelled = true
    }
  }, [connectionId, profile, refresh])

  return (
    <div className="grid gap-1 px-2">
      <Button onClick={() => setRefresh(value => value + 1)} variant="ghost">
        {labels.refreshGroups}
      </Button>
      {error && <p role="alert">{error}</p>}
      {rooms.map(room => (
        <Button key={room.key} onClick={() => onOpen(room.key)} variant="ghost">
          {room.name}
        </Button>
      ))}
    </div>
  )
}
