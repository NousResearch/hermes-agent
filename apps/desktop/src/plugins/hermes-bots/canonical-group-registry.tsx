import { atom, Button, Codicon, gatewayActivationEpoch, host, RowButton, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { groupCreationSource } from './canonical-group-capabilities'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})
/** Display names stay outside the bindings, so a rename never touches a room's routing identity. */
export const $canonicalGroupNames = atom<Record<string, string>>({})

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${room.room_id}`
  const bindings = $canonicalGroupBindings.get()

  if (!bindings[key]) {
    $canonicalGroupBindings.set({ ...bindings, [key]: { connectionId: route.connectionId, profile: route.profile, roomId: room.room_id } })
  }

  const names = $canonicalGroupNames.get()

  if (names[key] !== room.name) {
    $canonicalGroupNames.set({ ...names, [key]: room.name })
  }

  return key
}

/** A disbanded room leaves the registry; its key never resolves to a stale binding again. */
export function forgetCanonicalGroup(binding: CanonicalGroupBinding) {
  const remaining = Object.fromEntries(Object.entries($canonicalGroupBindings.get()).filter(([, bound]) =>
    bound.connectionId !== binding.connectionId || bound.profile !== binding.profile || bound.roomId !== binding.roomId))

  $canonicalGroupBindings.set(remaining)
  $canonicalGroupNames.set(Object.fromEntries(Object.entries($canonicalGroupNames.get()).filter(([key]) => key in remaining)))
}

export function CanonicalGroupList({ onOpen }: { onOpen: (key: string) => void }) {
  const labels = useCanonicalGroupLabels()
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const gateway = useValue(host.state.gateway)
  const bindings = useValue($canonicalGroupBindings)
  const [rooms, setRooms] = useState<Array<{ key: string; name: string }>>([])
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const socketGenerationRef = useRef(0)
  const [socketGeneration, setSocketGeneration] = useState(0)
  useEffect(() => host.state.gateway.listen(() => {
    // React can batch closed -> open into the same rendered value. Fence
    // outstanding reads immediately and retain every readiness transition.
    socketGenerationRef.current++
    setSocketGeneration(socketGenerationRef.current)
  }), [])
  useEffect(() => {
    let cancelled = false
    setRooms([])
    setError('')

    if (gateway !== 'open') {return}

    // Capture at the read boundary: the React Compiler can memoize a plain
    // getter in render for the lifetime of this mounted list.
    const activationEpoch = gatewayActivationEpoch()
    const sourceIsCurrent = groupCreationSource({ connectionId: connectionId ?? '', profile }, activationEpoch)

    const publish = (update: () => void) => {
      if (cancelled || socketGenerationRef.current !== socketGeneration) {return}

      if (sourceIsCurrent()) {
        update()

        return
      }

      // Startup can re-activate the same open primary without changing any
      // subscribed atom. Discard its old response and read under the new epoch.
      if (gatewayActivationEpoch() !== activationEpoch &&
          host.state.connectionId.get() === connectionId && host.state.profile.get() === profile &&
          host.state.gateway.get() === 'open') {setRefresh(value => value + 1)}
    }

    void (async () => {
      const route = captureCanonicalGroupRoute()
      // Socket readiness does not advance the route epoch. Refresh the
      // capability after startup/reconnect instead of reusing a closed read.
      const result = await discoverCanonicalGroups(route, activationEpoch, true)

      publish(() => setRooms(result.rooms.map(room => ({ key: registerCanonicalGroup(route, room), name: room.name }))))
    })().catch(e => {publish(() => setError(e instanceof Error ? e.message : String(e)))})

    return () => { cancelled = true }
  }, [connectionId, profile, gateway, socketGeneration, refresh])

  return <div className="grid gap-1 px-2">
    <div className="flex items-center justify-between gap-2 px-2">
      <span className="text-xs font-medium text-(--ui-text-secondary)">{labels.groupChats}</span>
      <Button aria-label={labels.refreshGroups} onClick={() => setRefresh(value => value + 1)} size="icon" title={labels.refreshGroups} variant="ghost"><Codicon name="refresh" /></Button>
    </div>
    {error && <p className="px-2 text-xs text-(--ui-text-secondary)" role="alert">{labels.driverUnavailable}</p>}
    {rooms.filter(room => bindings[room.key]).map(room => <RowButton className="flex min-w-0 items-center gap-2 rounded-md px-2 py-2 text-left text-sm text-(--ui-text-primary) hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring" key={room.key} onClick={() => onOpen(room.key)}>
      <Codicon className="shrink-0 text-(--ui-text-secondary)" name="comment-discussion" /><span className="truncate">{room.name}</span>
    </RowButton>)}
  </div>
}
