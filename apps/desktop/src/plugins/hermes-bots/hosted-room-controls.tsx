import { atom, Button, Codicon, ErrorState, GlyphSpinner, host, RowButton, useValue } from '@hermes/plugin-sdk'
import { useEffect } from 'react'

import {
  $hostedDirectories, $hostedRooms, discoverHostedRooms, invalidateHostedRoom,
  refreshHostedRoom, sendHostedInput, stopHostedRoom
} from './hosted-room-client'
import { hostedHolds } from './hosted-room-protocol'

const unknownConnection = atom('')

interface HostedRoomDirectoryProps {
  onOpen: (key: string) => void
}

/** A discovery section inside the existing Bots roster, including its empty
 * state. Selecting a row uses the same native Group Chat workspace door. */
export function HostedRoomDirectory({ onOpen }: HostedRoomDirectoryProps) {
  const connection = useValue(host.state.connectionId || unknownConnection) || host.activeConnectionId?.() || ''
  const gateway = useValue(host.state.gateway)
  const directories = useValue($hostedDirectories)
  const rooms = useValue($hostedRooms)
  const directory = directories[connection]
  useEffect(() => {
    if (connection && gateway === 'open') {void discoverHostedRooms(connection)}
  }, [connection, gateway])

  return (
    <section aria-label="Hosted group chats" className="grid gap-1 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-(--ui-text-secondary)">Hosted group chats</span>
        <Button disabled={!connection || directory?.loading} onClick={() => void discoverHostedRooms(connection)} size="xs" variant="ghost">
          Refresh
        </Button>
      </div>
      {directory?.loading ? <GlyphSpinner spinner="breathe" /> : null}
      {directory?.error ? <div className="text-xs text-destructive" role="alert">{directory.error}</div> : null}
      {!directory?.loading && !directory?.error && !directory?.keys.length ? (
        <p className="text-xs text-(--ui-text-tertiary)">No hosted rooms on this gateway. Create one on the gateway, then refresh.</p>
      ) : null}
      {directory?.keys.map(key => rooms[key] ? (
        <RowButton className="flex min-w-0 items-center gap-2 text-left text-xs" key={key} onClick={() => onOpen(key)}>
          <Codicon name="organization" />
          <span className="truncate">{rooms[key].name}</span>
          <span className="ml-auto text-(--ui-text-tertiary)">Hosted</span>
        </RowButton>
      ) : null)}
    </section>
  )
}

interface HostedRoomStatusProps {
  roomKey: string
  visible: boolean
}

/** Polling only observes the server. Parking/closing this view never stops or
 * resubmits work, and a transport failure waits for explicit recovery. */
export function HostedRoomStatus({ roomKey, visible }: HostedRoomStatusProps) {
  const cache = useValue($hostedRooms)[roomKey]
  const gateway = useValue(host.state.gateway)
  useEffect(() => {
    if (!visible) {return}
    let disposed = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      await refreshHostedRoom(roomKey)

      if (!disposed) {timer = setTimeout(() => {
        if (!$hostedRooms.get()[roomKey]?.error) {void poll()}
      }, 2000)}
    }

    void poll()

    return () => {
      disposed = true
      clearTimeout(timer)
      invalidateHostedRoom(roomKey)
    }
  }, [roomKey, visible, gateway])

  const driver = cache?.driverStatus
  const status = !driver?.running ? 'Worker unavailable' : driver.blocked ? 'Blocked' : driver.working ? 'Working' : 'Idle'
  const pendingActions = Array.isArray(driver?.pending_actions) ? driver.pending_actions : []
  // Held members come from the gateway that owns the room; this view never mints or clears a
  // hold, and the release gesture is the ordinary room message the backend already understands.
  const holds = hostedHolds(driver)

  return (
    <div className="grid gap-1 border-b border-(--ui-stroke-tertiary) px-2.5 pb-2 text-xs text-(--ui-text-secondary)">
      <p>Gateway-hosted · Text only · Backend Discussion policy: manual holds match Desktop, round policy does not.</p>
      <p>{cache?.capabilities?.persistentProcess ? 'Gateway reports an independently hosted process.' : 'Gateway does not advertise persistence. Work only survives Desktop exit on an independently run gateway.'}</p>
      <div className="flex items-center gap-2" role="status">
        {cache?.loading ? <GlyphSpinner spinner="breathe" /> : null}
        <span>{status} · Replayed through {cache?.cursor || 0}</span>
        <Button disabled={cache?.busy} onClick={() => void refreshHostedRoom(roomKey)} size="xs" variant="ghost">Refresh room</Button>
        <Button disabled={cache?.busy || !driver?.running} onClick={() => void stopHostedRoom(roomKey)} size="xs" variant="ghost">Stop room work</Button>
      </div>
      {holds.length ? (
        <p role="status">
          Paused by you: {holds.map(hold => hold.label).join(', ')}. Send “@{holds[0].handle} resume”
          (or “@all resume”) in the room to release them; Stop pauses everyone.
        </p>
      ) : null}
      {cache?.capabilities && !cache.capabilities.persistentHolds ? (
        <p>This gateway does not report durable pauses. Stop cancels current work here, but a paused bot may run again on the next message.</p>
      ) : null}
      {pendingActions.length ? <p role="status">Backend needs attention ({pendingActions.length} pending actions). Approval and task retry controls are not included in this text-only slice; use the gateway.</p> : null}
      {!cache ? <p role="alert">Hosted room identity is unavailable. Reopen it from the owning gateway.</p> : null}
      {cache?.error ? <div role="alert"><ErrorState description={cache.error} title="Hosted room unavailable" /></div> : null}
      {cache?.pending ? (
        <div className="grid gap-1">
          <p>Pending input, acceptance may be uncertain: {cache.pending.text}</p>
          <Button disabled={cache.busy} onClick={() => void sendHostedInput(roomKey)} size="xs" variant="secondary">Retry saved input</Button>
        </div>
      ) : null}
    </div>
  )
}
