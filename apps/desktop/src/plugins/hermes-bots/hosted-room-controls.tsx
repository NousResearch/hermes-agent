import { atom, Button, Codicon, ConfirmDialog, ErrorState, GlyphSpinner, host, RowButton, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import {
  $hostedDirectories, $hostedRooms, approveHostedTask, discardHostedInput, discoverHostedRooms, fetchHostedAttachment, invalidateHostedRoom,
  refreshHostedRoom, resolveHostedTelegramDelivery, retryHostedTask, sendHostedInput, stopHostedRoom
} from './hosted-room-client'
import { hostedHolds, hostedPendingActions } from './hosted-room-protocol'
import type { HostedTelegramDelivery } from './hosted-room-protocol'
import { useBots } from './i18n'
import type { Attachment } from './types'

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

/** Fetch on an explicit gesture, never during transcript replay. Only media is
 * rendered inline; documents and active content are offered as downloads. */
export function HostedRoomAttachment({ attachment, eventId, roomKey }: {
  attachment: Attachment
  eventId: string
  roomKey: string
}) {
  const [loaded, setLoaded] = useState<Attachment | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const loading = useRef(false)
  const name = attachment.name || 'attachment'

  const load = async () => {
    if (loading.current) {return}
    loading.current = true
    setBusy(true)
    setError('')

    try {
      setLoaded(await fetchHostedAttachment(roomKey, eventId, attachment))
    } catch (failure) {
      setError(String(failure))
    } finally {
      loading.current = false
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-1 rounded-md border border-(--ui-stroke-secondary) p-1.5 text-xs">
      {loaded ? (
        <>
          {loaded.kind === 'image' ? <img alt={name} className="max-h-40 max-w-60 object-contain" src={loaded.data} /> : null}
          {loaded.mime?.startsWith('audio/') ? <audio aria-label={name} controls src={loaded.data} /> : null}
          {loaded.mime?.startsWith('video/') ? <video aria-label={name} className="max-h-60 max-w-full" controls src={loaded.data} /> : null}
          <a className="text-(--ui-accent) underline" download={name} href={loaded.data}>Download {name}</a>
        </>
      ) : (
        <Button disabled={busy} onClick={() => void load()} size="xs" variant="ghost">
          {busy ? 'Loading' : 'Open attachment'}: {name}
        </Button>
      )}
      {error ? <p role="alert">{error}</p> : null}
    </div>
  )
}

/** Polling only observes the server. Parking/closing this view never stops or
 * resubmits work, and a transport failure waits for explicit recovery. */
export function HostedRoomStatus({ roomKey, visible }: HostedRoomStatusProps) {
  const b = useBots()
  const cache = useValue($hostedRooms)[roomKey]
  const gateway = useValue(host.state.gateway)
  const [discard, setDiscard] = useState<{ roomKey: string; eventId: string } | null>(null)
  const [telegramId, setTelegramId] = useState('')

  const [resolution, setResolution] = useState<{
    roomKey: string; chatId: number; delivery: HostedTelegramDelivery; decision: 'retry' | 'confirmed-delivered'; messageId?: number
  } | null>(null)

  useEffect(() => {
    if (!visible) {return}
    let disposed = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      // Keep the clock alive, but never retry a failed fetch without explicit recovery.
      if (!$hostedRooms.get()[roomKey]?.error) {await refreshHostedRoom(roomKey)}

      if (!disposed) {timer = setTimeout(() => {
        void poll()
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
  const telegram = cache?.telegramStatus
  const delivery = telegram?.attention
  const messageId = /^\d+$/.test(telegramId.trim()) ? Number(telegramId.trim()) : NaN
  const status = telegram?.blocked ? 'Telegram delivery blocked' : !driver?.running ? 'Worker unavailable' : driver.blocked ? 'Blocked' : driver.working ? 'Working' : 'Idle'
  const actions = hostedPendingActions(driver)
  const reported = Array.isArray(driver?.pending_actions) ? driver.pending_actions.length : 0
  const unsupported = Math.max(0, reported - actions.length)
  // Held members come from the gateway that owns the room; this view never mints or clears a
  // hold, and the release gesture is the ordinary room message the backend already understands.
  const holds = hostedHolds(driver)

  return (
    <div className="grid gap-1 border-b border-(--ui-stroke-tertiary) px-2.5 pb-2 text-xs text-(--ui-text-secondary)">
      <p>Gateway-hosted · {cache?.capabilities?.attachments ? 'Attachments enabled' : 'Text only'} · {b.group.hostedDiscussion}</p>
      <p>{cache?.capabilities?.persistentProcess ? 'Gateway reports an independently hosted process.' : 'Gateway does not advertise persistence. Work only survives Desktop exit on an independently run gateway.'}</p>
      <div className="flex items-center gap-2" role="status">
        {cache?.loading ? <GlyphSpinner spinner="breathe" /> : null}
        <span>{status} · Replayed through {cache?.cursor || 0}</span>
        <Button disabled={cache?.busy} onClick={() => void refreshHostedRoom(roomKey)} size="xs" variant="ghost">Refresh room</Button>
        <Button disabled={cache?.busy || !driver?.running} onClick={() => void stopHostedRoom(roomKey)} size="xs" variant="ghost">Stop room work</Button>
      </div>
      {delivery ? (
        <div className="grid min-w-0 gap-1" role="status">
          <p className="break-words">Telegram chat {telegram.chat_id}: {delivery.profile}, event {delivery.event_id}, part {delivery.chunk_index + 1}, attempt {delivery.attempt}: {delivery.status}.</p>
          {delivery.retry_after ? <p>Telegram cooldown until {new Date(delivery.retry_after * 1000).toLocaleString()}.</p> : null}
          {telegram.blocked ? <p>Check this exact part and sender in Telegram first. The Bot API cannot read chat history; these controls record your external readback, not Telegram proof. No model task is rerun.</p> : null}
          {telegram.blocked && !cache?.capabilities?.telegramRecovery ? <p>This gateway does not advertise Telegram delivery recovery. Update the backend to reconcile this part.</p> : null}
          {telegram.blocked && cache?.capabilities?.telegramRecovery ? (
            <>
              <label className="grid gap-1">
                Telegram message ID (last numeric segment of the copied message link)
                <input className="min-w-0 rounded border border-(--ui-stroke-secondary) p-1" disabled={cache.busy} inputMode="numeric" onChange={event => setTelegramId(event.target.value)} value={telegramId} />
              </label>
              <div className="flex flex-wrap gap-2">
                <Button disabled={cache.busy || !Number.isSafeInteger(messageId) || messageId <= 0} onClick={() => setResolution({ roomKey, chatId: telegram.chat_id, delivery, decision: 'confirmed-delivered', messageId })} size="xs" variant="secondary">Mark delivered</Button>
                <Button disabled={cache.busy || Boolean(delivery.retry_after && delivery.retry_after * 1000 > Date.now())} onClick={() => setResolution({ roomKey, chatId: telegram.chat_id, delivery, decision: 'retry' })} size="xs" variant="ghost">Authorize delivery retry</Button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}
      {holds.length ? (
        <p role="status">
          Paused by you: {holds.map(hold => hold.label).join(', ')}. Send “@{holds[0].handle} resume”
          (or “@all resume”) in the room to release them; Stop pauses everyone.
        </p>
      ) : null}
      {cache?.capabilities && !cache.capabilities.persistentHolds ? (
        <p>This gateway does not report durable pauses. Stop cancels current work here, but a paused bot may run again on the next message.</p>
      ) : null}
      {actions.map(action => (
        <div className="grid gap-1 rounded-md border border-(--ui-stroke-secondary) p-2" key={`${action.kind}:${action.taskId}:${action.requestId || ''}:${action.executionGeneration || ''}`}>
          {action.kind === 'retry' ? (
            <>
              <p>Task {action.taskId} has an uncertain outcome. Retry asks the gateway to reconcile it before executing again.</p>
              <Button disabled={cache?.busy} onClick={() => void retryHostedTask(roomKey, action.taskId)} size="xs" variant="secondary">Retry task</Button>
            </>
          ) : (
            <>
              <p>{cache?.room?.members.find(member => member.member_id === action.memberId)?.display_name || action.memberId} is waiting for approval.</p>
              {action.reason ? <p>{action.reason}</p> : null}
              {action.command?.trim() ? <pre className="max-h-40 overflow-auto whitespace-pre-wrap" data-selectable-text="true">{action.command}</pre> : <p>Command preview unavailable. Only denial is safe here.</p>}
              <div className="flex gap-2">
                {(action.choices || []).map(choice => (
                  <Button disabled={cache?.busy} key={choice} onClick={() => void approveHostedTask(roomKey, action, choice)} size="xs" variant={choice === 'once' ? 'secondary' : 'ghost'}>
                    {choice === 'once' ? 'Allow once' : 'Deny'}
                  </Button>
                ))}
              </div>
            </>
          )}
        </div>
      ))}
      {unsupported ? <p role="status">{unsupported} pending actions cannot be safely identified by this Desktop. Resolve them on the gateway.</p> : null}
      {!cache ? <p role="alert">Hosted room identity is unavailable. Reopen it from the owning gateway.</p> : null}
      {cache?.error ? <div role="alert"><ErrorState description={cache.error} title="Hosted room unavailable" /></div> : null}
      {cache?.pending ? (
        <div className="grid gap-1">
          <p>Pending input, acceptance may be uncertain: {cache.pending.text}</p>
          {cache.pending.attachments?.length ? <p>Saved attachments: {cache.pending.attachments.map(attachment => attachment.name || 'attachment').join(', ')}</p> : null}
          <Button disabled={cache.busy} onClick={() => void sendHostedInput(roomKey)} size="xs" variant="secondary">Retry saved input</Button>
          <Button disabled={cache.busy} onClick={() => setDiscard({ roomKey, eventId: cache.pending!.eventId })} size="xs" variant="ghost">{b.group.discardSavedInput}</Button>
        </div>
      ) : null}
      <ConfirmDialog
        confirmLabel={resolution?.decision === 'retry' ? 'Retry this part' : 'Confirm delivered'}
        description={`Telegram chat ${resolution?.chatId}, ${resolution?.delivery.profile}: ${resolution?.delivery.event_id || ''}, part ${(resolution?.delivery.chunk_index ?? 0) + 1}, attempt ${resolution?.delivery.attempt || ''}. ${resolution?.decision === 'retry'
          ? 'Retry only after external readback in Telegram shows this exact part was not delivered. An ambiguous send may already have arrived: retry can create a duplicate. Only this part is authorized, without rerunning the agent.'
          : `Confirm you checked the exact chat, sender and part in Telegram and message ${resolution?.messageId} is its delivered copy. This records your assertion and skips sending only this part. The Bot API cannot verify chat history.`}`}
        destructive={resolution?.decision === 'retry'}
        onClose={() => setResolution(null)}
        onConfirm={async () => {
          if (resolution) {await resolveHostedTelegramDelivery(resolution.roomKey, resolution.delivery, resolution.decision, resolution.messageId)}
        }}
        open={resolution !== null}
        title="Reconcile Telegram delivery?"
      />
      <ConfirmDialog
        confirmLabel={b.group.discardLocalInput}
        description={b.group.discardSavedWarning}
        destructive
        onClose={() => setDiscard(null)}
        onConfirm={async () => {
          if (discard) {await discardHostedInput(discard.roomKey, discard.eventId)}
        }}
        open={discard !== null}
        title={b.group.discardSavedTitle}
      />
    </div>
  )
}
