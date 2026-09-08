import { atom, Button, Codicon, ConfirmDialog, ErrorState, GlyphSpinner, host, RowButton, useI18n, useValue } from '@hermes/plugin-sdk'
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
  const b = useBots()
  const { t } = useI18n()
  const connection = useValue(host.state.connectionId || unknownConnection) || host.activeConnectionId?.() || ''
  const gateway = useValue(host.state.gateway)
  const directories = useValue($hostedDirectories)
  const rooms = useValue($hostedRooms)
  const directory = directories[connection]
  useEffect(() => {
    if (connection && gateway === 'open') {void discoverHostedRooms(connection)}
  }, [connection, gateway])

  return (
    <section aria-label={b.hosted.groupChats} className="grid gap-1 px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-(--ui-text-secondary)">{b.hosted.groupChats}</span>
        <Button disabled={!connection || directory?.loading} onClick={() => void discoverHostedRooms(connection)} size="xs" variant="ghost">
          {t.common.refresh}
        </Button>
      </div>
      {directory?.loading ? <GlyphSpinner spinner="breathe" /> : null}
      {directory?.error ? <div className="text-xs text-destructive" role="alert">{directory.error}</div> : null}
      {!directory?.loading && !directory?.error && !directory?.keys.length ? (
        <p className="text-xs text-(--ui-text-tertiary)">{b.hosted.empty}</p>
      ) : null}
      {directory?.keys.map(key => rooms[key] ? (
        <RowButton className="flex min-w-0 items-center gap-2 text-left text-xs" key={key} onClick={() => onOpen(key)}>
          <Codicon name="organization" />
          <span className="truncate">{rooms[key].name}</span>
          <span className="ml-auto text-(--ui-text-tertiary)">{b.hosted.hosted}</span>
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
  const b = useBots()
  const { t } = useI18n()
  const [loaded, setLoaded] = useState<Attachment | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const loading = useRef(false)
  const name = attachment.name || b.hosted.attachment

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
          <a className="text-(--ui-accent) underline" download={attachment.name || 'attachment'} href={loaded.data}>{t.fileMenu.download} {name}</a>
        </>
      ) : (
        <Button disabled={busy} onClick={() => void load()} size="xs" variant="ghost">
          {busy ? b.hosted.loadingAttachment(name) : b.hosted.openAttachment(name)}
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
  const { t } = useI18n()
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
  const status = telegram?.blocked ? b.hosted.telegramBlocked : !driver?.running ? b.hosted.workerUnavailable : driver.blocked ? b.hosted.blocked : driver.working ? t.sidebar.statusDivider.working : b.hosted.idle
  const actions = hostedPendingActions(driver)
  const reported = Array.isArray(driver?.pending_actions) ? driver.pending_actions.length : 0
  const unsupported = Math.max(0, reported - actions.length)
  // Held members come from the gateway that owns the room; this view never mints or clears a
  // hold, and the release gesture is the ordinary room message the backend already understands.
  const holds = hostedHolds(driver)

  return (
    <div className="grid gap-1 border-b border-(--ui-stroke-tertiary) px-2.5 pb-2 text-xs text-(--ui-text-secondary)">
      <p>{b.hosted.gatewayHosted} · {cache?.capabilities?.attachments ? b.hosted.attachmentsEnabled : b.hosted.textOnly} · {b.group.hostedDiscussion}</p>
      <p>{cache?.capabilities?.persistentProcess ? b.hosted.persistentProcess : b.hosted.noPersistence}</p>
      <div className="flex items-center gap-2" role="status">
        {cache?.loading ? <GlyphSpinner spinner="breathe" /> : null}
        <span>{status} · {b.hosted.replayedThrough(cache?.cursor || 0)}</span>
        <Button disabled={cache?.busy} onClick={() => void refreshHostedRoom(roomKey)} size="xs" variant="ghost">{b.hosted.refreshRoom}</Button>
        <Button disabled={cache?.busy || !driver?.running} onClick={() => void stopHostedRoom(roomKey)} size="xs" variant="ghost">{b.hosted.stopRoomWork}</Button>
      </div>
      {delivery ? (
        <div className="grid min-w-0 gap-1" role="status">
          <p className="break-words">{b.hosted.deliveryDetail(telegram.chat_id, delivery.profile, delivery.event_id, delivery.chunk_index + 1, delivery.attempt, b.hosted.deliveryStatus[delivery.status])}</p>
          {delivery.retry_after ? <p>{b.hosted.telegramCooldown(new Date(delivery.retry_after * 1000).toLocaleString())}</p> : null}
          {telegram.blocked ? <p>{b.hosted.telegramReadback}</p> : null}
          {telegram.blocked && !cache?.capabilities?.telegramRecovery ? <p>{b.hosted.noTelegramRecovery}</p> : null}
          {telegram.blocked && cache?.capabilities?.telegramRecovery ? (
            <>
              <label className="grid gap-1">
                {b.hosted.telegramMessageId}
                <input className="min-w-0 rounded border border-(--ui-stroke-secondary) p-1" disabled={cache.busy} inputMode="numeric" onChange={event => setTelegramId(event.target.value)} value={telegramId} />
              </label>
              <div className="flex flex-wrap gap-2">
                <Button disabled={cache.busy || !Number.isSafeInteger(messageId) || messageId <= 0} onClick={() => setResolution({ roomKey, chatId: telegram.chat_id, delivery, decision: 'confirmed-delivered', messageId })} size="xs" variant="secondary">{b.hosted.markDelivered}</Button>
                <Button disabled={cache.busy || Boolean(delivery.retry_after && delivery.retry_after * 1000 > Date.now())} onClick={() => setResolution({ roomKey, chatId: telegram.chat_id, delivery, decision: 'retry' })} size="xs" variant="ghost">{b.hosted.authorizeDeliveryRetry}</Button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}
      {holds.length ? (
        <p role="status">
          {b.hosted.pausedByYou(holds.map(hold => hold.label).join(', '), holds[0].handle)}
        </p>
      ) : null}
      {cache?.capabilities && !cache.capabilities.persistentHolds ? (
        <p>{b.hosted.noPersistentHolds}</p>
      ) : null}
      {actions.map(action => (
        <div className="grid gap-1 rounded-md border border-(--ui-stroke-secondary) p-2" key={`${action.kind}:${action.taskId}:${action.requestId || ''}:${action.executionGeneration || ''}`}>
          {action.kind === 'retry' ? (
            <>
              <p>{b.hosted.uncertainTask(action.taskId)}</p>
              <Button disabled={cache?.busy} onClick={() => void retryHostedTask(roomKey, action.taskId)} size="xs" variant="secondary">{b.hosted.retryTask}</Button>
            </>
          ) : (
            <>
              <p>{b.hosted.waitingForApproval(cache?.room?.members.find(member => member.member_id === action.memberId)?.display_name || action.memberId || '')}</p>
              {action.reason ? <p>{action.reason}</p> : null}
              {action.command?.trim() ? <pre className="max-h-40 overflow-auto whitespace-pre-wrap" data-selectable-text="true">{action.command}</pre> : <p>{b.hosted.noCommandPreview}</p>}
              <div className="flex gap-2">
                {(action.choices || []).map(choice => (
                  <Button disabled={cache?.busy} key={choice} onClick={() => void approveHostedTask(roomKey, action, choice)} size="xs" variant={choice === 'once' ? 'secondary' : 'ghost'}>
                    {choice === 'once' ? b.hosted.allowOnce : b.hosted.deny}
                  </Button>
                ))}
              </div>
            </>
          )}
        </div>
      ))}
      {unsupported ? <p role="status">{b.hosted.unsupportedActions(unsupported)}</p> : null}
      {!cache ? <p role="alert">{b.hosted.identityUnavailable}</p> : null}
      {cache?.error ? <div role="alert"><ErrorState description={cache.error} title={b.hosted.roomUnavailable} /></div> : null}
      {cache?.pending ? (
        <div className="grid gap-1">
          <p>{b.hosted.pendingInput(cache.pending.text)}</p>
          {cache.pending.attachments?.length ? <p>{b.hosted.savedAttachments(cache.pending.attachments.map(attachment => attachment.name || b.hosted.attachment).join(', '))}</p> : null}
          <Button disabled={cache.busy} onClick={() => void sendHostedInput(roomKey)} size="xs" variant="secondary">{b.hosted.retrySavedInput}</Button>
          <Button disabled={cache.busy} onClick={() => setDiscard({ roomKey, eventId: cache.pending!.eventId })} size="xs" variant="ghost">{b.group.discardSavedInput}</Button>
        </div>
      ) : null}
      <ConfirmDialog
        confirmLabel={resolution?.decision === 'retry' ? b.hosted.retryPart : b.hosted.confirmDelivered}
        description={resolution ? `${b.hosted.deliveryConfirmation(resolution.chatId, resolution.delivery.profile, resolution.delivery.event_id, resolution.delivery.chunk_index + 1, resolution.delivery.attempt)} ${resolution.decision === 'retry'
          ? b.hosted.retryPartWarning
          : b.hosted.confirmDeliveredWarning(resolution.messageId!)}` : ''}
        destructive={resolution?.decision === 'retry'}
        onClose={() => setResolution(null)}
        onConfirm={async () => {
          if (resolution) {await resolveHostedTelegramDelivery(resolution.roomKey, resolution.delivery, resolution.decision, resolution.messageId)}
        }}
        open={resolution !== null}
        title={b.hosted.reconcileDeliveryTitle}
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
