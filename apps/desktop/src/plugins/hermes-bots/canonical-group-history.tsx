import * as sdk from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { type CanonicalGroupAttachment, CanonicalGroupAttachments, fetchCanonicalGroupAttachment } from './canonical-group-attachments'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'

const MessageTextContent = typeof sdk === 'undefined' ? undefined : sdk.MessageTextContent
const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown

export interface CanonicalGroupEvent {
  seq: number
  event_id?: string
  room_id?: string
  kind: string
  payload: { text?: string; content?: string; attachments?: CanonicalGroupAttachment[]; member_id?: string; error?: string; reason?: string; reason_code?: string; status?: string }
  actor?: { kind?: string; id?: string; display_name?: string }
  created_at?: number
}

interface CanonicalGroupHistoryProps {
  binding: CanonicalGroupBinding
  events: CanonicalGroupEvent[]
  members?: CanonicalRoomMember[]
  disabled?: boolean
}

function eventAuthor(event: CanonicalGroupEvent, roster: Map<string, CanonicalRoomMember>, isMessage: boolean) {
  const memberId = isMessage
    ? event.actor?.kind === 'member' ? event.actor.id : undefined
    : event.payload.member_id

  const member = roster.get(memberId ?? '')

  // The log's label preserves who spoke even if the roster has since changed.
  return (isMessage ? event.actor?.display_name : undefined) || member?.display_name || member?.handle ||
    (isMessage ? event.actor?.id : memberId)
}

function EventTime({ timestamp }: { timestamp?: number }) {
  if (typeof timestamp !== 'number') {return null}
  const at = new Date(timestamp * 1000)

  if (!Number.isFinite(at.getTime())) {return null}

  return <time className="text-[0.625rem] text-(--ui-text-quaternary)" dateTime={at.toISOString()}>{sdk.relativeTime(at.getTime())}</time>
}

function EventBody({ event, notice }: { event: CanonicalGroupEvent; notice?: string }) {
  if (notice) {
    const detail = event.payload.error || event.payload.reason || event.payload.reason_code

    return <p className="text-xs text-(--ui-text-secondary)" data-selectable-text="true">
      <strong>{notice}</strong>{detail && <>: {detail}</>}
    </p>
  }

  const text = event.payload.text || event.payload.content || ''

  return <div className="min-w-0 text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:whitespace-pre-wrap [&_pre]:wrap-anywhere" data-selectable-text="true">
    {/* The shared renderer (layout and large-message safeguards). media={false}: room text
        may come from another machine, so its MEDIA: paths never resolve against this gateway;
        room files keep their own download path below. */}
    {MessageTextContent ? <MessageTextContent media={false} text={text} />
      : Streamdown ? <Streamdown>{text}</Streamdown> : text}
  </div>
}

// Previews download the whole file over RPC; larger or unsized images keep only Download.
const IMAGE_PREVIEW_MAX_BYTES = 5 * 1024 * 1024

// Automatic previews share one small download budget across the renderer, in request order;
// a file already downloading is joined rather than requested again. Manual Download stays outside it.
type Preview = Awaited<ReturnType<typeof fetchCanonicalGroupAttachment>>
const previewDownloads = new Map<string, Promise<Preview>>()
const previewWaiters: Array<() => void> = []
let previewSlots = 2

function releasePreviewSlot() {
  const next = previewWaiters.shift()

  if (next) {next()} else {previewSlots++}
}

function queuePreview(key: string, fetch: () => Promise<Preview>, onPreview: (preview: Preview) => void) {
  let cancelled = false
  const deliver = (download: Promise<Preview>) => void download.then(preview => { if (!cancelled) {onPreview(preview)} }, () => undefined)

  const run = () => {
    let download = previewDownloads.get(key)

    if (download) {releasePreviewSlot()} else {
      download = fetch().finally(() => { previewDownloads.delete(key); releasePreviewSlot() })
      previewDownloads.set(key, download)
    }

    deliver(download)
  }

  const running = previewDownloads.get(key)

  if (running) {deliver(running)} else if (previewSlots > 0) {previewSlots--; run()} else {previewWaiters.push(run)}

  // A preview that leaves before its turn gives up its place without sending a request.
  return () => {
    cancelled = true
    const at = previewWaiters.indexOf(run)

    if (at >= 0) {previewWaiters.splice(at, 1)}
  }
}

function ImagePreview({ binding, eventId, attachment, disabled }: {
  binding: CanonicalGroupBinding; eventId: string; attachment: CanonicalGroupAttachment; disabled: boolean
}) {
  const [src, setSrc] = useState('')
  const [near, setNear] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const { attachment_id: attachmentId, size } = attachment
  const { connectionId, profile, roomId } = binding
  const previewable = !!attachmentId && size !== undefined && size <= IMAGE_PREVIEW_MAX_BYTES

  // History loads the whole room: fetch only near the viewport. Leaving it before the bytes arrive releases the queued slot.
  useEffect(() => {
    const el = box.current

    if (src || disabled || !el) {return}
    const observer = new IntersectionObserver(entries => setNear(entries[entries.length - 1].isIntersecting), { rootMargin: '200px' })
    observer.observe(el)

    return () => observer.disconnect()
  }, [src, disabled])

  useEffect(() => {
    if (disabled || !near || !attachmentId || !previewable) {return}
    let url = ''
    const key = JSON.stringify([connectionId, profile, roomId, eventId, attachmentId])

    // A failed preview leaves the Download button as the way to the file.
    const cancel = queuePreview(key, () => fetchCanonicalGroupAttachment({ connectionId, profile, roomId }, eventId, attachmentId), ({ bytes, mime }) => {
      url = URL.createObjectURL(new Blob([bytes], { type: mime }))
      setSrc(url)
    })

    // Dropping `near` makes a re-shown room wait on the observer again, not refetch every preview seen so far at once.
    return () => { cancel(); URL.revokeObjectURL(url); setSrc(''); setNear(false) }
  }, [connectionId, profile, roomId, eventId, attachmentId, disabled, near, previewable])

  return previewable ? <div ref={box}>{src && <img alt={attachment.name} className="max-h-64 max-w-full rounded object-contain" src={src} />}</div> : null
}

export function CanonicalGroupHistory({ binding, events, members = [], disabled = false }: CanonicalGroupHistoryProps) {
  const labels = useCanonicalGroupLabels()
  const roster = new Map(members.map(member => [member.member_id, member]))

  const notices = new Map([
    ['turn.failed', labels.turnFailed],
    ['turn.cancelled', labels.turnCancelled],
    ['turn.deferred', labels.turnDeferred],
    ['member.unavailable', labels.memberUnavailable],
    ['room.stop_requested', labels.roomStopped],
    ['room.disbanded', labels.roomDisbanded],
    ['authority.lost', labels.authorityLost]
  ])

  return <>{events.map(event => {
    const notice = event.kind === 'room.activity' && event.payload.status === 'bounded'
      ? labels.workLimitReached : notices.get(event.kind)

    const isMessage = event.kind === 'message' || event.kind === 'message.user' || event.kind === 'message.member'

    if (!notice && !isMessage) {return null}
    const author = eventAuthor(event, roster, isMessage)
    const previewEventId = event.room_id === binding.roomId ? event.event_id : undefined

    return <div className="grid grid-cols-[minmax(0,1fr)] gap-1 px-2 py-1.5" key={event.seq}>
    <div className="flex items-center gap-2">
      {author && <strong className="text-[0.7rem] font-semibold text-(--ui-accent)">{author}</strong>}
      <EventTime timestamp={event.created_at} />
    </div>
    <EventBody event={event} notice={notice} />
    {previewEventId && event.payload.attachments?.filter(a => a.kind === 'image').map(a =>
      <ImagePreview attachment={a} binding={binding} disabled={disabled} eventId={previewEventId} key={a.attachment_id ?? a.name} />)}
    {!!event.payload.attachments?.length && <CanonicalGroupAttachments
      attachments={event.payload.attachments.map(attachment => ({ ...attachment, event_id: event.event_id }))}
      binding={binding} disabled={disabled || !event.event_id || event.room_id !== binding.roomId} readOnly />}
  </div>})}</>
}
