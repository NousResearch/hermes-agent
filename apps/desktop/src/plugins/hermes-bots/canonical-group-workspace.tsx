import { Button, host } from '@hermes/plugin-sdk'
import { type ClipboardEvent, type DragEvent, useEffect, useRef, useState } from 'react'

import {
  type CanonicalDriverActivity,
  CanonicalGroupWorkingLine,
  canonicalMemberPhases
} from './canonical-group-activity'
import { CanonicalGroupAttachments, type CanonicalGroupUploader } from './canonical-group-attachments'
import { CanonicalGroupHeader } from './canonical-group-header'
import { type CanonicalGroupEvent, CanonicalGroupHistory } from './canonical-group-history'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { forgetCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'
import type { PreparedCanonicalGroupSend } from './canonical-group-send'
import { actCanonicalGroup, canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalPendingAction, CanonicalRoom } from './canonical-groups'
import { GroupMentionInput } from './group-chat-parts'

type RoomEvent = CanonicalGroupEvent
interface Attachment {
  attachment_id?: string
  event_id?: string
  kind: string
  name: string
  mime: string
  size?: number
}
interface RoomState {
  room: CanonicalRoom
  driver_status?: CanonicalDriverActivity & { pending_actions?: CanonicalPendingAction[] }
}

const sameGroupPayload = (payload: Record<string, unknown>, text: string, attachments: Attachment[]) =>
  payload.text === text && JSON.stringify(payload.attachments ?? []) === JSON.stringify(attachments)

export function CanonicalGroupWorkspace({
  binding,
  visible = true,
  onBack
}: {
  binding: CanonicalGroupBinding
  visible?: boolean
  onBack?: () => void
}) {
  // Remount on identity changes: old polls and pending confirmations never cross rooms.
  return (
    <CanonicalRoomView
      binding={binding}
      key={JSON.stringify([binding.connectionId, binding.profile, binding.roomId])}
      onBack={onBack}
      visible={visible}
    />
  )
}

function CanonicalRoomView({
  binding: initialBinding,
  visible,
  onBack
}: {
  binding: CanonicalGroupBinding
  visible: boolean
  onBack?: () => void
}) {
  const [binding] = useState(() => ({ ...initialBinding }))
  const labels = useCanonicalGroupLabels()
  const [state, setState] = useState<RoomState | null>(null)
  const [events, setEvents] = useState<RoomEvent[]>([])
  const [error, setError] = useState('')
  const [readError, setReadError] = useState('')
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploading, setUploading] = useState(false)
  const [restored, setRestored] = useState(false)
  const [pending, setPending] = useState<PreparedCanonicalGroupSend | null>(null)
  // Only a journal entry found on mount is "restored"; this window's own in-flight send also sets `pending`.
  const [restoredPending, setRestoredPending] = useState(false)
  const uploader = useRef<CanonicalGroupUploader>(null)
  const [busy, setBusy] = useState(false)
  const busyRef = useRef(false)
  const alive = useRef(true)
  const [discard, setDiscard] = useState<CanonicalPendingAction | null>(null)
  const revision = useRef(0)
  // This window's typed draft while it sends the room's one pending intent another window prepared.
  const heldDraft = useRef<{ text: string; attachments: Attachment[] } | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- journal hydration and mounted lifetime, not a reactive store mirror
  useEffect(() => {
    alive.current = true
    let cancelled = false
    void readCanonicalGroupSend(binding)
      .then(entry => {
        if (cancelled) {
          return
        }

        if (entry) {
          setPending(entry)
          setRestoredPending(true)
          setDraft(String(entry.params.payload.text ?? ''))
          setAttachments((entry.params.payload.attachments as Attachment[] | undefined) ?? [])
        }

        setRestored(true)
      })
      .catch(e => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e))
        }
      })

    return () => {
      cancelled = true
      alive.current = false
      revision.current++
    }
  }, [binding])

  const refresh = async () => {
    const version = ++revision.current
    const snapshot = await canonicalGroupRequest<RoomState>(binding, 'groups.state', { room_id: binding.roomId })
    const log: RoomEvent[] = []
    let cursor = 0

    for (;;) {
      const page = await canonicalGroupRequest<{ events: RoomEvent[]; has_more?: boolean; next_seq?: number }>(
        binding,
        'groups.log',
        { room_id: binding.roomId, since_seq: cursor, limit: 100 }
      )
      log.push(...page.events)

      if (!page.has_more) {
        break
      }
      const next = page.events.at(-1)?.seq

      if (!next || next <= cursor) {
        throw new Error(labels.invalidLogCursor)
      }
      cursor = next
    }

    if (alive.current && version === revision.current) {
      setState(snapshot)
      setEvents(log)
      setReadError('')
    }
  }

  useEffect(() => {
    if (!visible) {
      return
    }
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = async () => {
      try {
        await refresh()
      } catch (e) {
        if (!cancelled) {
          setReadError(String(e instanceof Error ? e.message : e))
        }
      }

      if (!cancelled) {
        timer = setTimeout(() => void poll(), 2000)
      }
    }

    void poll()

    return () => {
      cancelled = true
      revision.current++
      clearTimeout(timer)
    }
    // The keyed parent freezes the authority binding for this lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  const mutate = async (operation: () => Promise<unknown>, settle: () => unknown = refresh) => {
    if (busyRef.current) {
      return
    }
    busyRef.current = true
    setBusy(true)
    setError('')

    try {
      await operation()

      if (alive.current) {
        await settle()
      }
    } catch (e) {
      if (alive.current) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      busyRef.current = false

      if (alive.current) {
        setBusy(false)
      }
    }
  }

  const send = () => {
    if (
      !restored ||
      busyRef.current ||
      uploading ||
      !state?.driver_status ||
      (!pending && !draft.trim() && !attachments.length)
    ) {
      return
    }
    void mutate(async () => {
      const exact = pending ?? (await prepareCanonicalGroupSend(binding, { text: draft, attachments }))

      if (!alive.current) {
        return
      }

      // Another window won the room's slot: send its intent first, keep this window's words.
      if (!pending && !sameGroupPayload(exact.params.payload, draft, attachments)) {
        heldDraft.current = { text: draft, attachments }
      }

      setPending(exact)
      setDraft(String(exact.params.payload.text ?? ''))
      setAttachments((exact.params.payload.attachments as Attachment[] | undefined) ?? [])
      await canonicalGroupRequest(exact.binding, 'groups.send', exact.params)
      await retireCanonicalGroupSend(exact.binding, exact.params.event_id)

      if (alive.current) {
        const held = heldDraft.current
        heldDraft.current = null
        setPending(null)
        setRestoredPending(false)
        setDraft(held?.text ?? '')
        setAttachments(held?.attachments ?? [])
      }
    })
  }

  const act = (action: CanonicalPendingAction, choice?: 'once' | 'deny') =>
    mutate(() => actCanonicalGroup(binding, action, choice))

  const rename = (name: string) =>
    void mutate(() =>
      canonicalGroupRequest(binding, 'groups.rename', { room_id: binding.roomId, event_id: crypto.randomUUID(), name })
    )

  // Rejects so ConfirmDialog stays open with the error; a disbanded room no longer reads back, so leave it.
  const end = async () => {
    await canonicalGroupRequest(binding, 'groups.disband', { room_id: binding.roomId, cancel_id: crypto.randomUUID() })

    if (alive.current) {
      onBack?.()
    }
    forgetCanonicalGroup(binding)
    host.notify({ kind: 'success', message: labels.disbanded(state?.room.name || binding.roomName || '') })
  }

  const composerLocked = !restored || busy || !!pending
  const phases = canonicalMemberPhases(state?.driver_status)
  const mentionMembers = (state?.room.members ?? []).map(member => ({
    name: member.handle,
    handle: member.handle,
    display_name: member.display_name
  }))

  // Text pastes stay in the composer; only files go through the picker's upload.
  const attachTransfer = (event: ClipboardEvent | DragEvent, data: DataTransfer | null) => {
    const files = [...(data?.files ?? [])]

    if (!files.length) {
      return
    }
    event.preventDefault()

    if (!composerLocked) {
      uploader.current?.upload(files)
    }
  }

  return (
    <section className="flex h-full min-h-0 flex-col gap-3 p-3">
      <CanonicalGroupHeader
        busy={busy}
        canStop={!!state?.driver_status}
        onBack={onBack}
        onDisband={end}
        onRename={rename}
        onStop={() =>
          void mutate(() =>
            canonicalGroupRequest(binding, 'groups.stop', { room_id: binding.roomId, cancel_id: crypto.randomUUID() })
          )
        }
        phases={phases}
        room={state?.room}
      />
      {readError && (
        <div role="alert">
          {readError}
          <Button onClick={() => void refresh().catch(e => setReadError(String(e)))}>{labels.refresh}</Button>
        </div>
      )}
      {error && <div role="alert">{error}</div>}
      {state && !state.driver_status && <p>{labels.driverUnavailable}</p>}
      <div className="min-h-0 flex-1 overflow-auto" role="log">
        <CanonicalGroupHistory binding={binding} disabled={!visible} events={events} members={state?.room.members} />
        <CanonicalGroupWorkingLine
          driver={state?.driver_status}
          labels={labels}
          members={state?.room.members}
          phases={phases}
        />
      </div>
      {(state?.driver_status?.pending_actions || []).map(action => (
        <div
          className="flex items-center gap-2"
          key={`${action.kind}:${action.task_id}:${action.execution_generation}`}
        >
          <span>{action.member_id}</span>
          {action.kind === 'discard' && (
            <Button disabled={busy} onClick={() => setDiscard({ ...action })}>
              {labels.discard}
            </Button>
          )}
          {action.kind === 'retry' && (
            <Button disabled={busy} onClick={() => void act({ ...action })}>
              {labels.retry}
            </Button>
          )}
          {action.kind === 'approval' && (
            <>
              <Button disabled={busy} onClick={() => void act({ ...action }, 'once')}>
                {labels.allowOnce}
              </Button>
              <Button disabled={busy} onClick={() => void act({ ...action }, 'deny')}>
                {labels.deny}
              </Button>
            </>
          )}
        </div>
      ))}
      {discard && (
        <div aria-label={labels.discardUnknown} role="alertdialog">
          <p>{labels.discardWarning}</p>
          <Button
            disabled={busy}
            onClick={() => {
              const exact = discard
              setDiscard(null)
              void act(exact)
            }}
          >
            {labels.confirmDiscard}
          </Button>
          <Button onClick={() => setDiscard(null)}>{labels.cancel}</Button>
        </div>
      )}
      {restoredPending && <p role="status">{labels.restoredPendingSend}</p>}
      <form
        className="flex gap-2"
        onDragOver={event => {
          if (event.dataTransfer.types.includes('Files')) {
            event.preventDefault()
          }
        }}
        onDrop={event => attachTransfer(event, event.dataTransfer)}
        onPaste={event => attachTransfer(event, event.clipboardData)}
        onSubmit={event => {
          event.preventDefault()
          send()
        }}
      >
        <CanonicalGroupAttachments
          attachments={attachments}
          binding={binding}
          disabled={composerLocked}
          onChange={setAttachments}
          onUploadingChange={setUploading}
          ref={uploader}
        />
        <GroupMentionInput
          aria-label={labels.groupMessage}
          className="min-w-0 flex-1"
          disabled={composerLocked}
          handleFor={member => member.handle}
          members={mentionMembers}
          onChange={setDraft}
          onSubmitDraft={send}
          value={draft}
        />
        <Button
          disabled={
            !restored ||
            busy ||
            uploading ||
            (!pending && !draft.trim() && !attachments.length) ||
            !state?.driver_status
          }
          type="submit"
        >
          {pending ? labels.retry : labels.send}
        </Button>
      </form>
    </section>
  )
}
