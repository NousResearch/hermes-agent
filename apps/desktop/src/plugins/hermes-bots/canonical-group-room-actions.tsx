import { Button, ConfirmDialog, Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, Input } from '@hermes/plugin-sdk'
import { useRef, useState } from 'react'

import { type GroupMethods, offersGroupMethod } from './canonical-group-capabilities'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupBinding } from './canonical-groups'

/** Rename and end a gateway room from its header, each offered only when the gateway lists its
 *  method. Each keeps one idempotency key per intent across retries, and ending removes the room
 *  only once the gateway returns its tombstone. */
export function CanonicalGroupRoomActions({ binding, name, methods, disabled, onRenamed, onEnded }: {
  binding: CanonicalGroupBinding
  name: string
  methods: GroupMethods
  disabled: boolean
  onRenamed: (name: string) => void
  onEnded: () => void
}) {
  const labels = useCanonicalGroupLabels()
  const [draft, setDraft] = useState<null | string>(null)
  const [ending, setEnding] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  // A retry of the same name is the same rename at the gateway; a different name is a new one.
  const renameIntent = useRef<null | { name: string; eventId: string }>(null)
  const endIntent = useRef<null | string>(null)

  const rename = async () => {
    const next = draft?.trim()

    if (!next || busy) {return}

    if (next === name) {
      setDraft(null)

      return
    }

    if (renameIntent.current?.name !== next) {renameIntent.current = { name: next, eventId: crypto.randomUUID() }}
    const intent = renameIntent.current
    setBusy(true)
    setError('')

    try {
      await canonicalGroupRequest(binding, 'groups.rename', { room_id: binding.roomId, event_id: intent.eventId, name: intent.name })
      renameIntent.current = null
      setDraft(null)
      onRenamed(intent.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // ConfirmDialog keeps itself open and shows a thrown message.
  const end = async () => {
    endIntent.current ??= crypto.randomUUID()
    setError('')
    let tombstone: { room_id?: unknown; disbanded_at?: unknown } | undefined

    try {
      tombstone = (await canonicalGroupRequest<{ tombstone?: typeof tombstone } | undefined>(binding, 'groups.disband', {
        room_id: binding.roomId, cancel_id: endIntent.current
      }))?.tombstone
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }

    if (tombstone?.room_id !== binding.roomId || typeof tombstone.disbanded_at !== 'number' || !Number.isFinite(tombstone.disbanded_at)) {
      throw new Error(labels.disbandUnconfirmed)
    }

    endIntent.current = null
    onEnded()
  }

  return <>
    {offersGroupMethod(methods, 'groups.rename') &&
      <Button disabled={disabled || busy} onClick={() => { setError(''); setDraft(name) }}>{labels.rename}</Button>}
    {offersGroupMethod(methods, 'groups.disband') &&
      <Button disabled={disabled} onClick={() => { setError(''); setEnding(true) }}>{labels.disband}</Button>}
    {draft !== null && <Dialog onOpenChange={open => { if (!open && !busy) {setDraft(null)} }} open>
      <DialogContent aria-describedby={undefined} className="max-w-sm">
        <DialogHeader><DialogTitle>{labels.rename}</DialogTitle></DialogHeader>
        <form className="grid gap-3" onSubmit={event => { event.preventDefault(); void rename() }}>
          {/* Create's limit: a rename never gives a room a name Create could not. */}
          <Input aria-label={labels.roomName} autoFocus disabled={busy} maxLength={64} onChange={event => setDraft(event.target.value)} value={draft} />
          {error && <div role="alert">{labels.pendingActionUnconfirmed}<details><summary>{labels.details}</summary>{error}</details></div>}
          <DialogFooter>
            <Button disabled={busy} onClick={() => setDraft(null)} type="button">{labels.cancel}</Button>
            <Button disabled={busy || !draft.trim()} type="submit">{labels.save}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>}
    {ending && <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={labels.disband} description={labels.disbandWarning} destructive
      onClose={() => setEnding(false)} onConfirm={end} open title={labels.disband}>
      {error && <details><summary>{labels.details}</summary>{error}</details>}
    </ConfirmDialog>}
  </>
}
