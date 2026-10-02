import { Button, ConfirmDialog, gatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupBinding } from './canonical-groups'

/** Rename and disband for a gateway room, offered only when its gateway advertises them. */
export function CanonicalGroupRoomActions({ binding, name, onChanged, onDisbanded }: {
  binding: CanonicalGroupBinding; name: string; onChanged: () => void; onDisbanded?: () => void
}) {
  const labels = useCanonicalGroupLabels()
  const [methods, setMethods] = useState<string[]>([])
  const [draft, setDraft] = useState<null | string>(null)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef(false)
  const disbandIntent = useRef<string | null>(null)
  // One intended name keeps one event id across retries; a different name is a new intent.
  const renameIntent = useRef<null | { name: string; eventId: string }>(null)

  useEffect(() => {
    let current = true
    void readGroupExecutionMode(binding, gatewayActivationEpoch()).then(surface => {
      if (current) {setMethods(surface.methods ?? [])}
    })

    return () => { current = false }
  }, [binding])

  const run = async (operation: () => Promise<void>, rethrow = false) => {
    if (pending.current) {if (rethrow) {throw new Error(labels.pendingActionUnconfirmed)}; return}
    pending.current = true
    setBusy(true)
    setError('')

    try {
      await operation()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      if (rethrow) {throw e}
    } finally {
      pending.current = false
      setBusy(false)
    }
  }

  const rename = () => {
    const next = draft?.trim()

    if (!next || next === name) {
      setDraft(null)

      return
    }

    if (renameIntent.current?.name !== next) {renameIntent.current = { name: next, eventId: crypto.randomUUID() }}
    const intent = renameIntent.current
    void run(async () => {
      await canonicalGroupRequest(binding, 'groups.rename', { room_id: binding.roomId, event_id: intent.eventId, name: intent.name })
      renameIntent.current = null
      setDraft(null)
      onChanged()
    })
  }

  const disband = () => {
    disbandIntent.current ??= crypto.randomUUID()
    return run(async () => {
      const result = await canonicalGroupRequest<{ tombstone?: { room_id: string; disbanded_at: number } } | undefined>(binding, 'groups.disband', {
        room_id: binding.roomId, cancel_id: disbandIntent.current
      })

      const tombstone = result?.tombstone
      if (tombstone?.room_id !== binding.roomId || !Number.isFinite(tombstone.disbanded_at)) {throw new Error(labels.disbandUnconfirmed)}
      disbandIntent.current = null
      onDisbanded?.()
    }, true)
  }

  return <>
    {methods.includes('groups.rename') && (draft === null
      ? <Button disabled={busy} onClick={() => setDraft(name)}>{labels.rename}</Button>
      : <form className="flex gap-1" onSubmit={event => { event.preventDefault(); rename() }}>
        <input aria-label={labels.roomName} maxLength={120} onChange={event => setDraft(event.target.value)} value={draft} />
        <Button disabled={busy || !draft.trim()} type="submit">{labels.save}</Button>
        <Button disabled={busy} onClick={() => setDraft(null)}>{labels.cancel}</Button>
      </form>)}
    {methods.includes('groups.disband') && <Button disabled={busy} onClick={() => setConfirming(true)}>{labels.disband}</Button>}
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={labels.confirmDisband} description={labels.disbandWarning}
      destructive onClose={() => setConfirming(false)} onConfirm={disband} open={confirming} title={labels.disband} />
    {error && <p role="alert">{error}</p>}
  </>
}
