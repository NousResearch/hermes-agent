import { Button, gatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { CanonicalGroupFiles } from './canonical-group-files'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupBinding } from './canonical-groups'

/** Files, rename and disband for a gateway room, each offered only when its gateway advertises it. */
export function CanonicalGroupRoomActions({ binding, name, latestFileSeq = 0, visible = true, onChanged, onDisbanded }: {
  binding: CanonicalGroupBinding; name: string; latestFileSeq?: number; visible?: boolean; onChanged: () => void
  onDisbanded?: () => void
}) {
  const labels = useCanonicalGroupLabels()
  const [methods, setMethods] = useState<string[]>([])
  const [draft, setDraft] = useState<null | string>(null)
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  // One intended name keeps one event id across retries; a different name is a new intent.
  const renameIntent = useRef<null | { name: string; eventId: string }>(null)

  useEffect(() => {
    let current = true
    void readGroupExecutionMode(binding, gatewayActivationEpoch()).then(surface => {
      if (current) {setMethods(surface.methods ?? [])}
    })

    return () => { current = false }
  }, [binding])

  const run = async (operation: () => Promise<void>) => {
    setBusy(true)
    setError('')

    try {
      await operation()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
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

  const disband = () => void run(async () => {
    const result = await canonicalGroupRequest<{ tombstone?: boolean } | undefined>(binding, 'groups.disband', {
      room_id: binding.roomId, cancel_id: crypto.randomUUID()
    })

    if (result?.tombstone !== true) {throw new Error(labels.disbandUnconfirmed)}
    onDisbanded?.()
  })

  return <>
    {visible && methods.includes('groups.attachment.list') &&
      <CanonicalGroupFiles binding={binding} latestFileSeq={latestFileSeq} roomName={name} />}
    {methods.includes('groups.rename') && (draft === null
      ? <Button disabled={busy} onClick={() => setDraft(name)}>{labels.rename}</Button>
      : <form className="flex gap-1" onSubmit={event => { event.preventDefault(); rename() }}>
        <input aria-label={labels.roomName} maxLength={120} onChange={event => setDraft(event.target.value)} value={draft} />
        <Button disabled={busy || !draft.trim()} type="submit">{labels.save}</Button>
        <Button disabled={busy} onClick={() => setDraft(null)}>{labels.cancel}</Button>
      </form>)}
    {methods.includes('groups.disband') && <Button disabled={busy} onClick={() => setConfirming(true)}>{labels.disband}</Button>}
    {confirming && <div aria-label={labels.disband} role="alertdialog">
      <p>{labels.disbandWarning}</p>
      <Button disabled={busy} onClick={disband}>{labels.confirmDisband}</Button>
      <Button disabled={busy} onClick={() => setConfirming(false)}>{labels.cancel}</Button>
    </div>}
    {error && <p role="alert">{error}</p>}
  </>
}
