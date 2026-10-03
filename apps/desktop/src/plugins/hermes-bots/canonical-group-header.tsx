import { Button, Codicon, ConfirmDialog, Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle,
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
  gatewayActivationEpoch, Input, Popover, PopoverContent, PopoverTrigger, StatusDot, Tip } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { CanonicalMemberFace, canonicalMemberName } from './canonical-group-identity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import { canonicalGroupRequest, readGroupExecutionMode } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalRoomMember } from './canonical-groups'

export function CanonicalGroupHeader({ name, members, status, working, attention, visible = true, onBack, info, unavailable, children }: {
  name: string; members: CanonicalRoomMember[]; status?: string; working?: boolean; attention?: boolean
  visible?: boolean; onBack?: () => void
  /** Where the group runs, with its group info (backup copies). */
  info?: ReactNode
  /** Member id → why that Bot can't take part right now. */
  unavailable?: Record<string, string>
  children?: ReactNode
}) {
  const labels = useCanonicalGroupLabels()

  return <header className="flex shrink-0 items-center gap-3 px-4 py-3">
    {onBack && <Tip label={labels.back}><Button aria-label={labels.back} onClick={onBack} size="icon-xs" variant="ghost"><Codicon name="arrow-left" /></Button></Tip>}
    <div aria-label={labels.members} className="hidden shrink-0 items-center -space-x-1.5 sm:flex">
      {members.slice(0, 3).map(member => <div className="rounded-full bg-(--ui-bg-primary) p-0.5" key={member.member_id}>
        <CanonicalMemberFace member={member} name={canonicalMemberName(member, labels.unknownBot)} size={28} />
      </div>)}
    </div>
    <div className="min-w-0 flex-1">
      <h2 className="truncate text-sm font-medium text-(--ui-text-primary)"><bdi>{name}</bdi></h2>
      <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {!!members.length && (visible ? <Popover>
          <PopoverTrigger asChild><Button aria-label={`${labels.members}: ${labels.memberCount.replace('{count}', String(members.length))}`} size="inline" variant="text">
            {labels.memberCount.replace('{count}', String(members.length))}
          </Button></PopoverTrigger>
          <PopoverContent align="start" className="max-h-72 w-64 overflow-y-auto" variant="menu">
            <p className="mb-3 text-xs font-medium text-(--ui-text-secondary)">{labels.members}</p>
            <ul aria-label={labels.members} className="grid gap-3">
              {members.map(member => <li className="flex min-w-0 items-center gap-2" key={member.member_id}>
                <CanonicalMemberFace member={member} name={canonicalMemberName(member, labels.unknownBot)} />
                <span className="min-w-0 truncate text-xs"><bdi>{canonicalMemberName(member, labels.unknownBot)}</bdi>
                  {unavailable?.[member.member_id] && <span className="text-(--ui-text-tertiary)"> · {unavailable[member.member_id]}</span>}</span>
              </li>)}
            </ul>
          </PopoverContent>
        </Popover> : <span>{labels.memberCount.replace('{count}', String(members.length))}</span>)}
        {visible && info}
        {status && <span aria-live="polite" className="flex min-w-0 items-center gap-1.5"><StatusDot tone={attention ? 'warn' : working ? 'good' : 'muted'} /><span className="truncate">{status}</span></span>}
      </div>
    </div>
    <div className="flex shrink-0 items-center gap-1">{children}</div>
  </header>
}

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

  const run = async (operation: () => Promise<void>) => {
    if (pending.current) {return}
    pending.current = true
    setBusy(true)
    setError('')

    try {
      await operation()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
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

  const disband = async () => {
    if (pending.current) {throw new Error(labels.actionInFlight)}
    pending.current = true
    disbandIntent.current ??= crypto.randomUUID()
    setBusy(true)
    setError('')

    try {
      const result = await canonicalGroupRequest<{ tombstone?: { room_id: string; disbanded_at: number } } | undefined>(binding, 'groups.disband', {
        room_id: binding.roomId, cancel_id: disbandIntent.current
      })

      const tombstone = result?.tombstone

      if (tombstone?.room_id !== binding.roomId || !Number.isFinite(tombstone.disbanded_at)) {
        throw new Error(labels.disbandUnconfirmed)
      }

      disbandIntent.current = null
      onDisbanded?.()
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error))
      throw new Error(labels.disbandUnconfirmed)
    } finally {pending.current = false; setBusy(false)}
  }

  return <>
    {(methods.includes('groups.rename') || methods.includes('groups.disband')) && <DropdownMenu>
      <DropdownMenuTrigger asChild><Button aria-label={labels.groupActions} disabled={busy} size="icon-xs" variant="ghost"><Codicon name="ellipsis" /></Button></DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {methods.includes('groups.rename') && <DropdownMenuItem onSelect={() => { setError(''); setDraft(name) }}><Codicon name="edit" />{labels.rename}</DropdownMenuItem>}
        {methods.includes('groups.rename') && methods.includes('groups.disband') && <DropdownMenuSeparator />}
        {methods.includes('groups.disband') && <DropdownMenuItem onSelect={() => { setError(''); setConfirming(true) }} variant="destructive"><Codicon name="close" />{labels.disband}</DropdownMenuItem>}
      </DropdownMenuContent>
    </DropdownMenu>}
    <Dialog onOpenChange={open => { if (!open && !busy) {setDraft(null)} }} open={draft !== null}>
      <DialogContent aria-describedby={undefined} className="max-w-sm">
        <DialogHeader><DialogTitle>{labels.rename}</DialogTitle></DialogHeader>
        <form className="grid gap-4" onSubmit={event => { event.preventDefault(); rename() }}>
          <Input aria-label={labels.roomName} autoFocus disabled={busy} maxLength={120} onChange={event => setDraft(event.target.value)} value={draft ?? ''} />
          {error && <div className="grid gap-1 text-xs text-destructive" role="alert"><p>{labels.pendingActionUnconfirmed}</p>
            <details className="text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary><p className="mt-1 whitespace-pre-wrap break-words">{error}</p></details>
          </div>}
          <DialogFooter><Button disabled={busy} onClick={() => setDraft(null)} type="button" variant="ghost">{labels.cancel}</Button>
            <Button disabled={busy || !draft?.trim()} loading={busy} type="submit" variant="secondary">{labels.save}</Button></DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
    <ConfirmDialog cancelLabel={labels.cancel} confirmLabel={labels.confirmDisband} description={labels.disbandWarning}
      destructive onClose={() => setConfirming(false)} onConfirm={disband} open={confirming} title={labels.disband}>
      {error && error !== labels.disbandUnconfirmed && <details className="text-xs text-(--ui-text-quaternary)"><summary className="cursor-pointer">{labels.setupDetails}</summary><p className="mt-1 whitespace-pre-wrap break-words">{error}</p></details>}
    </ConfirmDialog>
  </>
}
