import { Button, Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, randomUuid, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { groupSeatableProfiles } from './canonical-group-capabilities'
import { type CanonicalGroupBinding, canonicalGroupRequest, canonicalLocalGroupMember, type CanonicalRoom, type CanonicalRoomMember, onCanonicalConnection } from './canonical-groups'
import { $lastRoster } from './data'
import { useBots } from './i18n'

interface EditorProps {
  binding: CanonicalGroupBinding
  onSaved: () => Promise<void>
  room: CanonicalRoom
}

export function CanonicalGroupMemberEditor({ binding, onSaved, room }: EditorProps) {
  const { group } = useBots()
  const [draft, setDraft] = useState<CanonicalRoom | null>(null)

  return <>
    <Button onClick={() => setDraft(room)} size="sm" variant="secondary">{group.manageMembers}</Button>
    {draft && <MemberPicker binding={binding} onClose={() => setDraft(null)} onSaved={onSaved} room={draft} />}
  </>
}

function MemberPicker({ binding, onClose, onSaved, room }: EditorProps & { onClose: () => void }) {
  const { t } = useI18n()
  const { group } = useBots()
  const [bots] = useState(() => $lastRoster.get())
  const [seatable, setSeatable] = useState<Set<string> | null>(null)
  const [selected, setSelected] = useState(() => new Set(room.members.map(member => member.member_id)))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const intent = useRef<{ key: string; id: string } | null>(null)

  useEffect(() => {
    let open = true
    void canonicalGroupRequest(binding, 'groups.capabilities').then(
      result => { if (open) {setSeatable(groupSeatableProfiles(result))} },
      failure => { if (open) {setError(String(failure instanceof Error ? failure.message : failure))} }
    )

    return () => {open = false}
  }, [binding])

  // Include the room's saved seats even if their profiles are absent from a roster poll.
  const candidates = new Map<string, CanonicalRoomMember>(room.members.map(member => [member.member_id, member]))
  const seatedLocally = new Set(room.members.filter(member => member.target?.kind !== 'peer').map(member => member.profile))

  for (const bot of bots.filter(bot => !bot.ghost && onCanonicalConnection(binding, bot))) {
    const entry = canonicalLocalGroupMember(bot)

    if (seatable?.has(entry.profile) && !seatedLocally.has(entry.profile) && !candidates.has(entry.member_id)) {
      candidates.set(entry.member_id, entry)
    }
  }

  const members = [...candidates.values()].filter(member => selected.has(member.member_id))
  const changed = members.length !== room.members.length || members.some((member, index) => member !== room.members[index])

  async function save() {
    setError('')
    setSaving(true)
    const key = JSON.stringify(members)

    if (intent.current?.key !== key) {intent.current = { key, id: randomUuid() }}

    try {
      await canonicalGroupRequest(binding, 'groups.members.update', {
        room_id: binding.roomId, event_id: intent.current.id, expected_revision: room.revision, members
      })
    } catch (failure) {
      setError(String(failure instanceof Error ? failure.message : failure))
      setSaving(false)

      return
    }

    onClose()
    void onSaved().catch(() => undefined)
  }

  return <Dialog onOpenChange={open => { if (!open && !saving) {onClose()} }} open>
    <DialogContent className="max-w-md">
      <DialogHeader>
        <DialogTitle>{group.manageMembers}</DialogTitle>
        <DialogDescription>{group.manageMembersDesc(6, room.name)}</DialogDescription>
      </DialogHeader>
      <div className="grid max-h-80 gap-1 overflow-y-auto">
        {[...candidates.values()].map(member => {
          const checked = selected.has(member.member_id)

          return <label className="flex items-center gap-2 rounded-md px-2 py-1 text-sm" key={member.member_id}>
            <input checked={checked} disabled={saving || member.target?.kind === 'peer' || (!checked && members.length >= 6)}
              onChange={() => setSelected(current => {
                const next = new Set(current)

                if (checked) {next.delete(member.member_id)} else {next.add(member.member_id)}

                return next
              })} type="checkbox" />
            <span>{member.display_name || member.handle} <span className="text-(--ui-text-tertiary)">@{member.handle}</span></span>
          </label>
        })}
      </div>
      {error && <p className="text-sm text-destructive" role="alert">{error}</p>}
      <DialogFooter>
        <Button disabled={saving} onClick={onClose} variant="secondary">{t.common.cancel}</Button>
        <Button disabled={saving || members.length < 2 || !changed} onClick={() => void save()}>{group.saveMembers}</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
}
