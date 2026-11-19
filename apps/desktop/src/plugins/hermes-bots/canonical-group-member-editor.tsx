import { Button, Checkbox, Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, randomUuid, SearchField, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { type GroupMethods, groupSeatableProfiles } from './canonical-group-capabilities'
import { type CanonicalGroupBinding, canonicalGroupRequest, canonicalLocalGroupMember, type CanonicalRoom, onCanonicalConnection } from './canonical-groups'
import { $botMeta, $lastRoster } from './data'
import { GROUP_CHAT_MAX_MEMBERS } from './group-chat'
import { RoomBotAvatar } from './group-room-avatar'
import { useBots } from './i18n'
import { displayName } from './labels'
import { botRosterMeta } from './routing'

interface EditorProps {
  binding: CanonicalGroupBinding
  onClose: () => void
  onSaved: () => Promise<void>
  returnFocus?: HTMLElement | null
  room: CanonicalRoom
}

export function useCanonicalGroupMemberEditor({ binding, methods, onSaved, room, visible }: {
  binding: CanonicalGroupBinding
  methods: GroupMethods
  onSaved: EditorProps['onSaved']
  room?: CanonicalRoom
  visible: boolean
}) {
  const [draft, setDraft] = useState<CanonicalRoom | null>(null)
  const triggerRef = useRef<HTMLElement | null>(null)

  const openMembers = methods?.has('groups.members.update') && room?.revision !== undefined
    ? (trigger: HTMLElement) => {
      triggerRef.current = trigger
      setDraft(room)
    } : undefined

  const memberEditor = draft && visible
    ? <CanonicalGroupMemberEditor binding={binding} onClose={() => setDraft(null)}
      onSaved={onSaved} returnFocus={triggerRef.current} room={draft} /> : null

  return { memberEditor, openMembers }
}

export function CanonicalGroupMemberEditor({ binding, onClose, onSaved, returnFocus, room }: EditorProps) {
  const { t } = useI18n()
  const { canonical, group } = useBots()
  const [bots] = useState(() => $lastRoster.get().filter(bot => !bot.ghost && onCanonicalConnection(binding, bot)))
  const [meta] = useState(() => $botMeta.get())
  const [seatable, setSeatable] = useState<Set<string> | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(() => new Set(room.members.map(member => member.member_id)))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const intent = useRef<{ key: string; id: string } | null>(null)

  useEffect(() => {
    let open = true
    void canonicalGroupRequest(binding, 'groups.capabilities').then(
      result => {
        if (!open) {return}
        const profiles = groupSeatableProfiles(result)
        setSeatable(profiles)
        setLoadFailed(profiles === null)
      },
      () => { if (open) {setLoadFailed(true)} }
    )

    return () => {open = false}
  }, [binding, attempt])

  // Keep saved seats and their exact wire descriptors independently of roster availability.
  const candidates = new Map(room.members.map(member => [member.member_id, {
    member, label: member.display_name || displayName({ name: member.profile })
  }]))

  const seatedLocally = new Map(room.members.filter(member => member.target?.kind !== 'peer').map(member => [member.profile, member.member_id]))

  for (const bot of bots) {
    const entry = canonicalLocalGroupMember(bot)
    const seatedId = seatedLocally.get(entry.profile)
    const label = displayName(bot, botRosterMeta(bot, meta))

    if (seatedId !== undefined) {
      candidates.get(seatedId)!.label = label
    } else if (seatable?.has(entry.profile) && !candidates.has(entry.member_id)) {
      candidates.set(entry.member_id, { member: entry, label })
    }
  }

  const matches = query.trim().toLowerCase()

  const visible = [...candidates.values()].filter(({ label, member }) =>
    label.toLowerCase().includes(matches) || `@${member.handle.toLowerCase()}`.includes(matches))

  const members = [...candidates.values()].map(({ member }) => member).filter(member => selected.has(member.member_id))
  const changed = members.length !== room.members.length || members.some((member, index) => member !== room.members[index])

  async function save() {
    if (saving || members.length < 2 || !changed) {return}
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
    <DialogContent className="max-w-md" onCloseAutoFocus={event => {
      if (!returnFocus?.isConnected) {return}
      event.preventDefault()
      returnFocus.focus()
    }}>
      <DialogHeader>
        <DialogTitle>{group.manageMembers}</DialogTitle>
        <DialogDescription>{group.manageMembersDesc(GROUP_CHAT_MAX_MEMBERS, room.name)}</DialogDescription>
      </DialogHeader>
      <SearchField aria-label={group.searchToAdd} containerClassName="w-full" inputClassName="w-full"
        onChange={setQuery} placeholder={group.searchToAddPlaceholder} value={query} />
      <div className="grid max-h-[min(18rem,42dvh)] gap-0.5 overflow-y-auto overscroll-contain">
        {visible.map(({ label, member }) => {
          const checked = selected.has(member.member_id)
          const remote = member.target?.kind === 'peer'

          return <label className="flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5 hover:bg-(--chrome-action-hover)" key={member.member_id}>
            <RoomBotAvatar name={remote ? member.handle : member.profile} size={24} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-xs font-medium text-foreground">{label}</span>
              <span className="block truncate text-xs text-(--ui-text-tertiary)">@{member.handle}</span>
              {remote && <span className="block text-xs text-(--ui-text-secondary)">{group.remoteMemberLocked}</span>}
            </span>
            <Checkbox aria-label={`${label} (@${member.handle})`} checked={checked}
              disabled={saving || remote || (!checked && members.length >= GROUP_CHAT_MAX_MEMBERS)}
              onCheckedChange={value => setSelected(current => {
                const next = new Set(current)

                if (value === true) {next.add(member.member_id)} else {next.delete(member.member_id)}

                return next
              })} />
          </label>
        })}
        {!visible.length && seatable !== null && <p className="px-2 py-3 text-center text-xs text-(--ui-text-secondary)">{group.noBotsMatch(query.trim())}</p>}
      </div>
      {loadFailed ? <div className="flex items-center gap-2">
        <p className="text-xs text-(--ui-text-secondary)" role="status">{canonical.driverUnavailable}</p>
        <Button onClick={() => { setLoadFailed(false); setAttempt(value => value + 1) }} size="xs" variant="secondary">{t.common.retry}</Button>
      </div> : seatable === null && <p className="text-xs text-(--ui-text-secondary)" role="status">{t.common.loading}</p>}
      {error && <p className="text-sm text-destructive" role="alert">{error}</p>}
      <DialogFooter className="items-center">
        <p aria-live="polite" className="me-auto text-xs text-(--ui-text-secondary)">
          {members.length < 2 ? group.memberSizeError(GROUP_CHAT_MAX_MEMBERS) : group.memberCount(members.length)}
        </p>
        <Button disabled={saving} onClick={onClose} variant="secondary">{t.common.cancel}</Button>
        <Button disabled={saving || members.length < 2 || !changed} onClick={() => void save()}>{saving ? t.common.saving : group.saveMembers}</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
}
