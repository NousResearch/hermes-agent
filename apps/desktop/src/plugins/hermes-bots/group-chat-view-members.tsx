/**
 * Membership editing for an EXISTING group room (#91329, #110736): the
 * room-side "Manage members" picker and the save it performs. Split out of
 * group-chat-view.tsx (the room surface facade) as a topical sibling.
 */

import {
  Button,
  cn,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  host,
  RowButton,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useState } from 'react'

import { $botMeta, $lastRoster, botHandle, botRosterKey, saveBotMeta } from './data'
import { GROUP_CHAT_MAX_MEMBERS, updateGroupChat } from './group-chat'
import { botGroups, durableGroupChatMembers, groupMembershipPatch } from './group-membership'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { BotMeta, RosterRow } from './types'

/** Persist the room-side member picker result.  Stored descriptors are the
 * source of truth for remote Bots; local metadata remains a compatibility
 * projection so existing roster surfaces continue to find the room. */
export async function setGroupChatMembers(group: string, selected: RosterRow[]) {
  if (selected.length < 2 || selected.length > GROUP_CHAT_MAX_MEMBERS) {
    throw new Error(`A group chat needs between 2 and ${GROUP_CHAT_MAX_MEMBERS} bots`)
  }

  // A remote and local Bot may share a profile name.  Membership selection is
  // source-qualified everywhere else, so the room picker must use the same
  // identity rather than accidentally seating its local namesake too.
  const selectedKeys = new Set(selected.map(botRosterKey))
  const meta = $botMeta.get()

  for (const bot of $lastRoster.get().filter(bot => !bot.remoteSource)) {
    const enabled = selectedKeys.has(botRosterKey(bot))
    const current = botGroups(botRosterMeta(bot, meta))

    if (current.includes(group) !== enabled) {
      await saveBotMeta(bot, groupMembershipPatch(botRosterMeta(bot, meta), group, enabled))
    }
  }

  updateGroupChat(group, room => ({ ...room, members: durableGroupChatMembers(selected) }))
}

interface GroupMemberPickerProps {
  group: string
  members: RosterRow[]
  onClose: () => void
  open: boolean
}

/** Room-side membership editor (#91329, #110736): the New Group Chat picker's
 *  checklist, pre-checked with the room's current seats. Nothing is written
 *  until Save; Cancel leaves membership untouched. */
export function GroupMemberPicker({ group, members, open, onClose }: GroupMemberPickerProps) {
  const allMeta: Record<string, BotMeta> = useValue($botMeta)
  const liveRoster: RosterRow[] = useValue($lastRoster)

  // Current members always stay listed, even when their source is offline
  // (ghost rows) — otherwise an offline remote member would silently drop
  // out of the room on the next Save.
  const roster = useMemo(() => {
    const rows = liveRoster.filter(bot => !bot.ghost)
    const seen = new Set(rows.map(botRosterKey))

    return [...rows, ...members.filter(member => !seen.has(botRosterKey(member)))]
  }, [liveRoster, members])

  const [selected, setSelected] = useState(() => new Set(members.map(botRosterKey)))
  const atCap = selected.size >= GROUP_CHAT_MAX_MEMBERS

  // Re-seed from the room only when the dialog opens: `members` is rebuilt on
  // every roster poll, so keying on it would wipe the user's picks mid-edit.
  useEffect(() => {
    if (open) {
      setSelected(new Set(members.map(botRosterKey)))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const save = async () => {
    try {
      await setGroupChatMembers(
        group,
        roster.filter(bot => selected.has(botRosterKey(bot)))
      )
      onClose()
    } catch (error) {
      host.notify({
        kind: 'error',
        message: String(error instanceof Error ? error.message : error)
      })
    }
  }

  return (
    <Dialog onOpenChange={value => !value && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Manage members</DialogTitle>
          <DialogDescription>{`Pick 2–${GROUP_CHAT_MAX_MEMBERS} bots for “${group}”. The room, its history and its member sessions stay as they are.`}</DialogDescription>
        </DialogHeader>
        <div className="grid max-h-80 gap-0.5 overflow-y-auto" data-testid="group-member-picker">
          {roster.map(bot => {
            const key = botRosterKey(bot)
            const checked = selected.has(key)
            const disabled = !checked && atCap
            const meta = botRosterMeta(bot, allMeta)

            return (
              <RowButton
                aria-checked={checked}
                className={cn(
                  'flex items-center gap-2 rounded-md px-1.5 py-1 hover:bg-(--chrome-action-hover)',
                  disabled && 'cursor-not-allowed opacity-50'
                )}
                disabled={disabled}
                key={key}
                onClick={() =>
                  setSelected(current => {
                    const next = new Set(current)

                    if (checked) {
                      next.delete(key)
                    } else {
                      next.add(key)
                    }

                    return next
                  })
                }
                role="checkbox"
              >
                <Codicon className={cn(checked ? 'text-(--ui-accent)' : 'text-(--ui-text-quaternary)')} name={checked ? 'pass-filled' : 'circle-large-outline'} />
                <div className="min-w-0 flex-1 text-left">
                  <div className="truncate text-xs text-foreground">{displayName(bot, meta)}</div>
                  <div className="truncate text-[0.625rem] text-(--ui-text-quaternary)">
                    {`@${botHandle(bot.name, bot)}${bot.remoteSource && bot.connectionLabel ? ` · ${bot.connectionLabel}` : ''}`}
                  </div>
                </div>
              </RowButton>
            )
          })}
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="secondary">
            Cancel
          </Button>
          <Button disabled={selected.size < 2 || selected.size > GROUP_CHAT_MAX_MEMBERS} onClick={() => void save()}>
            Save members
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

