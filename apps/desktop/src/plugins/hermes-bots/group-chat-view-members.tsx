/**
 * Membership editing for an EXISTING group room (#91329, #110736): the
 * room-side "Manage members" picker, the per-Bot "Manage groups" toggle, and
 * the one write both doors share. The room reads membership from two places —
 * each local Bot's `groups[]` metadata and the room record's durable member
 * descriptors — so every edit rewrites both, or the removed Bot keeps its seat
 * through whichever source the edit skipped.
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
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useState } from 'react'

import {
  botOnConnection,
  connectionName,
  groupNeedsDesktopCourier,
  seatableGroupBots,
  useGroupCapability
} from './canonical-group-capabilities'
import {
  $botMeta,
  $lastRoster,
  botHandle,
  botRosterKey,
  botSourceStatus,
  readBotMetaFromServer,
  saveBotMeta,
  setLocalBotMeta
} from './data'
import { $groupChats, GROUP_CHAT_MAX_MEMBERS, updateGroupChat } from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { stickyGroupEngine } from './group-engine'
import {
  botGroups,
  durableGroupChatMembers,
  groupChatMemberBots,
  groupMemberKey,
  groupMembershipPatch,
  groupSessionMemberKey
} from './group-membership'
import { harvestStrandedGroupReply } from './group-turns'
import { botsText, useBots } from './i18n'
import { displayName } from './labels'
import { botRosterMeta } from './routing'
import type { BotMeta, RosterRow } from './types'

/** Rewrite the room record for a new seat list. Removed members lose their
 *  per-member room state (sticky hold, stranded marker, thread sessions and
 *  watermarks) so a later re-add starts clean instead of stop-held; a
 *  stranded reply is harvested first so finished work is not lost. The
 *  roster write bumps the room's sync revision: the gateway mirror unions
 *  members on a revision tie, so an unbumped save would let the very next
 *  poll seat the removed Bot again. */
async function commitGroupChatRoster(group: string, previous: RosterRow[], seated: RosterRow[]) {
  const kept = new Set(seated.map(botRosterKey))
  const removed = previous.filter(member => !kept.has(botRosterKey(member)))

  for (const member of removed) {
    await harvestStrandedGroupReply(group, member)
  }

  // Local members are keyed by bare name, stored descriptors by roster key;
  // the same Bot may own state under either, so drop both spellings.
  const gone = new Set(removed.flatMap(member => [groupMemberKey(member), botRosterKey(member), member.name]))

  const without = <T,>(record: Record<string, T> | undefined, memberOf: (key: string) => string) =>
    Object.fromEntries(Object.entries(record || {}).filter(([key]) => !gone.has(memberOf(key))))

  // Watermarks are `<thread>::<memberKey>` (group-round-members); the member
  // key may itself carry a `::`, so split at the first one only.
  const watermarkMember = (key: string) => (key.includes('::') ? key.slice(key.indexOf('::') + 2) : key)
  // A room that spans machines on either side of this edit is Desktop-driven
  // for good: dropping its last remote member must not hand it to a gateway
  // (older records carry no engine field until their first edit).
  const connectionId = host.state.connectionId.get()
  const desktopDriven = groupNeedsDesktopCourier(previous, connectionId) || groupNeedsDesktopCourier(seated, connectionId)

  updateGroupChat(group, (room: GroupChatRoom) => ({
    ...room,
    ...stickyGroupEngine(room, desktopDriven ? { engine: 'desktop' } : undefined),
    members: durableGroupChatMembers(seated),
    heldMessages: without(room.heldMessages, key => key),
    holds: without(room.holds, key => key),
    stranded: without(room.stranded, key => key),
    sessions: without(room.sessions, groupSessionMemberKey),
    sessionOwners: without(room.sessionOwners, groupSessionMemberKey),
    watermarks: without(room.watermarks, watermarkMember),
    syncRevision: Math.max(0, Number(room.syncRevision || 0)) + 1
  }))
}

/** A member on another machine this Desktop cannot reach right now. Its seat
 *  also lives in that machine's own ui_meta, so a removal saved while it is
 *  offline lands only here and the next roster from that machine re-seats it. */
export function memberMachineOffline(bot: RosterRow, connectionId: null | string): boolean {
  return onOtherMachine(bot, connectionId) && (Boolean(bot.ghost) || !botSourceStatus(bot).available)
}

/** A stored member descriptor carries its connection but not `remoteSource`. */
function onOtherMachine(bot: RosterRow, connectionId: null | string): boolean {
  return !botOnConnection(bot, connectionId) && Boolean(bot.remoteSource || bot.connectionId || bot.route?.connectionId)
}

function memberSaveFailed(bot: RosterRow, meta: Record<string, BotMeta>): Error {
  return new Error(botsText().crossGateway.memberSaveFailed(connectionName(bot), displayName(bot, botRosterMeta(bot, meta))))
}

/** A remote row's write that provably did not land on its machine. An older
 *  gateway with no ui_meta contract ('unsupported') holds no seat to restore. */
const REMOTE_WRITE_MISSED = new Set(['failed', 'unreachable'])

/** An undo its machine confirmed, or one with no server copy to undo (older gateway). */
const UNDO_LANDED = new Set(['persisted', 'unsupported'])

interface MembershipWrite {
  bot: RosterRow
  enabled: boolean
}

interface AppliedWrite extends MembershipWrite {
  outcome: string
}

/** The tail of every bot's membership edits, keyed by roster identity
 *  (connection + profile). An edit reads the bot's groups[] and the room's
 *  seats when it STARTS, so two edits running at once would each restore a
 *  list the other already changed. */
const membershipEdits = new Map<string, Promise<unknown>>()

/** Run `edit` once every earlier membership edit touching any of `bots` has settled. */
function queueMembershipEdit<T>(bots: RosterRow[], edit: () => Promise<T>): Promise<T> {
  const keys = [...new Set(bots.map(botRosterKey))]
  const run = Promise.allSettled(keys.map(key => membershipEdits.get(key))).then(edit)

  const settled = run.then(
    () => undefined,
    () => undefined
  )

  for (const key of keys) {
    membershipEdits.set(key, settled)
  }

  void settled.then(() => {
    for (const key of keys) {
      if (membershipEdits.get(key) === settled) {
        membershipEdits.delete(key)
      }
    }
  })

  return run
}

/** Undo each applied write as an inverse delta — drop only the room the write
 *  added, re-add only the room it removed — against the bot's groups[] as its
 *  own backend holds them now, never a snapshot taken before the edit. Returns
 *  the writes whose undo its machine did not confirm. */
async function undoMembershipWrites(group: string, applied: AppliedWrite[]): Promise<AppliedWrite[]> {
  const stuck: AppliedWrite[] = []

  for (const write of [...applied].reverse()) {
    const undo = (meta: BotMeta | null | undefined) => groupMembershipPatch(meta, group, !write.enabled)
    const result = await saveBotMeta(write.bot, undo(botRosterMeta(write.bot, $botMeta.get())), undo)

    // A write that never reached its machine left nothing there to undo.
    if (write.outcome !== 'unreachable' && !UNDO_LANDED.has(result.serverOutcome)) {
      stuck.push(write)
    }
  }

  return stuck
}

/** After undos that did not land, make this window and the room record say what
 *  those machines actually hold: re-read each one, and when it still does not
 *  answer, keep its change (it may well have persisted there). */
async function reconcileStuckWrites(group: string, previous: RosterRow[], stuck: AppliedWrite[]) {
  let seats = previous

  for (const write of stuck) {
    const server = await readBotMetaFromServer(write.bot)
    const groups = server ? botGroups(server) : botGroups(groupMembershipPatch(botRosterMeta(write.bot, $botMeta.get()), group, write.enabled))
    await setLocalBotMeta(write.bot, { group: groups[0] || null, groups })

    const key = botRosterKey(write.bot)
    seats = seats.filter(member => botRosterKey(member) !== key)

    if (groups.includes(group)) {
      seats = [...seats, write.bot]
    }
  }

  if ($groupChats.get()[group]) {
    await commitGroupChatRoster(group, previous, seats)
  }
}

/** Write every changed row's groups[], other machines first. When one of them
 *  did not take the change, undo every row already written (and that one's
 *  local copy) and fail. An undo that did not land is reported as such, with
 *  the room left showing what that machine holds. */
async function writeMembershipChanges(group: string, writes: MembershipWrite[], previous: RosterRow[]) {
  const connectionId = host.state.connectionId.get()
  const remote = (write: MembershipWrite) => onOtherMachine(write.bot, connectionId)
  const ordered = [...writes.filter(remote), ...writes.filter(write => !remote(write))]
  const applied: AppliedWrite[] = []

  for (const write of ordered) {
    const patch = groupMembershipPatch(botRosterMeta(write.bot, $botMeta.get()), group, write.enabled)
    const { serverOutcome } = await saveBotMeta(write.bot, patch)
    applied.push({ ...write, outcome: serverOutcome })

    if (remote(write) && REMOTE_WRITE_MISSED.has(serverOutcome)) {
      const stuck = await undoMembershipWrites(group, applied)
      const meta = $botMeta.get()

      if (stuck.length) {
        await reconcileStuckWrites(group, previous, stuck)
        const held = stuck[0].bot

        throw new Error(
          botsText().crossGateway.memberRollbackFailed(connectionName(held), displayName(held, botRosterMeta(held, meta)))
        )
      }

      throw memberSaveFailed(write.bot, meta)
    }
  }
}

/** Persist the room-side member picker result: each Bot's `groups[]` metadata
 * (on its owning connection) and the room's stored descriptors together. Fails
 * when a removed member's machine is offline or a remote write does not land. */
export function setGroupChatMembers(group: string, selected: RosterRow[]) {
  if (selected.length < 2 || selected.length > GROUP_CHAT_MAX_MEMBERS) {
    return Promise.reject(new Error(botsText().group.memberSizeError(GROUP_CHAT_MAX_MEMBERS)))
  }

  return queueMembershipEdit([...$lastRoster.get(), ...selected], () => saveGroupChatMembers(group, selected))
}

async function saveGroupChatMembers(group: string, selected: RosterRow[]) {
  // A remote and local Bot may share a profile name.  Membership selection is
  // source-qualified everywhere else, so the room picker must use the same
  // identity rather than accidentally seating its local namesake too.
  const selectedKeys = new Set(selected.map(botRosterKey))
  const meta = $botMeta.get()
  const previous = groupChatMemberBots(group, $lastRoster.get(), meta)

  // Every row, remote ones included: member resolution seats any bot whose
  // groups[] names the room, so a remote member whose list still names it would
  // take its seat back. saveBotMeta writes a remote row through its own
  // connection, as the per-Bot door does.
  const writes: MembershipWrite[] = []

  for (const bot of $lastRoster.get()) {
    const enabled = selectedKeys.has(botRosterKey(bot))

    if (botGroups(botRosterMeta(bot, meta)).includes(group) !== enabled) {
      writes.push({ bot, enabled })
    }
  }

  const connectionId = host.state.connectionId.get()
  const stranded = writes.find(write => !write.enabled && memberMachineOffline(write.bot, connectionId))

  if (stranded) {
    throw memberSaveFailed(stranded.bot, meta)
  }

  await writeMembershipChanges(group, writes, previous)
  await commitGroupChatRoster(group, previous, selected)
}

/** The per-Bot door ("Manage groups"): toggle ONE Bot's seat in a room. Writes
 *  the Bot's `groups[]` metadata and the room's stored descriptors together —
 *  unchecking a Bot here used to leave its stored descriptor behind, so the
 *  room still gave it a turn (#91329). Like Manage members, it fails when the
 *  Bot's machine is offline or the write misses it. Edits of one Bot run one
 *  at a time, so a fast second toggle never races the first. */
export function setGroupMembership(bot: RosterRow, group: string, enabled: boolean) {
  return queueMembershipEdit([bot], () => saveGroupMembership(bot, group, enabled))
}

async function saveGroupMembership(bot: RosterRow, group: string, enabled: boolean) {
  const meta = $botMeta.get()
  const previous = groupChatMemberBots(group, $lastRoster.get(), meta)

  if (memberMachineOffline(bot, host.state.connectionId.get())) {
    throw memberSaveFailed(bot, meta)
  }

  await writeMembershipChanges(group, [{ bot, enabled }], previous)

  if (!$groupChats.get()[group]) {
    // Metadata-only group (never opened as a room): no stored seats to reconcile.
    return
  }

  const key = botRosterKey(bot)
  const others = previous.filter(member => botRosterKey(member) !== key)

  await commitGroupChatRoster(group, previous, enabled ? [...others, bot] : others)
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
  const b = useBots()
  const { t } = useI18n()
  const allMeta: Record<string, BotMeta> = useValue($botMeta)
  const liveRoster: RosterRow[] = useValue($lastRoster)

  const { connectionId, mode, seatable } = useGroupCapability(open)

  // Add candidates are the rows New Group Chat would seat (the same predicate,
  // so a row whose route disagrees with its connection is never offered).
  // Current members always stay listed, even when their source is offline
  // (ghost rows) — otherwise an offline remote member would silently drop
  // out of the room on the next Save.
  const roster = useMemo(() => {
    const rows = seatableGroupBots(liveRoster, { connectionId, mode, seatable })
    const seen = new Set(rows.map(botRosterKey))

    return [...rows, ...members.filter(member => !seen.has(botRosterKey(member)))]
  }, [connectionId, liveRoster, members, mode, seatable])

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
          <DialogTitle>{b.group.manageMembers}</DialogTitle>
          <DialogDescription>{b.group.manageMembersDesc(GROUP_CHAT_MAX_MEMBERS, group)}</DialogDescription>
        </DialogHeader>
        <div className="grid max-h-80 gap-0.5 overflow-y-auto" data-testid="group-member-picker">
          {roster.map(bot => {
            const key = botRosterKey(bot)
            const checked = selected.has(key)
            const meta = botRosterMeta(bot, allMeta)
            // A seated member whose machine is offline cannot be removed from
            // here: the removal could not reach that machine (see memberMachineOffline).
            const locked = checked && memberMachineOffline(bot, connectionId)
            const disabled = locked || (!checked && atCap)

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
                <Codicon
                  className={cn(checked ? 'text-(--ui-accent)' : 'text-(--ui-text-quaternary)')}
                  name={checked ? 'pass-filled' : 'circle-large-outline'}
                />
                <div className="min-w-0 flex-1 text-left">
                  <div className="truncate text-xs text-foreground">{displayName(bot, meta)}</div>
                  <div className="truncate text-[0.625rem] text-(--ui-text-quaternary)">
                    {`@${botHandle(bot.name, bot)}${bot.remoteSource && bot.connectionLabel ? ` · ${bot.connectionLabel}` : ''}`}
                  </div>
                  {locked && (
                    <div className="text-[0.625rem] text-(--ui-text-tertiary)" data-testid="group-member-locked">
                      {b.crossGateway.memberOffline(connectionName(bot), displayName(bot, meta))}
                    </div>
                  )}
                </div>
              </RowButton>
            )
          })}
        </div>
        <DialogFooter>
          <Button onClick={onClose} variant="secondary">
            {t.common.cancel}
          </Button>
          <Button disabled={selected.size < 2 || selected.size > GROUP_CHAT_MAX_MEMBERS} onClick={() => void save()}>
            {b.group.saveMembers}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
