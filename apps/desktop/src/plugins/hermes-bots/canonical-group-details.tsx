import { atom, Button, Codicon, Tip } from '@hermes/plugin-sdk'
import { type ReactNode, useLayoutEffect, useRef } from 'react'

import { canonicalMemberName } from './canonical-group-activity'
import { type GroupMethods, offersGroupMethod } from './canonical-group-capabilities'
import type { CanonicalGroupRoute, CanonicalPendingAction, CanonicalRoom } from './canonical-groups'
import type { MemberStatus } from './group-room-status'
import { AvatarStack, fill, memberStatusText, type RoomStatusLabels } from './group-room-ui'

/** What an open room last saw, for its row in the Bots list. */
export interface CanonicalRoomLive { working: boolean; failed: boolean; needsYou: boolean }

export const $canonicalRoomLive = atom<Record<string, CanonicalRoomLive>>({})

export const canonicalRoomLiveKey = ({ connectionId, profile }: CanonicalGroupRoute, roomId: string) =>
  JSON.stringify([connectionId, profile, roomId])

export function publishCanonicalRoomLive(binding: CanonicalGroupRoute & { roomId: string }, live: CanonicalRoomLive): void {
  const key = canonicalRoomLiveKey(binding, binding.roomId)
  const current = $canonicalRoomLive.get()[key]

  if (current?.working !== live.working || current?.failed !== live.failed || current?.needsYou !== live.needsYou) {
    $canonicalRoomLive.set({ ...$canonicalRoomLive.get(), [key]: live })
  }
}

const liveHolders = new Map<string, number>()

/** Claim a room's Bots-list state for one open workspace; the returned release drops it once the room's
 *  last workspace closes, since nothing polls a closed room and its glyph would never update again. */
export function holdCanonicalRoomLive(binding: CanonicalGroupRoute & { roomId: string }): () => void {
  const key = canonicalRoomLiveKey(binding, binding.roomId)
  liveHolders.set(key, (liveHolders.get(key) ?? 0) + 1)

  return () => {
    const left = (liveHolders.get(key) ?? 1) - 1

    if (left > 0) {
      liveHolders.set(key, left)

      return
    }

    liveHolders.delete(key)

    if (key in $canonicalRoomLive.get()) {
      const open = { ...$canonicalRoomLive.get() }
      delete open[key]
      $canonicalRoomLive.set(open)
    }
  }
}

/** Keep a log scrolled to its newest entry unless the reader scrolled up. Follows the content's size, so
 *  rows that add no log event (approval cards, late image previews) stay in view too. */
export function useStickToBottom() {
  const ref = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  useLayoutEffect(() => {
    const el = ref.current
    const content = el?.firstElementChild

    if (!el || !content) {return}

    const onScroll = () => { pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48 }

    const follow = () => { if (pinned.current) {el.scrollTop = el.scrollHeight} }
    const observer = new ResizeObserver(follow)
    el.addEventListener('scroll', onScroll)
    observer.observe(content)
    follow()

    return () => { el.removeEventListener('scroll', onScroll); observer.disconnect() }
  }, [])

  return ref
}

interface PendingLabels {
  allowOnce: string
  deny: string
  discard: string
  retry: string
  pendingApproval: string
  pendingRetry: string
  pendingDiscard: string
}

const PENDING_COPY: Record<string, keyof PendingLabels> = {
  approval: 'pendingApproval',
  retry: 'pendingRetry',
  discard: 'pendingDiscard'
}

const ACTION_METHODS: Record<string, string> = { discard: 'groups.discard', retry: 'groups.retry', approval: 'groups.approve' }

/** A decision the room is waiting on, named by the bot that needs it. */
export function CanonicalPendingCard({ action, avatar, busy, labels, methods, name, onAct, onDiscard }: {
  action: CanonicalPendingAction; avatar: ReactNode; busy: boolean; labels: PendingLabels; methods: GroupMethods; name: string
  onAct: (action: CanonicalPendingAction, choice?: 'deny' | 'once') => unknown; onDiscard: () => void
}) {
  const copy = PENDING_COPY[action.kind]
  const offered = Object.hasOwn(ACTION_METHODS, action.kind) && offersGroupMethod(methods, ACTION_METHODS[action.kind])

  return <div className="mx-3 my-1 flex items-start gap-2.5 rounded-xl bg-(--chrome-action-hover) px-3 py-2" data-pending-action={action.kind}>
    <div className="mt-0.5 shrink-0">{avatar}</div>
    <div className="grid min-w-0 flex-1 gap-1.5">
      <p className="text-xs text-foreground">{copy ? fill(labels[copy], { name }) : name}</p>
      <div className="flex flex-wrap gap-1.5">
        {offered && action.kind === 'discard' && <Button disabled={busy} onClick={onDiscard} size="sm" variant="secondary">{labels.discard}</Button>}
        {offered && action.kind === 'retry' && <Button disabled={busy} onClick={() => void onAct({ ...action })} size="sm" variant="secondary">{labels.retry}</Button>}
        {offered && action.kind === 'approval' && <>
          <Button disabled={busy} onClick={() => void onAct({ ...action }, 'once')} size="sm">{labels.allowOnce}</Button>
          <Button disabled={busy} onClick={() => void onAct({ ...action }, 'deny')} size="sm" variant="secondary">{labels.deny}</Button>
        </>}
      </div>
    </div>
  </div>
}

interface DetailsLabels extends RoomStatusLabels {
  close: string
  groupDetails: string
  members: string
  membersFixed: string
}

/** The side panel the room title opens: picture (member faces), name, and every member's
 *  live status with its full detail. */
export function CanonicalGroupDetails({ avatarFor, children, labels, onClose, room, statuses }: {
  avatarFor: (name: string, size: number) => ReactNode; labels: DetailsLabels; onClose: () => void; room: CanonicalRoom
  statuses: Map<string, MemberStatus>
  /** Room settings (purpose, notes, lead) mount here. */
  children?: ReactNode
}) {
  const faceName = (member: CanonicalRoom['members'][number]) => member.profile || member.handle || member.member_id

  return <aside aria-label={labels.groupDetails} className="flex w-72 shrink-0 flex-col gap-4 overflow-y-auto border-l border-(--ui-stroke-tertiary) px-4 py-3">
    <div className="flex items-center justify-between">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">{labels.groupDetails}</span>
      <Button aria-label={labels.close} onClick={onClose} size="icon-xs" variant="ghost"><Codicon name="close" /></Button>
    </div>
    <div className="flex flex-col items-center gap-2 text-center">
      <AvatarStack avatars={room.members.map(member => ({ key: member.member_id, node: avatarFor(faceName(member), 40) }))} max={4} />
      <span className="text-sm font-semibold text-foreground">{room.name}</span>
    </div>
    <div className="grid gap-1">
      <span className="text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-tertiary)">{labels.members} · {room.members.length}</span>
      {room.members.map(member => {
        const status = statuses.get(member.member_id) ?? { kind: 'idle' as const }
        const detail = 'detail' in status ? status.detail : undefined

        return <div className="flex items-center gap-2 py-1" data-member-status={status.kind} key={member.member_id}>
          {avatarFor(faceName(member), 24)}
          <div className="min-w-0">
            <div className="truncate text-xs font-medium text-foreground">{canonicalMemberName(member)} <span className="font-normal text-(--ui-text-quaternary)">@{member.handle}</span></div>
            <Tip label={detail}><div className="truncate text-[0.6875rem] text-(--ui-text-tertiary)">{memberStatusText(status, labels)}</div></Tip>
          </div>
        </div>
      })}
      <p className="text-[0.6875rem] text-(--ui-text-quaternary)">{labels.membersFixed}</p>
    </div>
    {children}
  </aside>
}
