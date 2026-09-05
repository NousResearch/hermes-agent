/**
 * The group-chat room surface: the merged room view, its settings dialog, the
 * MAIN-window tab it opens into, and the two room-lifecycle mutations that
 * surface drives — disband and rename.
 *
 * Disband and rename live here rather than beside the pane registry because a
 * rename re-opens the room's main tab, and that tab renders this file's
 * workspace: settings dialog → rename → open → main view → workspace is one
 * cycle, so it is one module. The tab registry and the composer drafts they
 * touch stay below, in `group-panes.ts`.
 */

import * as sdk from '@hermes/plugin-sdk'
import {
  atom,
  Button,
  cn,
  Codicon,
  ConfirmDialog,
  CopyButton,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  host,
  Input,
  queryClient,
  relativeTime,
  RowButton,
  Tip,
  useI18n,
  useValue
} from '@hermes/plugin-sdk'
import type { ClipboardEvent, DragEvent, ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { avatarColor, botAppearance, BotFace } from './avatar'
import { isBackfilledFacePng } from './avatar-image'
import {
  $botMeta,
  $lastRoster,
  botHandle,
  botMetaV2Active,
  botSourceStatus,
  noteBotMetaWrite,
  persistBotMetaSnapshot,
  ROSTER_KEY,
  saveBotMeta
} from './data'
import {
  $groupActivity,
  currentGroupActivity,
  GROUP_ACTIVITY_GLYPHS,
  groupActivityLabel,
  groupActivityTone
} from './group-activity'
import type { GroupActivityEntry } from './group-activity'
import { filesToGroupAttachments, pickGroupAttachments } from './group-attachments'
import {
  $groupChats,
  $groupChatWorkspace,
  $groupClarify,
  $groupNeedsYou,
  groupSpeakerLabel,
  groupThreadOf,
  scheduleGroupChatServerSync,
  setGroupChatImage,
  updateGroupChat
} from './group-chat'
import type { GroupChatRoom } from './group-chat'
import { GroupClarifyCard, GroupImageControls, GroupMentionInput } from './group-chat-parts'
import type { GroupRoomPrompt } from './group-chat-parts'
import { GroupHoldStatus } from './group-hold-status'
import {
  botGroups,
  groupChatMemberBots,
  groupDisbandMetadataPlan,
  groupWorkspaceOwnerKey,
  liveGroupChatNames
} from './group-membership'
import {
  clearGroupComposerDraft,
  closeGroupChatMainTab,
  dropGroupMainTab,
  groupChatMainTabs,
  groupComposerDraftKey,
  groupComposerDraftSnapshot,
  migrateGroupComposerDraft,
  recordGroupMainTab,
  restoreGroupComposerDraft,
  updateGroupComposerDraft
} from './group-panes'
import type { GroupComposerDraft, GroupDraftSetter } from './group-panes'
import { sendToGroupChat, stopGroupThread } from './group-rounds'
import { clearGroupClarify } from './group-turns'
import { botsText, useBots } from './i18n'
import { displayName, slugify, stripPreviewMarkdown } from './labels'
import { botRosterMeta, setBotsWorkspaceOwner } from './routing'
import { bumpBotOpenGeneration, getPluginCtx, ID } from './shared'
import type { Attachment, BotMeta, GroupChat, GroupMember, GroupMessage, RosterRow } from './types'

const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown

/** Soft-disband a group chat: remove only this group from every local member's
 *  membership list (the metadata syncs cross-machine via ui_meta), drop the
 *  room log from the atom + plugin storage, and close the room view if it's
 *  open. Other group memberships and the members' per-group gateway sessions
 *  ("Group: <roomId>", or legacy "Group: <name>") are intentionally KEPT. */
export async function disbandGroupChat(group: string, members: RosterRow[]) {
  // Invalidate any in-flight round-robin FIRST: bump the epoch so a running
  // drive bails at its next member boundary instead of appending to a room
  // the user just discarded.
  const all = {
    ...$groupChats.get()
  }

  const prior = all[group] || {}
  const metaBefore = $botMeta.get()
  const cleanup = groupDisbandMetadataPlan(group, members, prior, $lastRoster.get(), metaBefore)
  let metadataPersistence: Promise<unknown> = Promise.resolve()

  if (cleanup.patches.size) {
    const nextMeta = {
      ...metaBefore
    }

    for (const [key, patch] of cleanup.patches) {
      nextMeta[key] = {
        ...(nextMeta[key] || {}),
        ...patch
      }
      noteBotMetaWrite(key)
    }

    // Paint the deletion before any remote write can stall. This also removes
    // orphaned legacy metadata that cannot safely be routed to a source.
    $botMeta.set(nextMeta)
    metadataPersistence = persistBotMetaSnapshot(
      nextMeta,
      botMetaV2Active || Object.keys(nextMeta).some(key => key.includes('::'))
    )
  }

  delete all[group]

  // Keep a runtime-only tombstone while a drive may still be mid-turn; it
  // carries no log and is flagged so persistence and name-dedup skip it —
  // updateGroupChat writes the WHOLE atom map, so an unflagged tombstone
  // would be persisted by the next unrelated room write and its name would
  // count as taken, suffixing a same-name recreate to "<name> 2" forever.
  if (prior.running) {
    all[group] = {
      log: [],
      watermarks: {},
      sessions: {},
      epoch: (prior.epoch || 0) + 1,
      running: false,
      tombstone: true
    }
  }

  $groupChats.set(all)

  if ($groupChatWorkspace.get() === group) {
    $groupChatWorkspace.set(null)
  }

  // Retire the room's MAIN-window tab too (host.openWorkspace path).
  closeGroupChatMainTab(group)

  const needs = {
    ...$groupNeedsYou.get()
  }

  delete needs[group]
  $groupNeedsYou.set(needs)
  clearGroupClarify(group)

  // Persist the room map WITHOUT the disbanded room so it can't come back
  // on the next window load.
  try {
    const durable: Record<string, GroupChat> = {}

    for (const [name, room] of Object.entries($groupChats.get())) {
      if (name !== group && Array.isArray(room.log)) {
        durable[name] = {
          log: room.log,
          watermarks: room.watermarks,
          sessions: room.sessions || {},
          sessionOwners: room.sessionOwners || {},
          members: Array.isArray(room.members) ? room.members : [],
          roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
          image: room.image || null,
          syncRevision: Math.max(0, Number(room.syncRevision || 0))
        }
      }
    }

    await Promise.resolve(getPluginCtx()?.storage?.set?.('group-chats', durable))
  } catch {
    /* storage unavailable — the atom reset above still empties the room */
  }

  scheduleGroupChatServerSync($groupChats.get(), {
    allowEmpty: true,
    deletedRooms: [group]
  })
  await metadataPersistence

  // Persist the cleanup to every exact owner we can prove. saveBotMeta never
  // throws (local storage + best-effort profiles.configure per owner), so a
  // flaky gateway cannot strand the local disband halfway.
  for (const [key, owner] of cleanup.owners) {
    const patch = cleanup.patches.get(key)

    if (patch) {
      await saveBotMeta(owner, patch)
    }
  }

  // Converge on server truth: the cached roster still carries the pre-disband
  // ui_meta (the write-fence in mergeServerMeta keeps it from resurrecting
  // the membership, but a fresh snapshot is what makes every surface agree).
  if (typeof queryClient !== 'undefined' && queryClient?.invalidateQueries) {
    queryClient.invalidateQueries({
      queryKey: ROSTER_KEY
    })
  }
}

/** Rename a group chat. The group's NAME is its identity everywhere — the
 *  room-map key, each local member's ui_meta membership list, and derived
 *  state — so a rename re-keys all of them. Member gateway sessions are kept
 *  as-is: stored sids keep resuming, so no history is lost. The room's
 *  immutable roomId (the member-session title) is preserved across the
 *  rename, so even a member whose sid is later lost falls back to the same
 *  "Group: <roomId>" title lookup instead of a fresh "Group: <new name>".
 *  Returns the new name, or null when the target name is taken. */
async function renameGroupChat(oldName: string, newName: string, members: GroupMember[] | null | undefined) {
  const next = String(newName || '')
    .trim()
    .slice(0, 64)

  if (!next || next === oldName) {
    return oldName
  }

  // Renames are explicit user intent: reject a collision honestly instead of
  // silently suffixing like creation does. Disband tombstones don't hold
  // their name — the room is gone, only its epoch survives briefly.
  const taken = new Set(liveGroupChatNames())

  for (const meta of Object.values($botMeta.get() || {})) {
    for (const existing of botGroups(meta)) {
      taken.add(existing)
    }
  }

  taken.delete(oldName)

  if (taken.has(next)) {
    host.notify({
      kind: 'error',
      message: botsText().group.nameTaken(next)
    })

    return null
  }

  // Move the room record wholesale — log, watermarks, sessions, members,
  // picture, and runtime flags all belong to the same room under its new name.
  const all: Record<string, GroupChat> = {
    ...$groupChats.get()
  }

  const room = all[oldName]

  if (room) {
    migrateGroupComposerDraft(groupComposerDraftKey(oldName, room), groupComposerDraftKey(next, room))
  }

  delete all[oldName]

  if (room) {
    all[next] = room
  }

  $groupChats.set(all)

  const needs: Record<string, boolean> = {
    ...$groupNeedsYou.get()
  }

  if (oldName in needs) {
    needs[next] = needs[oldName]
    delete needs[oldName]
    $groupNeedsYou.set(needs)
  }

  // Mirrored clarify cards key by group name; drop the old room's — the
  // next poll re-mirrors any still-blocking question under the new name.
  clearGroupClarify(oldName)

  // Local memberships: swap the name inside each member's canonical groups
  // list (syncs cross-machine via ui_meta). Remote members' seating lives in
  // the room record we just moved.
  for (const member of members || []) {
    if (!member?.name) {
      continue
    }

    const meta = botRosterMeta(member, $botMeta.get()) || {}
    const groups = [...new Set(botGroups(meta).map(g => (g === oldName ? next : g)))]
    await saveBotMeta(member, {
      groups,
      group: groups[0] || null
    })
  }

  // Persist the re-keyed map (updateGroupChat writes the whole durable map).
  updateGroupChat(next, (r: GroupChatRoom) => r, {
    sync: false
  })
  // A rename is one revisioned state transition: the new identity is updated
  // and the old identity is tombstoned together, so cold hydration cannot
  // merge the pre-rename room back into the roster.
  scheduleGroupChatServerSync($groupChats.get(), {
    changedRooms: [next],
    deletedRooms: [oldName]
  })

  // Follow the open views to the new identity.
  if ($groupChatWorkspace.get() === oldName) {
    $groupChatWorkspace.set(next)
  }

  if (groupChatMainTabs.has(oldName)) {
    closeGroupChatMainTab(oldName)
    openGroupChat(next)
  }

  // Same convergence as disband: drop the pre-rename roster snapshot so the
  // old name can't linger anywhere the fence doesn't cover.
  if (typeof queryClient !== 'undefined' && queryClient?.invalidateQueries) {
    queryClient.invalidateQueries({
      queryKey: ROSTER_KEY
    })
  }

  return next
}

interface GroupChatSettingsDialogProps {
  group: string
  members?: GroupMember[]
  onClose: () => void
  onRenamed?: (group: string) => void
  open: boolean
}

/** Edit an existing group chat's name and picture. Renames re-key the room
 *  and every local member's membership (renameGroupChat); the picture rides
 *  the room record. Both apply on Save so a cancelled dialog changes nothing. */
function GroupChatSettingsDialog({ group, members, open, onClose, onRenamed }: GroupChatSettingsDialogProps) {
  const { t } = useI18n()
  const b = useBots()
  const rooms: Record<string, GroupChatRoom> = useValue($groupChats)
  const current = (rooms[group] || {}).image || null
  const [name, setName] = useState(group)
  const [image, setImage] = useState(current)
  useEffect(() => {
    if (open) {
      setName(group)
      setImage(current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, group])

  const save = async () => {
    const finalName = await renameGroupChat(group, name, members)

    if (finalName === null) {
      return // collision — dialog stays open for a different name
    }

    if (image !== current) {
      setGroupChatImage(finalName, image)
    }

    onClose()

    if (finalName !== group) {
      onRenamed?.(finalName)
    }
  }

  return (
    <Dialog
      onOpenChange={value => {
        if (!value) {
          onClose()
        }
      }}
      open={open}
    >
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{b.group.settingsTitle}</DialogTitle>
          <DialogDescription>{b.group.settingsDesc}</DialogDescription>
        </DialogHeader>
        <GroupImageControls
          image={image}
          onImage={setImage}
          seedMembers={(members || []).map(member => member.name)}
          seedName={name.trim() || group}
        />
        <form
          onSubmit={event => {
            event.preventDefault()
            void save()
          }}
        >
          <Input
            aria-label={b.group.nameLabel}
            autoFocus
            maxLength={64}
            onChange={event => setName(event.target.value)}
            value={name}
          />
        </form>
        <DialogFooter>
          <Button onClick={onClose} variant="secondary">
            {t.common.cancel}
          </Button>
          <Button disabled={!name.trim()} onClick={() => void save()}>
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// ── threads: the Slack/Discord shape ─────────────────────────────────────────
// Every room entry belongs to a THREAD. Messaging the room composer starts a
// new thread with the whole group; replying inside a thread continues that
// work. Member turns are scoped to the thread that triggered them — deltas,
// watermarks, and responder resolution all key on the thread id.

/** One thread's slice of the room log, with each entry's index in that log. */
interface GroupThreadBucket {
  entries: Array<{ entry: GroupMessage; index: number }>
  id: string
  startIndex: number
}

interface GroupChatWorkspaceProps {
  group: string
  members: GroupMember[]
  onBack?: () => void
  visible?: boolean
}

export function GroupChatWorkspace({ group, members, onBack, visible = true }: GroupChatWorkspaceProps) {
  const b = useBots()
  const rooms: Record<string, GroupChatRoom> = useValue($groupChats)
  const allMeta: Record<string, BotMeta> = useValue($botMeta)

  const room: GroupChatRoom = rooms[group] || {
    log: [],
    running: false
  }

  const composerKey = groupComposerDraftKey(group, room)
  const composerKeyRef = useRef(composerKey)
  const [composerDraft, setComposerDraft] = useState(() => groupComposerDraftSnapshot(composerKey))

  if (composerKeyRef.current !== composerKey) {
    migrateGroupComposerDraft(composerKeyRef.current, composerKey)
    composerKeyRef.current = composerKey
  }

  const updateComposerDraft = (mutate: (draft: GroupComposerDraft) => GroupComposerDraft) => {
    const next = updateGroupComposerDraft(composerKeyRef.current, mutate)
    setComposerDraft(next)

    return next
  }

  const draft = composerDraft.main || ''
  const replyDrafts = composerDraft.replies || {}
  const replyThread = composerDraft.activeReplyThread || null
  const pendingImages = composerDraft.pendingAttachments || {}

  const setDraft = (value: GroupDraftSetter<string>) =>
    updateComposerDraft(current => ({
      ...current,
      main: typeof value === 'function' ? value(current.main || '') : value
    }))

  const setReplyDrafts = (value: GroupDraftSetter<Record<string, string>>) =>
    updateComposerDraft(current => ({
      ...current,
      replies: typeof value === 'function' ? value(current.replies || {}) : value
    }))

  const setReplyThread = (value: GroupDraftSetter<null | string>) =>
    updateComposerDraft(current => ({
      ...current,
      activeReplyThread: typeof value === 'function' ? value(current.activeReplyThread || null) : value
    }))

  const setPendingImages = (value: GroupDraftSetter<Record<string, Attachment[]>>) =>
    updateComposerDraft(current => ({
      ...current,
      pendingAttachments: typeof value === 'function' ? value(current.pendingAttachments || {}) : value
    }))

  // Browser Comment Mode can live in its own Electron pop-out on another
  // monitor. The ordinary composer bus is window-local and this room uses its
  // own New Thread composer anyway, so accept only batches explicitly pinned to
  // THIS room + THIS renderer. Packaged Electron windows use the preload IPC
  // relay; BroadcastChannel stays only as a browser/dev fallback. The protocol
  // strings remain literal here to preserve the plugin fence (Bot Mode imports
  // only the SDK + its own files). A stale route gets no ACK, leaving the
  // annotations intact.
  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const desktopBridge = window.hermesDesktop?.previewAnnotate

    const channel =
      !desktopBridge && typeof BroadcastChannel !== 'undefined'
        ? new BroadcastChannel('hermes.desktop.preview-annotate-handoff.v1')
        : null

    if (!desktopBridge && !channel) {
      return
    }

    const send = (payload: unknown) => {
      if (desktopBridge) {
        desktopBridge.send(payload)
      } else {
        channel?.postMessage(payload)
      }
    }

    const onMessage = (data: unknown) => {
      if (!data || typeof data !== 'object') {
        return
      }

      const request = data as {
        count?: unknown
        destination?: { composerKey?: unknown; group?: unknown; kind?: unknown; windowId?: unknown }
        images?: unknown
        prompt?: unknown
        requestId?: unknown
        type?: unknown
      }

      const destination = request.destination
      const sourceWindowId = window.sessionStorage.getItem('hermes.desktop.previewAnnotate.windowId')?.trim()

      if (
        !sourceWindowId ||
        request.type !== 'preview-annotate-handoff' ||
        typeof request.requestId !== 'string' ||
        typeof request.prompt !== 'string' ||
        !Array.isArray(request.images) ||
        destination?.kind !== 'group' ||
        destination.windowId !== sourceWindowId ||
        destination.group !== group ||
        destination.composerKey !== composerKeyRef.current
      ) {
        return
      }

      try {
        const prompt = request.prompt

        const attachments: Attachment[] = request.images
          .filter((image): image is { dataUrl: string; name: string; number?: number } =>
            Boolean(
              image &&
              typeof image === 'object' &&
              typeof (image as { dataUrl?: unknown }).dataUrl === 'string' &&
              typeof (image as { name?: unknown }).name === 'string'
            )
          )
          .map(image => ({ data: image.dataUrl, kind: 'image' as const, name: image.name }))

        const next = updateGroupComposerDraft(composerKeyRef.current, current => ({
          ...current,
          activeReplyThread: null,
          main: current.main.trim() ? `${current.main.trimEnd()}\n\n${prompt}` : prompt,
          pendingAttachments: {
            ...(current.pendingAttachments || {}),
            main: [...(current.pendingAttachments?.main || []), ...attachments]
          }
        }))

        setComposerDraft(next)
        host.notify({
          kind: 'success',
          message: `${Number(request.count) || attachments.length} Browser comment${Number(request.count) === 1 ? '' : 's'} added to ${group}. Review the New Thread draft before sending.`
        })
        send({
          ok: true,
          requestId: request.requestId,
          type: 'preview-annotate-handoff-ack'
        })
      } catch (error) {
        send({
          error: error instanceof Error ? error.message : String(error),
          ok: false,
          requestId: request.requestId,
          type: 'preview-annotate-handoff-ack'
        })
      }
    }

    const stopDesktop = desktopBridge?.onMessage(onMessage)
    const onBroadcast = (event: MessageEvent<unknown>) => onMessage(event.data)
    channel?.addEventListener('message', onBroadcast)

    return () => {
      stopDesktop?.()
      channel?.removeEventListener('message', onBroadcast)
      channel?.close()
    }
  }, [composerKey, group])

  const [confirmDisband, setConfirmDisband] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  // Click-to-disambiguate: which log entry is showing its speaker's full
  // @handle (the roster's name-device form when names collide across
  // connections). Naturally every speaker just shows its display name.
  const [revealedSpeaker, setRevealedSpeaker] = useState<null | string>(null)
  // Threads, the Slack/Discord shape: entries carry a thread id. The most
  // recently active thread renders open; older ones collapse to summary rows.
  // `openThreads` is the user's explicit expand/collapse overrides, and
  // `replyThread` is the thread whose reply box currently owns the composer
  // (null = the main composer, which STARTS a new thread).
  const [openThreads, setOpenThreads] = useState<Record<string, boolean>>({})
  // Pending image attachments per composer: `null` thread key = the main
  // composer, otherwise the reply box of that thread. Data URLs, already
  // downscaled — they ride the send into every responding member's session.

  // Scroll anchoring (#89835): rooms used to open at scroll position 0 and
  // stay there while replies streamed in. Scroll the bottom sentinel into
  // view on mount and whenever the log grows — but only when the user is
  // already near the bottom, so reading history is never yanked away.
  const bottomSentinelRef = useRef<HTMLDivElement | null>(null)
  const stickToBottomRef = useRef(true)
  // eslint-disable-next-line no-restricted-syntax -- tracks live scroll position from a DOM listener, not an atom
  useEffect(() => {
    const sentinel = bottomSentinelRef.current

    if (!sentinel) {
      return
    }

    const viewport = sentinel.closest('[data-slot="scroll-area-viewport"]')

    if (viewport) {
      const onScroll = () => {
        stickToBottomRef.current = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 80
      }

      viewport.addEventListener('scroll', onScroll, {
        passive: true
      })

      return () => viewport.removeEventListener('scroll', onScroll)
    }
  }, [])
  useEffect(() => {
    if (stickToBottomRef.current) {
      bottomSentinelRef.current?.scrollIntoView({
        block: 'end'
      })
    }
  }, [room.log.length, room.running])

  // Retained-pane reopen (#89835 follow-up): a hot-mounted room pane stays
  // mounted while another workspace tab is active, so returning to it never
  // remounts and the mount-time anchor doesn't rerun. Re-anchor on the
  // hidden → visible edge — an explicit reopen, so it overrides a stale
  // read-position and mirrors what a fresh open does.
  const wasVisibleRef = useRef(visible)
  // eslint-disable-next-line no-restricted-syntax -- previous-value tracker for the hidden → visible edge; lagging a render IS the contract
  useEffect(() => {
    if (visible && !wasVisibleRef.current) {
      stickToBottomRef.current = true
      bottomSentinelRef.current?.scrollIntoView({
        block: 'end'
      })
    }

    wasVisibleRef.current = visible
  }, [visible])
  const imagesFor = (thread: null | string) => pendingImages[thread ?? 'main'] || []

  const addImages = (thread: null | string, picked: Attachment[]) => {
    if (!picked.length) {
      return
    }

    const key = thread ?? 'main'
    setPendingImages(prev => ({
      ...prev,
      [key]: [...(prev[key] || []), ...picked]
    }))
  }

  const removeImage = (thread: null | string, index: number) => {
    const key = thread ?? 'main'
    setPendingImages(prev => ({
      ...prev,
      [key]: (prev[key] || []).filter((_, i) => i !== index)
    }))
  }

  // Ctrl/⌘-V a screenshot (or any file) into any composer in this room.
  const pasteImages = (thread: null | string, event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = [...(event.clipboardData?.files || [])]

    if (!files.length) {
      return
    }

    event.preventDefault()
    void filesToGroupAttachments(files).then(picked => addImages(thread, picked))
  }

  // Drag & drop anywhere on the room drops into the ACTIVE composer — the
  // open reply box when one owns the composer, else the main (new-thread)
  // composer. Matches the 1:1 chat's drop affordance.
  const [dragOver, setDragOver] = useState(false)

  const dropFiles = (event: DragEvent<HTMLDivElement>) => {
    const files = [...(event.dataTransfer?.files || [])]
    setDragOver(false)

    if (!files.length) {
      return
    }

    event.preventDefault()
    void filesToGroupAttachments(files).then(picked => addImages(replyThread, picked))
  }

  // Collapsible Activity view: collapsed by default — opening it is always an
  // explicit user action, it never steals focus, and it never auto-scrolls.
  const [activityOpen, setActivityOpen] = useState(false)
  // Subscribe: activity rows re-render as turn events land.
  useValue($groupActivity)
  // Pending member questions for THIS room (#90694), oldest first.
  const clarifyAll: Record<string, GroupRoomPrompt> = useValue($groupClarify)

  const roomClarifies = Object.values(clarifyAll || {})
    .filter(entry => entry?.group === group)
    .sort((a, b) => (a.at || 0) - (b.at || 0))

  const availableMembers = members.filter(member => botSourceStatus(member).available).length
  const availabilityLabel = `${availableMembers} of ${members.length} available`

  const memberNames =
    members.map(b => displayName(b, botRosterMeta(b, allMeta))).join(', ') || 'No bots in this group chat'

  const header = (
    <div className="flex items-center gap-2 px-2.5 pt-2.5 pb-2">
      <Button onClick={() => (onBack ? onBack() : $groupChatWorkspace.set(null))} size="sm" variant="ghost">
        Back
      </Button>
      {/* Room picture (set via Group settings) leads the title when present. */}
      {room.image ? (
        <img
          alt=""
          className="size-6 shrink-0 rounded-md object-cover ring-1 ring-(--ui-stroke-secondary)"
          src={room.image}
        />
      ) : (
        <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-(--chrome-action-hover) text-(--ui-text-tertiary)">
          <Codicon name="organization" />
        </span>
      )}
      <div className="min-w-0 flex-1 truncate text-sm font-semibold">{group}</div>
      <Tip label={memberNames}>
        <span
          aria-label={availabilityLabel}
          className={cn(
            'shrink-0 text-[0.65rem] text-(--ui-text-quaternary)',
            members.length > 0 && availableMembers < members.length && 'text-amber-600 dark:text-amber-300'
          )}
        >
          {members.length > 0 && availableMembers < members.length
            ? availabilityLabel
            : b.group.memberCount(members.length)}
        </span>
      </Tip>
      <Tip label={b.group.settingsHint(group)}>
        <Button
          aria-label={b.group.settingsLabel(group)}
          className="shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
          onClick={() => setSettingsOpen(true)}
          size="sm"
          variant="ghost"
        >
          <Codicon name="gear" />
        </Button>
      </Tip>
      <Tip label={b.group.disbandHint(group)}>
        <Button
          aria-label={b.group.disbandLabel(group)}
          className="shrink-0 text-(--ui-text-tertiary) hover:text-destructive"
          onClick={() => setConfirmDisband(true)}
          size="sm"
          variant="ghost"
        >
          <Codicon name="trash" />
        </Button>
      </Tip>
    </div>
  )

  const memberDescriptors = () =>
    members.map(b => ({
      ...b,
      title: (b.remoteSource ? '' : allMeta[b.name]?.title) || b.title || ''
    }))

  // Activity disclosure: quiet, collapsed by default. The collapsed row shows
  // the latest event; expanding lists the current run's events newest-first.
  // Events are epoch-tagged, so a superseded run's history drops out of view.
  const activityEvents: GroupActivityEntry[] = currentGroupActivity(group)
  const latestActivity = activityEvents.length ? activityEvents[activityEvents.length - 1] : null

  // #94570 shell rewired onto the real primitive (#91868/#94569): the button
  // must stop the ROUND, not just spray per-member interrupts — without the
  // epoch bump + holds the loop marched on to the next member. Thread scope:
  // the run being stopped is the one the latest activity belongs to.
  const stopRoomRun = async () => {
    await stopGroupThread(group, latestActivity?.thread || null, memberDescriptors())
    host.notify({
      kind: 'success',
      message: b.group.stopped(group)
    })
  }

  const activityPanel = (
    <div className="border-b border-(--ui-stroke-secondary)">
      <div className="flex items-center gap-1">
        <RowButton
          aria-controls={`group-activity:${group}`}
          aria-expanded={activityOpen}
          className="flex min-w-0 flex-1 items-center gap-1.5 px-2.5 py-1 text-left text-[0.7rem] text-(--ui-text-quaternary) transition-colors hover:text-foreground"
          onClick={() => setActivityOpen(prev => !prev)}
          title={activityOpen ? b.group.hideActivity : b.group.showActivity}
        >
          <Codicon className="shrink-0 text-[0.65rem]" name={activityOpen ? 'chevron-down' : 'chevron-right'} />
          <span className="shrink-0 font-medium">{b.group.activity}</span>
          {latestActivity ? (
            <span className="min-w-0 flex-1 truncate">{`${groupActivityLabel(latestActivity)} · ${relativeTime(latestActivity.at)}`}</span>
          ) : null}
        </RowButton>
        {room.running ? (
          <Tip label={b.group.stopHint}>
            <Button
              className="shrink-0 text-(--ui-accent)"
              onClick={() => void stopRoomRun()}
              size="xs"
              variant="ghost"
            >
              <Codicon name="debug-stop" />
              {b.group.stop}
            </Button>
          </Tip>
        ) : null}
      </div>
      {activityOpen ? (
        <div className="grid gap-0.5 px-2.5 pb-1.5" id={`group-activity:${group}`}>
          {activityEvents.length ? (
            [...activityEvents].reverse().map((event, i) => (
              <div className="flex items-center gap-1.5 text-[0.7rem]" key={`${event.at}:${i}`}>
                <Codicon
                  className={cn('shrink-0 text-[0.65rem]', groupActivityTone(event.kind))}
                  name={GROUP_ACTIVITY_GLYPHS[event.kind] || 'circle-outline'}
                />
                <span className={cn('min-w-0 flex-1 truncate', groupActivityTone(event.kind))}>
                  {groupActivityLabel(event)}
                </span>
                <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{relativeTime(event.at)}</span>
                {event.kind === 'working' ? (
                  <Tip label={b.group.stopHint}>
                    <Button
                      className="shrink-0 text-(--ui-accent)"
                      onClick={() => void stopRoomRun()}
                      size="micro"
                      variant="ghost"
                    >
                      <Codicon name="debug-stop" />
                      {b.group.stop}
                    </Button>
                  </Tip>
                ) : null}
              </div>
            ))
          ) : (
            <div className="px-0.5 pb-0.5 text-[0.625rem] text-(--ui-text-quaternary)">{b.group.noActivityYet}</div>
          )}
        </div>
      ) : null}
    </div>
  )

  const submit = () => {
    const text = draft.trim()
    const images = imagesFor(null)

    if (!text && !images.length) {
      return
    }

    const before = groupComposerDraftSnapshot(composerKeyRef.current)

    const cleared = updateComposerDraft(current => ({
      ...current,
      main: '',
      pendingAttachments: {
        ...(current.pendingAttachments || {}),
        main: []
      }
    }))

    // Main composer = START A NEW THREAD with the whole group (Slack shape).
    // Full descriptors ride into the turn loop: remote members keep their
    // connection fields so their turns route to their own machines.
    const minted = sendToGroupChat(group, memberDescriptors(), text, null, images)

    if (minted) {
      setOpenThreads(prev => ({
        ...prev,
        [minted]: true
      }))
    } else {
      const restored = restoreGroupComposerDraft(composerKeyRef.current, cleared.revision, before)

      if (restored) {
        setComposerDraft(restored)
      }
    }
  }

  const submitReply = (thread: string) => {
    const text = (replyDrafts[thread] || '').trim()
    const images = imagesFor(thread)

    if (!text && !images.length) {
      return
    }

    const before = groupComposerDraftSnapshot(composerKeyRef.current)

    const cleared = updateComposerDraft(current => ({
      ...current,
      pendingAttachments: {
        ...(current.pendingAttachments || {}),
        [thread]: []
      },
      replies: {
        ...(current.replies || {}),
        [thread]: ''
      }
    }))

    // Reply box = CONTINUE this thread; the member turns it triggers are
    // scoped to it.
    const sent = sendToGroupChat(group, memberDescriptors(), text, thread, images)

    if (sent) {
      setOpenThreads(prev => ({
        ...prev,
        [thread]: true
      }))
    } else {
      const restored = restoreGroupComposerDraft(composerKeyRef.current, cleared.revision, before)

      if (restored) {
        setComposerDraft(restored)
      }
    }
  }

  /** Pending-attachment chips + the picker for one composer (thread = null →
   *  main). Image chips preview the pixels; PDFs/files show a type icon.
   *  X removes it. */
  const attachmentRow = (thread: null | string) => {
    const images = imagesFor(thread)

    if (!images.length) {
      return null
    }

    return (
      <div className="flex flex-wrap items-center gap-1.5 px-1 pb-1">
        {images.map((img, index) => (
          <div
            className="flex items-center gap-1 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) px-1 py-0.5"
            key={`${img.name || 'img'}:${index}`}
          >
            {img.kind === 'pdf' || img.kind === 'file' ? (
              <Codicon
                className="text-[0.9rem] text-(--ui-text-tertiary)"
                name={img.kind === 'pdf' ? 'file-pdf' : 'file'}
              />
            ) : (
              <img alt="" className="size-6 rounded object-cover" src={img.data} />
            )}
            <span className="max-w-32 truncate text-[0.65rem] text-(--ui-text-tertiary)">{img.name || 'image'}</span>
            <Tip label={b.group.removeAttachment}>
              <Button
                aria-label={b.group.removeAttachment}
                onClick={() => removeImage(thread, index)}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="close" />
              </Button>
            </Tip>
          </div>
        ))}
      </div>
    )
  }

  const attachButton = (thread: null | string) => (
    <Button
      className="shrink-0 text-(--ui-text-tertiary) hover:text-foreground"
      onClick={() => void pickGroupAttachments().then(picked => addImages(thread, picked))}
      size="sm"
      title={b.group.attachHint}
      type="button"
      variant="ghost"
    >
      <Codicon name="attach" />
    </Button>
  )

  // One log entry, rendered exactly as before conversation folding existed.
  const renderEntry = (entry: GroupMessage, index: number) => {
    const isUser = entry.from.kind === 'user'
    const meta = isUser || entry.from.source ? null : allMeta[entry.from.name]

    // Match this speaker back to its member descriptor so display
    // names and disambiguating handles come from the roster (the
    // primary "default" profile renders as Hermes, remote dupes
    // carry their @name-device handle) instead of raw profile ids.
    const member = isUser
      ? null
      : members.find(
          b =>
            b.name === entry.from.name &&
            (entry.from.source ? (b.connectionLabel || b.connectionId) === entry.from.source : !b.remoteSource)
        ) || null

    const display = isUser
      ? 'You'
      : displayName(
          member || {
            name: entry.from.name
          },
          meta
        )

    const entryKey = `${entry.at}:${index}`
    const revealed = !isUser && revealedSpeaker === entryKey

    // Clicked: append the gateway name so same-named agents on
    // two connections are tellable apart on demand.
    const label = isUser
      ? 'You'
      : revealed
        ? `${display}${entry.from.source ? `-${entry.from.source}` : ''} (@${botHandle(entry.from.name, member || undefined)})`
        : display

    // Speaker avatar: same appearance pipeline as the roster
    // (custom image/pet, else deterministic shape+color face).
    // Remote speakers have no local meta and get the
    // deterministic face for their name — stable per bot.
    // Non-null exactly when !isUser — the user's own lines carry no avatar.
    const appearance = isUser ? null : botAppearance(entry.from.name, meta)
    const image = appearance?.image ?? null
    const photo = Boolean(image && !isBackfilledFacePng(image))

    return (
      <div
        className={cn(
          'group flex items-start gap-2',
          isUser ? 'rounded-md bg-(--chrome-action-hover) px-2 py-1.5' : 'px-2 py-1'
        )}
        key={entryKey}
      >
        {appearance ? (
          <div className="mt-0.5 shrink-0">
            <BotFace
              color={avatarColor(appearance.color, entry.from.name)}
              image={photo ? image : null}
              name={entry.from.name}
              shape={appearance.shape}
              size={24}
            />
          </div>
        ) : null}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {isUser ? (
              <span className="text-[0.7rem] font-semibold text-foreground">{label}</span>
            ) : (
              <Button
                className="text-left text-[0.7rem] font-semibold text-(--ui-accent)"
                onClick={() => setRevealedSpeaker(revealed ? null : entryKey)}
                size="inline"
                title={revealed ? 'Hide full handle' : 'Show full handle'}
                variant="text"
              >
                {label}
              </Button>
            )}
            <span className="text-[0.625rem] text-(--ui-text-quaternary)">{relativeTime(entry.at)}</span>
            {entry.text.trim() ? (
              <div className="ml-auto shrink-0 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100">
                <CopyButton appearance="icon" buttonSize="icon" stopPropagation text={entry.text} />
              </div>
            ) : null}
          </div>
          <div
            className="text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:overflow-x-auto" // The app shell sets user-select: none globally; message bodies opt
            // back in so drag-select and ⌘C work in group chat logs.
            data-selectable-text="true"
          >
            {Streamdown ? <Streamdown>{entry.text}</Streamdown> : entry.text}
          </div>
          {/* User attachments: what every responding bot was */
          /* shown — image previews, or a named chip for */
          /* PDFs/files. */}
          {Array.isArray(entry.images) && entry.images.length ? (
            <div className="mt-1 flex flex-wrap items-center gap-1.5">
              {entry.images.map((img, imgIndex) =>
                img.kind === 'pdf' || img.kind === 'file' ? (
                  <div
                    className="flex items-center gap-1 rounded-md border border-(--ui-stroke-secondary) px-1.5 py-1 text-[0.65rem] text-(--ui-text-tertiary)"
                    key={`${entryKey}:img:${imgIndex}`}
                    title={img.name || 'attached file'}
                  >
                    <Codicon className="text-[0.8rem]" name={img.kind === 'pdf' ? 'file-pdf' : 'file'} />
                    <span className="max-w-48 truncate">{img.name || 'attached file'}</span>
                  </div>
                ) : (
                  <img
                    alt={img.name || 'attached image'}
                    className="max-h-40 max-w-60 rounded-md border border-(--ui-stroke-secondary) object-contain"
                    key={`${entryKey}:img:${imgIndex}`}
                    src={img.data}
                    title={img.name || 'attached image'}
                  />
                )
              )}
            </div>
          ) : null}
        </div>
      </div>
    )
  }

  // Threads: group entries by thread id (hydration assigned legacy ids, but
  // guard live pre-thread entries too), ordered by last activity — oldest
  // first, so the busiest/newest thread sits at the bottom by the composer.
  // The most recently ACTIVE thread renders open; older ones collapse to a
  // Slack-style summary row unless explicitly opened. Every open thread gets
  // its own reply box, which continues THAT thread.
  const threadsById = new Map<string, GroupThreadBucket>()

  for (let i = 0; i < room.log.length; i++) {
    const entry = room.log[i]
    const id = groupThreadOf(entry)
    let bucket = threadsById.get(id)

    if (!bucket) {
      bucket = {
        entries: [],
        id,
        startIndex: i
      }
      threadsById.set(id, bucket)
    }

    bucket.entries.push({
      entry,
      index: i
    })
  }

  const threads = [...threadsById.values()].sort(
    (a, b) => (a.entries[a.entries.length - 1].entry.at || 0) - (b.entries[b.entries.length - 1].entry.at || 0)
  )

  const newestThread = threads.length ? threads[threads.length - 1].id : null
  const logChildren: ReactNode[] = []
  threads.forEach(threadBucket => {
    const { entries, id } = threadBucket
    const head = entries.find(({ entry }) => entry.from.kind === 'user')?.entry || entries[0].entry
    const isNewest = id === newestThread
    const expanded = openThreads[id] ?? isNewest

    if (!expanded) {
      const replies = entries.length - 1
      const headText = stripPreviewMarkdown(head?.text || '').slice(0, 80)
      logChildren.push(
        <RowButton
          className="flex w-full items-center gap-2 rounded-md border border-(--ui-stroke-secondary) px-2 py-1.5 text-left text-xs text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover)"
          key={`fold:${id}`}
          onClick={() =>
            setOpenThreads(prev => ({
              ...prev,
              [id]: true
            }))
          }
          title={b.group.openThread}
        >
          <Codicon className="shrink-0 text-[0.65rem]" name="chevron-right" />
          <span className="min-w-0 flex-1 truncate">{headText || b.group.threadFallback}</span>
          <span className="shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{`${b.group.replyCount(replies)} · ${relativeTime(entries[entries.length - 1].entry.at)}`}</span>
        </RowButton>
      )

      return
    }

    // Open thread: a rail-indented block — collapse affordance, its entries,
    // and its own reply box (Slack's "reply in thread").
    const threadRows: ReactNode[] = []

    if (!isNewest || openThreads[id] !== undefined) {
      threadRows.push(
        <RowButton
          className="flex w-full items-center gap-1.5 px-2 pt-1 text-left text-[0.65rem] text-(--ui-text-quaternary) transition-colors hover:text-foreground"
          key={`unfold:${id}`}
          onClick={() =>
            setOpenThreads(prev => ({
              ...prev,
              [id]: false
            }))
          }
          title={b.group.collapseThreadLabel}
        >
          <Codicon className="text-[0.6rem]" name="chevron-down" />
          {b.group.collapseThread}
        </RowButton>
      )
    }

    for (const { entry, index } of entries) {
      threadRows.push(renderEntry(entry, index))
    }

    // Reply-in-thread: the newest thread's continuation ALSO lives here, so
    // the main composer below can stay "new thread" without ambiguity.
    threadRows.push(
      replyThread === id ? (
        <form
          className="grid gap-0 px-2 pb-1"
          key={`replybox:${id}`}
          onSubmit={event => {
            event.preventDefault()
            submitReply(id)
          }}
        >
          {attachmentRow(id)}
          <div className="flex items-center gap-1.5">
            <GroupMentionInput
              aria-label={b.group.replyInThread}
              autoFocus
              members={members}
              onChange={text =>
                setReplyDrafts(prev => ({
                  ...prev,
                  [id]: text
                }))
              }
              onPaste={event => pasteImages(id, event)}
              onSubmitDraft={() => submitReply(id)}
              placeholder={b.group.replyInThreadPlaceholder}
              value={replyDrafts[id] || ''}
            />
            {attachButton(id)}
            <Button disabled={!(replyDrafts[id] || '').trim() && !imagesFor(id).length} size="sm" type="submit">
              {b.group.reply}
            </Button>
          </div>
        </form>
      ) : (
        <Button
          className="w-fit px-2 pb-1 text-left text-[0.65rem] text-(--ui-accent) transition-colors"
          key={`replylink:${id}`}
          onClick={() => setReplyThread(id)}
          size="inline"
          variant="link"
        >
          {b.group.replyInThread}
        </Button>
      )
    )
    logChildren.push(
      <div className="grid gap-1.5 border-l-2 border-(--ui-stroke-secondary) pl-1.5" key={`thread:${id}`}>
        {threadRows}
      </div>
    )
  })

  return (
    <div
      className="relative flex h-full flex-col"
      data-preview-annotate-composer-key={composerKey}
      data-preview-annotate-destination="group"
      data-preview-annotate-group={group}
      data-preview-annotate-owner-key={groupWorkspaceOwnerKey(group)}
      onDragLeave={event => {
        // Only clear when leaving the room container itself, not when the
        // cursor moves between its children. React types relatedTarget as a
        // bare EventTarget; on a drag it is always an element or null.
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setDragOver(false)
        }
      }}
      onDragOver={event => {
        if ([...(event.dataTransfer?.types || [])].includes('Files')) {
          event.preventDefault()
          setDragOver(true)
        }
      }}
      onDrop={dropFiles}
    >
      {dragOver ? (
        <div
          className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center border-2 border-dashed border-(--ui-accent) text-sm font-medium text-(--ui-accent)"
          key={'dropzone'}
        >
          {replyThread ? b.group.dropToThread : b.group.dropToRoom}
        </div>
      ) : null}
      {header}
      <GroupHoldStatus
        holds={room.holds}
        memberLabel={member => displayName(member, botRosterMeta(member, allMeta))}
        members={members}
      />
      {activityPanel}
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="grid gap-1.5 px-2.5 pb-2">
          {room.log.length
            ? logChildren
            : [
                <div className="px-2 py-4 text-center text-xs text-(--ui-text-tertiary)" key={'empty'}>
                  {b.group.composerPlaceholder}
                </div>
              ]}
          {roomClarifies.map(entry => (
            <GroupClarifyCard entry={entry} key={`clarify:${entry.memberKey}:${entry.requestId}`} members={members} />
          ))}
          {room.running ? (
            <div className="px-2 py-1 text-[0.7rem] italic text-(--ui-text-quaternary)" key={'working'}>
              {roomClarifies.length
                ? b.group.waitingForAnswer
                : room.turn
                  ? b.group.memberThinking(groupSpeakerLabel(room.turn))
                  : b.group.roomWorking}
            </div>
          ) : null}
          {/* Scroll anchor (#89835): rooms opened at scroll position 0, mid- */
          /* history. The effect below scrolls this sentinel into view on */
          /* mount and on log growth — unless the user has scrolled up. */}
          <div aria-hidden key={'bottom-sentinel'} ref={bottomSentinelRef} />
        </div>
      </div>
      <div className="border-t border-(--ui-stroke-secondary) p-2">
        <form
          className="grid gap-0"
          onSubmit={event => {
            event.preventDefault()
            submit()
          }}
        >
          {attachmentRow(null)}
          <div className="flex items-center gap-1.5">
            <GroupMentionInput
              aria-label={b.group.messageRoom(group)}
              members={members}
              onChange={setDraft}
              onPaste={event => pasteImages(null, event)}
              onSubmitDraft={submit}
              placeholder={b.group.newThreadPlaceholder(group)}
              value={draft}
            />
            {attachButton(null)}
            <Button disabled={!draft.trim() && !imagesFor(null).length} size="sm" type="submit">
              {b.group.newThread}
            </Button>
          </div>
        </form>
      </div>
      <GroupChatSettingsDialog
        group={group}
        members={members}
        onClose={() => setSettingsOpen(false)}
        open={settingsOpen}
      />
      <ConfirmDialog
        busyLabel={b.group.disbanding}
        confirmLabel={b.group.disbandAction}
        description={
          /* New rooms title member sessions by roomId, legacy rooms by name — */
          /* so the copy names the concept, not a literal session title. The */
          /* name is bolded mid-sentence, so the copy splits around it and the */
          /* prefix goes empty where the name leads (core's deleteDesc* shape). */
          <span>
            {b.group.disbandDescPrefix}
            <span className="font-medium text-foreground">{group}</span>
            {b.group.disbandDescSuffix(members.length)}
          </span>
        }
        destructive
        doneLabel={b.group.disbandDone}
        onClose={() => setConfirmDisband(false)}
        onConfirm={async () => {
          clearGroupComposerDraft(composerKeyRef.current)
          await disbandGroupChat(group, members)
          host.notify({
            kind: 'success',
            message: b.group.disbanded(group)
          })
        }}
        open={confirmDisband}
        title={b.group.disbandTitle}
      />
    </div>
  )
}

/** Main-window wrapper: seats the member roster reactively (live roster +
 *  bot meta + the room's stored cross-connection descriptors) so the room
 *  keeps working as members change while the tab is open. Also subscribes to
 *  this pane's visibility (feature-detected host.paneVisibility): retained
 *  panes stay mounted while hidden, so the workspace needs the hidden →
 *  visible edge to re-anchor its log to the bottom (#89835 follow-up). */
interface GroupChatMainViewProps {
  group: string
}

function GroupChatMainView({ group }: GroupChatMainViewProps) {
  const allMeta = useValue($botMeta)
  // Subscribe: membership changes ride bot meta AND the room record.
  useValue($groupChats)
  const roster = useValue($lastRoster)
  const members = groupChatMemberBots(group, roster, allMeta)

  // Older SDKs have no paneVisibility: fall back to an always-visible atom so
  // the hook order stays stable and behavior matches the previous build.
  const $visible = useMemo(
    () =>
      typeof host.paneVisibility === 'function'
        ? host.paneVisibility(`plugin-workspace:${ID}:group:${slugify(group)}`)
        : atom(true),
    [group]
  )

  const visible = useValue($visible)

  return (
    <GroupChatWorkspace group={group} members={members} onBack={() => closeGroupChatMainTab(group)} visible={visible} />
  )
}

/** Open a group chat the Discord way: a tab taking over the MAIN chat window
 *  (host.openWorkspace, newer desktops), falling back to the in-panel room
 *  view on desktops whose SDK predates the main-area door.
 *
 *  Ordering matters (#89788 follow-up): the main tab must be RECORDED before
 *  the selection atom is set. Setting the atom first opened a window where
 *  BotsPane rendered with a selected group and an empty tab map — the
 *  in-pane fallback painted alongside the main tab, and because the map
 *  write itself repaints nothing, the duplicate stuck until an unrelated
 *  re-render. */
export function openGroupChat(group: string): void {
  // A room selection supersedes any bot-open transition still hydrating.
  // The in-flight host navigation may complete underneath this workspace,
  // but it may not later close or visually steal the room the user chose.
  bumpBotOpenGeneration()
  $groupNeedsYou.set({
    ...$groupNeedsYou.get(),
    [group]: false
  })
  const ownerKey = groupWorkspaceOwnerKey(group)
  setBotsWorkspaceOwner(ownerKey, null, 'New group conversations start in the group composer.')

  if (typeof host.openWorkspace === 'function') {
    try {
      const close = host.openWorkspace(`${ID}:group:${slugify(group)}`, {
        title: group,
        minWidth: '24rem',
        render: () => <GroupChatMainView group={group} />,
        onClose: () => {
          dropGroupMainTab(group)

          if ($groupChatWorkspace.get() === group) {
            $groupChatWorkspace.set(null)
          }
        }
      })

      recordGroupMainTab(group, close)
      // Tab ownership is on record — the atom now only drives the roster
      // highlight; shouldRenderGroupChatInPane stays false throughout.
      $groupChatWorkspace.set(group)

      return
    } catch {
      // Fall through to the in-panel room below.
    }
  }

  // No main-window door (older desktop) or it threw: select the group so
  // the in-panel room renders as the fallback surface.
  $groupChatWorkspace.set(group)
}
