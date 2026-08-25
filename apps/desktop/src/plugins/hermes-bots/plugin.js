/**
 * Hermes Bot Mode — a "one chat per agent" roster for the Hermes desktop.
 *
 * Left pane "Bots": one row per Hermes profile (a bot = an agent profile) with
 * a customizable avatar (shape + color + eyes, image, or pet). Click opens that
 * bot's chat; right-click → Edit Profile (avatar, title, description).
 * "New Bot" creates a profile — Name / Title / Description with an
 * "Advanced" disclosure for full profile config.
 *
 * Right tile "Routines": scheduled tasks (Hermes cron jobs) scoped to the
 * bot you're currently chatting with — follows the live gateway profile.
 *
 * Bots message each other straight into each bot's ONE canonical "Bot
 * Chat" — @-mentions deliver over gateway RPCs (no CLI relay), and
 * bot-initiated sends use `hermes -p <bot> chat --in ~ -c "Bot Chat"`.
 */

import * as sdk from '@hermes/plugin-sdk'
import {
  atom,
  Button,
  Checkbox,
  cn,
  Codicon,
  COMPOSER_AREAS,
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
  ConfirmDialog,
  CopyButton,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  EmptyState,
  GlyphSpinner,
  haptic,
  host,
  Input,
  PALETTE_AREA,
  profileColor,
  queryClient,
  relativeTime,
  ScrollArea,
  SearchField,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Switch,
  Textarea,
  Tip,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const { McpTab, ToolsetConfigPanel } = sdk
// Keep optional exports feature-detected; test harnesses may strip the SDK namespace.
const SkillsView = typeof sdk === 'undefined' ? undefined : sdk.SkillsView
// TRUE only on builds whose SkillsView routes `fixedConnection` to the pinned
// registry connection's backend. Older builds export SkillsView WITHOUT the
// prop — rendering it for a remote-target draft there would read/write the
// ACTIVE gateway's skills under the remote bot's name (the wrong machine),
// so those builds keep the staged checklists for remote targets.
const skillsViewRoutesConnections = Boolean(SkillsView && SkillsView.supportsFixedConnection)
const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown
// Deterministic blob avatars (name → face). Feature-detected: older SDKs
// without the export fall back to the legacy math-face shapes below.
const blobatarSvg = typeof sdk === 'undefined' ? undefined : sdk.blobatarSvg
// Budgeted render loop (fps cap + observability pause + dormancy + teardown).
// Feature-detected: older desktops fall back to the hand-rolled clock below.
const createBudgetedLoop = typeof sdk === 'undefined' ? undefined : sdk.createBudgetedLoop

const ID = 'hermes-bots'
/** Tree pane id of the Bots home workspace tab (openWorkspace prefixes
 *  `plugin-workspace:`). Tab visibility — not session focus — is what says
 *  who owns the CENTER once tabs exist; session focus only vetoes passive
 *  opens and, on its rising edge, yields the center to the chat. */
const BOTS_HOME_PANE_ID = `plugin-workspace:${ID}:home`
const ROSTER_KEY = [ID, 'roster']
const ROUTINES_KEY = [ID, 'routines']
const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
const BOT_META_V1_KEY = 'bot-meta'
const BOT_META_V2_KEY = 'bot-meta-v2'
const BOT_META_MIGRATION_KEY = 'bot-meta-v2-migrated'
let botMetaV2Active = false
let botMetaV2Commit = Promise.resolve()
const migratedLocalRoutes = new Map()

/** Captured in register() so components can reach plugin storage. */
let pluginCtx = null

/** Live roster snapshot for imperative handlers (context menus). */
const $lastRoster = atom([])

/** Last source inventory returned by the desktop-wide agent roster. */
const $lastSources = atom([])

/** Bots with chat activity the user hasn't seen yet (connectionId::profile -> true).
 *  Fed by the roster poll's activity watermark, so it catches EVERY
 *  delivery path: RPC, CLI (bot-to-bot), cron runs, other machines. */
const $botUnread = atom({})

// last_active watermark per source-qualified bot, seeded on first poll so a
// fresh mount doesn't mark ancient history unread.
const rosterWatermarks = new Map()
let watermarksSeeded = false

/** User pref: toast on every new bot activity. Default OFF — a busy roster
 *  (cron runs, bot-to-bot chatter) turns the toasts into a firehose, and the
 *  unread badge already carries the signal. Persisted via ctx.storage. */
const $activityToasts = atom(false)

/** Flip the activity-toast pref and persist it. */
function setActivityToasts(enabled) {
  $activityToasts.set(enabled)

  try {
    Promise.resolve(pluginCtx?.storage?.set?.('activity-toasts', enabled)).catch(() => undefined)
  } catch {
    /* storage unavailable — pref holds for this window only */
  }
}

/** Detect new inbound activity from a fresh roster: last_active moved past
 *  the watermark for a bot whose chat isn't on screen -> unread + toast.
 *  Watermarks follow botActivitySession (canonical Bot Chat included) —
 *  last_session alone never sees the hidden Bot Chat, so DMs delivered
 *  there would neither badge nor toast. */
function trackInboundActivity(roster) {
  const seeding = !watermarksSeeded
  watermarksSeeded = true

  for (const bot of roster) {
    const key = botSelectionKey(bot)
    const activity = botActivitySession(bot)
    const ts = activity?.last_active || 0
    const prev = rosterWatermarks.get(key) || 0
    rosterWatermarks.set(key, Math.max(prev, ts))

    if (seeding || ts <= prev) {
      continue
    }

    // Activity in the exact bot owner the user is currently looking at is
    // already visible — never badge the open chat or its same-named twin.
    if ($selectedBot.get() === key) {
      continue
    }

    $botUnread.set({ ...$botUnread.get(), [key]: true })

    // Roster-hidden bots stay quiet: the unread flag above accumulates
    // silently (unhiding reveals the badge) but a hidden bot never toasts.
    if (botRosterMeta(bot, $botMeta.get())?.hidden) {
      continue
    }

    // Toasts are opt-in: the unread badge is always set above, but the
    // per-message notification fires only when the user enabled it.
    if ($activityToasts.get()) {
      const meta = botRosterMeta(bot, $botMeta.get())
      const label = displayName(bot, meta)
      const preview = (activity?.preview || '').trim()
      const inbound = /^Message from/i.test(preview)

      host.notify({
        kind: 'info',
        title: inbound ? `\uD83E\uDD16 New message for ${label}` : `${label} has new activity`,
        message: preview.slice(0, 140) || 'Open the chat to see it.'
      })
    }
  }
}

// ── needs-attention badge (#93091 item 3) ───────────────────────────────────
// Attention-worthy failure classes — matches the #93091 item-1 reason-code
// enum (shipped separately). Until reason codes flow end-to-end,
// attentionReasonFromError ALSO classifies raw error text as a fallback so
// the badge works against current gateway error strings.
const BOT_ATTENTION_CLASSES = new Set([
  'agent_blocked',
  'provider_auth_or_access',
  'provider_quota_limit',
  'missing_config'
])

/** One-line user hint per attention class (roster badge tooltip). */
const BOT_ATTENTION_HINTS = {
  provider_auth_or_access: 'Sign in again for this profile',
  provider_quota_limit: 'Quota or balance exhausted',
  missing_config: 'Provider not configured — run hermes model',
  agent_blocked: 'Bot is blocked — see its last message'
}

/** Map an error (a #93091 reason code or raw error text) to an attention
 *  class, or null when the failure is transient (rate limit, server error,
 *  timeout) — transient classes must NEVER badge. Pure; tested directly. */
function attentionReasonFromError(errorTextOrReason) {
  const raw = String(errorTextOrReason || '').trim()

  if (!raw) {
    return null
  }

  if (BOT_ATTENTION_CLASSES.has(raw)) {
    return raw
  }

  const text = raw.toLowerCase()

  // Transient failures first, so a retryable error never sticks a badge.
  if (/rate.?limit|too many requests|\b429\b|\b5\d\d\b|server error|overloaded|timed?.?out|timeout|temporar/.test(text)) {
    return null
  }

  if (/no llm provider|no access token|not configured|no api key|missing api key/.test(text)) {
    return 'missing_config'
  }

  if (/\b401\b|\b403\b|unauthorized|forbidden|authentication|invalid.?api.?key|credentials? (are )?(invalid|expired)/.test(text)) {
    return 'provider_auth_or_access'
  }

  if (/quota|out of funds|insufficient (credits?|funds|balance)|payment required|\b402\b|billing/.test(text)) {
    return 'provider_quota_limit'
  }

  if (/\bblocked\b/.test(text)) {
    return 'agent_blocked'
  }

  return null
}

/** Per-bot needs-attention state: roster key -> {reason, at, message}.
 *  Display-only presentation state (never persisted, never alters delivery).
 *  Latest failure wins; the bot's next good turn clears it. Hidden bots keep
 *  their entry — hiding is a roster-DISPLAY concern only. */
const $botAttention = atom({})

/** Record attention for a bot after a failed turn/delivery. Transient errors
 *  classify to null and set nothing. Latest failure wins. */
function noteBotAttention(key, errorTextOrReason) {
  const reason = attentionReasonFromError(errorTextOrReason)

  if (!key || !reason) {
    return
  }

  $botAttention.set({
    ...$botAttention.get(),
    [key]: { reason, at: Date.now(), message: String(errorTextOrReason || '').trim().slice(0, 200) }
  })
}

/** A good turn clears the badge. */
function clearBotAttention(key) {
  if (!key || !$botAttention.get()[key]) {
    return
  }

  const next = { ...$botAttention.get() }
  delete next[key]
  $botAttention.set(next)
}

/** Last good cron list, same idea as the roster snapshot. */
const $lastJobs = atom([])

// Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar:
// canonical Bot Chats are plugin-owned forever-chats and group-chat member
// sessions are room plumbing — neither is a scratch conversation, and a
// 6-member room would otherwise dump six identical "Group: ..." rows into
// recents. Backed by the core generic `hidden` session flag (session.create
// hidden:true / session.set_hidden). Older gateways ignore the flag and the
// sessions simply stay visible there.

/** Bot the Routines tile is scoped to. Follows the live gateway profile
 *  (the bot you're actually chatting with) and roster clicks. */
const $selectedBot = atom('default')

/** Owner of the chat the user is LOOKING AT. Newer desktops expose a
 *  connection-qualified owner. Older builds synthesize the previous
 *  profile/gateway fallback and listen to both atoms when available. */
/** Source-qualified Bot Mode selection. Restoring it is presentation-only:
 *  it never activates a gateway or creates a session. */
const $selectedRosterKey = atom('')
const $selectedRosterHydrated = atom(false)
const $rosterHydrated = atom(false)
/** Mirrors host.paneVisibility('hermes-bots:pane') — wired in register(). */
const $botsPaneVisible = atom(false)
/** An explicit open landed: {key, openedRegistryId}. This transient view
 *  observation is empty only for the legacy newChat draft fallback. */
const $openBotChat = atom(null)
/** A session owns the main workspace. The roster highlight and the home /
 *  Cronjobs lifecycles all key off this rather than reading host.state
 *  conditionally from render. */
const $botChatFocused = atom(false)
/** True only while the Bots home is the visible main-area surface. A focused
 *  chat can remain alive behind it, so session focus alone cannot decide which
 *  roster row owns the visible workspace. */
const $botsHomeFronted = atom(false)

let botsHomeClose = null
let suppressBotsHomeReopen = false
// Latched while a re-front attempt has not yet been answered with visibility.
// Cleared the moment the home is actually fronted, and whenever the tab is
// retired — a fresh open starts the budget over. See openBotsHomeWorkspace.
let botsHomeRefrontTried = false

function saveSelectedRosterBot(bot) {
  const key = botRosterKey(bot)

  $selectedBot.set(botSelectionKey(bot))
  $selectedRosterKey.set(key)

  try {
    Promise.resolve(pluginCtx?.storage?.set?.('selected-roster-bot-v1', key)).catch(() => undefined)
  } catch {
    /* storage unavailable — selection lasts for this window */
  }
}

function clearSelectedRosterBot(bot) {
  clearSelectedRosterKey(botRosterKey(bot))
}

/** Drop the persisted selection when it is exactly this key — the caller has
 *  proven the owner is gone, not merely unreachable. An unreachable source
 *  KEEPS its key so the selection reconciles when the gateway returns. */
function clearSelectedRosterKey(key) {
  if ($selectedRosterKey.get() !== key) {
    return
  }

  $selectedRosterKey.set('')

  try {
    Promise.resolve(pluginCtx?.storage?.set?.('selected-roster-bot-v1', '')).catch(() => undefined)
  } catch {
    /* storage unavailable — selection is cleared for this window */
  }
}

/** Split a roster key back into its owner parts. Profile names cannot contain
 *  ':' (NAME_RE), so the first '::' is unambiguous. */
function parseRosterKey(key) {
  const raw = String(key || '')
  const at = raw.indexOf('::')

  if (at < 0) {
    return { connectionId: '', name: '' }
  }

  return { connectionId: raw.slice(0, at), name: raw.slice(at + 2) }
}

const $focusedBotProfile = host.state.focusedSessionProfile || host.state.profile

/** Profile that owns the chat currently on screen. Bot Mode opens another
 *  profile's session without moving the gateway socket, so mention filtering
 *  and sender identity must follow focus rather than host.state.profile. */
function focusedMentionProfile() {
  return String($focusedBotProfile.get?.() || '').trim() || 'default'
}

function fallbackFocusedBotOwner(profile = $focusedBotProfile.get?.()) {
  const focusedProfile = String(profile || 'default').trim() || 'default'
  const activeProfile = String(host.state.profile?.get?.() || 'default').trim() || 'default'

  // focusedSessionProfile without focusedSessionOwner is a legacy half-shape:
  // it carries no source identity. Only reuse the active connection when the
  // focused profile is also the active profile; otherwise fail closed rather
  // than manufacturing a cross-source owner from unrelated atoms.
  if (host.state.focusedSessionProfile && focusedProfile !== activeProfile) {
    return null
  }

  const connectionId = String(
    host.state.connectionId?.get?.() ||
    (typeof host.activeConnectionId === 'function' ? host.activeConnectionId() : '') ||
    ''
  ).trim()

  return {
    authoritative: false,
    connectionId,
    profile: focusedProfile
  }
}

const $focusedBotOwner = host.state.focusedSessionOwner || {
  get: () => fallbackFocusedBotOwner(),
  listen: listener => {
    const emit = profile => listener(fallbackFocusedBotOwner(profile))
    const unbindProfile = $focusedBotProfile.listen(emit)
    const unbindConnection = host.state.connectionId?.listen?.(() => emit($focusedBotProfile.get?.()))

    return () => {
      unbindProfile?.()
      unbindConnection?.()
    }
  }
}

function focusedRosterOwner(owner) {
  const name = String(owner?.profile || owner?.name || '').trim()

  if (!owner || !name) {
    return null
  }

  return {
    authoritative: owner.authoritative !== false,
    connectionId: String(owner.connectionId || '').trim(),
    name
  }
}

/** Optional secondary navigation inside the Bots pane (group-chat rooms). */

/** Group-chat rooms: { [group]: { log: [{from:{kind,name},text,at}], watermarks:{[member]:idx}, epoch, running } }.
 *  Log + watermarks persist via plugin storage; epoch/running are runtime-only. */
const $groupChats = atom({})
/** Group whose room view is open in the Bots pane (secondary navigation
 *  inside the pane; a normal row click returns to the roster). */
const $groupChatWorkspace = atom(null)
/** Groups whose latest room activity mentions @user — the needs-you badge. */
const $groupNeedsYou = atom({})
// Pending prompts (clarify questions AND command approvals) raised inside
// hidden group-member sessions, keyed `${group}::${memberKey}` (#90694).
// Members run in invisible plumbing sessions, so a member's blocking prompt
// used to park server-side with no surface to answer it — the user saw
// "is thinking…" until the prompt timeout. The turn poll mirrors each
// member's `pending_clarify` / `pending_approval` resume fields in here;
// the room renders answer cards from it.
const $groupClarify = atom({})

// ── group activity feed ─────────────────────────────────────────────────────
// Runtime-only, bounded per-room record of turn events that feeds the
// collapsible Activity view. Never persisted — it is presentation state like
// running/epoch, and the room transcript (log) stays the only durable record.
// Every event is tagged with the room epoch it belongs to, so the view shows
// only the CURRENT run: a newer send bumps the epoch (old-run events drop
// away), and a rename re-keys the room (the feed starts clean under the new
// name — stale events under the old key simply have no room to attach to).
const GROUP_ACTIVITY_LIMIT = 50
const $groupActivity = atom({})

function recordGroupActivity(group, event) {
  const room = $groupChats.get()[group]

  if (!room) {
    return null
  }

  const current = $groupActivity.get()[group] || { events: [] }
  const entry = { at: Date.now(), epoch: room.epoch || 0, ...event }
  const events = [...current.events, entry].slice(-GROUP_ACTIVITY_LIMIT)
  $groupActivity.set({ ...$groupActivity.get(), [group]: { ...current, events } })

  return entry
}

/** Events for the room's CURRENT run — superseded runs (epoch moved on)
 *  are dropped from view instead of describing work that already ended. */
function currentGroupActivity(group) {
  const epoch = ($groupChats.get()[group] || {}).epoch || 0
  return ($groupActivity.get()[group] || {}).events?.filter(event => (event.epoch || 0) === epoch) || []
}

/** Human label for one activity event, used by the collapsed summary and
 *  the expanded rows. */
function groupActivityLabel(event) {
  const kind = event?.kind
  const base = GROUP_ACTIVITY_LABELS[kind] || kind || 'did something'

  if (kind === 'cancelled' || kind === 'settled') {
    return base
  }

  const who = event?.member === 'You' ? 'You' : groupSpeakerLabel(event?.member || 'A bot')

  return `${who} ${base}`
}

const GROUP_ACTIVITY_LABELS = {
  queued: 'sent a message',
  working: 'is working…',
  replied: 'replied',
  passed: 'passed',
  'timed-out': 'took too long',
  failed: 'hit an error',
  cancelled: 'turn interrupted by a newer message',
  settled: 'turn settled',
  delivered: 'delivered a late reply',
  held: 'is held (stopped by you) — @mention it or say resume to release'
}

const GROUP_ACTIVITY_GLYPHS = {
  queued: 'comment',
  working: 'sync',
  replied: 'check',
  passed: 'circle-outline',
  'timed-out': 'clock',
  failed: 'error',
  cancelled: 'close',
  settled: 'check-all',
  delivered: 'mail-read',
  held: 'debug-pause'
}

/** Text tone for an activity row: quiet for pass/cancel/settle, accent for
 *  work and real replies, destructive for failures and timeouts. */
function groupActivityTone(kind) {
  if (kind === 'failed' || kind === 'timed-out') {
    return 'text-destructive'
  }

  if (kind === 'working' || kind === 'replied' || kind === 'delivered') {
    return 'text-(--ui-accent,#4f9cf9)'
  }

  return 'text-(--ui-text-tertiary)'
}

const GROUP_CHAT_SYNC_META_KEY = 'hermes-bots-groups'
// Gateway ui_meta is capped after Python JSON serialization. Keep a healthy
// margin below that limit because Python escapes Unicode while JS does not.
const GROUP_CHAT_SYNC_MAX_BYTES = 48000
const GROUP_CHAT_SYNC_MESSAGES = 16
const GROUP_CHAT_SYNC_TEXT_CHARS = 1200
const GROUP_CHAT_SYNC_IMAGE_CHARS = 24000
let groupChatSyncTimer = null
// Fan-out scheduler state, keyed by gateway connectionId ('' = active/local).
// Every connected gateway carries the full projection so a room survives any
// single gateway being removed and surfaces on every remote backend.
const groupChatSyncPendingByConnection = new Map()
const groupChatSyncInFlightConnections = new Set()
const groupChatSyncRetryTimers = new Map()
const groupChatSyncRetryCounts = new Map()
let groupChatSyncDisposed = false

/** Conservative byte count for the gateway's ensure_ascii JSON encoding.
 *  Python also inserts separator spaces, so reserve one extra byte per JS
 *  structural separator on top of escaped Unicode code-point widths. */
function groupChatGatewayJsonSize(value) {
  const json = JSON.stringify(value)
  let bytes = 0

  for (const character of json) {
    const codePoint = character.codePointAt(0)

    if (codePoint <= 0x7f) {
      bytes += 1
      if (character === ',' || character === ':') {
        bytes += 1
      }
    } else {
      bytes += codePoint <= 0xffff ? 6 : 12
    }
  }

  return bytes
}

/** Durable room identity for the sync projection. Rooms minted on current
 *  builds carry an immutable roomId; the projection keys rooms by
 *  `id:<roomId>` so rename is a display-name edit, not a distributed
 *  delete+create, and disband tombstones follow the room itself. Legacy
 *  rooms (no roomId) fall back to `name:<name>` keys with the older
 *  revision-gated tombstone semantics. */
function groupChatRoomKey(name, room) {
  return typeof room?.roomId === 'string' && room.roomId
    ? `id:${room.roomId}`
    : `name:${String(name)}`
}

/** Lift any historical projection shape (v1 wall-clock, v2 name-keyed) to
 *  the v3 room-key shape so one merge path serves mixed-version fleets. */
function normalizeGroupChatSyncSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') {
    return { version: 3, rooms: {}, deleted: {} }
  }
  if (Number(snapshot.version || 0) >= 3) {
    return {
      version: 3,
      updatedAt: Number(snapshot.updatedAt || 0),
      rooms: snapshot.rooms && typeof snapshot.rooms === 'object' ? snapshot.rooms : {},
      deleted: snapshot.deleted && typeof snapshot.deleted === 'object' ? snapshot.deleted : {}
    }
  }
  const rooms = {}
  for (const [name, room] of Object.entries(snapshot.rooms || {})) {
    if (!room || !Array.isArray(room.log)) {
      continue
    }
    rooms[`name:${name}`] = { ...room, name }
  }
  const deleted = {}
  for (const [name, at] of Object.entries(snapshot.deleted || {})) {
    // v1 tombstones carried wall-clock ms, not gateway revisions — they must
    // not outrank real revisions (same rule groupChatSyncDeletedRevision
    // applied before v3).
    deleted[`name:${name}`] = Number(snapshot.version || 0) >= 2 ? Math.max(0, Number(at || 0)) : 0
  }
  return { version: 3, updatedAt: Number(snapshot.updatedAt || 0), rooms, deleted }
}

/** Compact, display-oriented copy of Desktop's room log for gateway clients.
 *  The live orchestration state stays in plugin storage; this bounded mirror
 *  rides the default profile's ui_meta so mobile can show the same messages.
 *  Newest rooms/messages win when the profile metadata size cap is reached. */
function groupChatSyncSnapshot(all = $groupChats.get(), deleted = {}) {
  const ranked = Object.entries(all || {})
    // Empty runtime tombstones are used to stop an in-flight room after
    // disband. They are not real rooms and must never reappear on mobile.
    .filter(([, room]) => room && Array.isArray(room.log) && room.log.length > 0)
    .sort(([, left], [, right]) => {
      const leftAt = Number(left.log[left.log.length - 1]?.at || 0)
      const rightAt = Number(right.log[right.log.length - 1]?.at || 0)
      return rightAt - leftAt
    })
  const rooms = {}
  const boundedDeleted = Object.fromEntries(
    Object.entries(deleted)
      .sort(([, left], [, right]) => Number(right || 0) - Number(left || 0))
      .slice(0, 64)
  )
  const envelope = {
    version: 3,
    updatedAt: Date.now(),
    rooms,
    ...(Object.keys(boundedDeleted).length ? { deleted: boundedDeleted } : {})
  }

  for (const [name, room] of ranked) {
    const log = room.log.slice(-GROUP_CHAT_SYNC_MESSAGES).map(entry => ({
      ...(entry?.id ? { id: String(entry.id).slice(0, 160) } : {}),
      from: {
        kind: entry?.from?.kind === 'member' ? 'member' : 'user',
        name: String(entry?.from?.name || (entry?.from?.kind === 'member' ? 'Bot' : 'You')).slice(0, 128),
        ...(entry?.from?.source ? { source: String(entry.from.source).slice(0, 128) } : {})
      },
      text: String(entry?.text || '').slice(0, GROUP_CHAT_SYNC_TEXT_CHARS),
      at: Number(entry?.at || 0),
      ...(entry?.thread ? { thread: String(entry.thread).slice(0, 128) } : {})
    }))
    const compact = {
      name: String(name).slice(0, 64),
      ...(typeof room?.roomId === 'string' && room.roomId ? { roomId: String(room.roomId).slice(0, 128) } : {}),
      log,
      revision: Math.max(0, Number(room?.syncRevision ?? room?.revision ?? 0)),
      members: (Array.isArray(room.members) ? room.members : []).slice(0, GROUP_CHAT_MAX_MEMBERS).map(member => ({
        name: String(member?.name || '').slice(0, 128),
        ...(member?.handle ? { handle: String(member.handle).slice(0, 128) } : {}),
        ...(member?.connectionId ? { connectionId: String(member.connectionId).slice(0, 128) } : {}),
        ...(member?.connectionKind ? { connectionKind: String(member.connectionKind).slice(0, 64) } : {}),
        ...(member?.connectionLabel ? { connectionLabel: String(member.connectionLabel).slice(0, 128) } : {}),
        ...(member?.sourceScoped ? { sourceScoped: true } : {})
      })),
      ...(typeof room?.image === 'string' && room.image.length <= GROUP_CHAT_SYNC_IMAGE_CHARS
        ? { image: room.image }
        : {})
    }

    const key = groupChatRoomKey(name, room)
    rooms[key] = compact
    while (compact.log.length > 1 && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      compact.log.shift()
    }
    if (compact.image && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete compact.image
    }
    if (groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete rooms[key]
    }
  }

  return envelope
}

function groupChatSyncEntryKey(entry) {
  if (entry?.id) {
    return `id:${String(entry.id)}`
  }
  return JSON.stringify([
    Number(entry?.at || 0),
    String(entry?.from?.kind || ''),
    String(entry?.from?.name || ''),
    String(entry?.from?.source || ''),
    // Threadless entries (pre-thread rooms, older Desktop builds) get
    // SYNTHETIC `legacy-N` ids from assignLegacyThreads. Those ids are
    // position-derived — not stable across a gateway round-trip (the
    // projection copy may be threadless or numbered differently). Collapse
    // the whole synthetic family to one bucket, or the merge duplicates
    // every id-less entry — shifting watermarks and manufacturing phantom
    // member turns that re-submit into busy sessions.
    String(entry?.thread || 'legacy').replace(/^legacy-\d+$/, 'legacy'),
    String(entry?.text || '')
  ])
}

function groupChatSyncMemberKey(member) {
  return JSON.stringify([
    String(member?.source || ''),
    String(member?.connectionId || ''),
    String(member?.connectionLabel || ''),
    String(member?.handle || ''),
    String(member?.name || '')
  ])
}

function groupChatSyncDeletedRevision(source, value) {
  return Number(source?.version || 0) >= 2 ? Math.max(0, Number(value || 0)) : 0
}

/** Merge two bounded projections without treating an absent room/message as
 *  deletion. Rooms are identified by durable room keys (id:<roomId> when the
 *  room carries one), so a rename is a same-key field update — never a
 *  distributed delete+create — and a disband tombstone follows the room
 *  itself. Gateway revisions order identity/membership/picture and
 *  tombstones; stable message ids make concurrent log union idempotent.
 *  `changedRooms`/`deletedRooms` accept display names or room keys. */
function mergeGroupChatSyncSnapshots(
  remote,
  local,
  { changedRooms = [], deletedRooms = [], writeRevision = 0 } = {}
) {
  const remoteNorm = normalizeGroupChatSyncSnapshot(remote)
  const localNorm = normalizeGroupChatSyncSnapshot(local)
  const keysFor = (label, norm) => {
    const keys = new Set()
    for (const [key, room] of Object.entries(norm.rooms || {})) {
      if (key === label || String(room?.name || '') === label || key === `name:${label}`) {
        keys.add(key)
      }
    }
    if (String(label).startsWith('id:') || String(label).startsWith('name:')) {
      keys.add(label)
    } else if (!keys.size) {
      keys.add(`name:${label}`)
    }
    return keys
  }
  const changed = new Set()
  for (const label of changedRooms) {
    for (const key of keysFor(label, localNorm)) {
      changed.add(key)
    }
  }
  const deleted = {}
  for (const source of [remoteNorm, localNorm]) {
    for (const [key, at] of Object.entries(source.deleted || {})) {
      deleted[key] = Math.max(Number(deleted[key] || 0), Math.max(0, Number(at || 0)))
    }
  }
  for (const label of deletedRooms) {
    for (const key of new Set([...keysFor(label, remoteNorm), ...keysFor(label, localNorm)])) {
      // Rename passes changedRooms:[newName] + deletedRooms:[oldName]. For an
      // id-keyed room both labels resolve to the SAME durable key (the remote
      // copy still carries the old display name), and id tombstones are
      // final — so tombstoning here would kill the room being renamed. A key
      // that is being written this cycle is a rename target, not a disband.
      if (changed.has(key)) {
        continue
      }
      deleted[key] = Math.max(Number(deleted[key] || 0), Number(writeRevision || 0))
    }
  }

  const rooms = {}
  const roomKeys = new Set([...Object.keys(remoteNorm.rooms || {}), ...Object.keys(localNorm.rooms || {})])
  for (const key of roomKeys) {
    const remoteRoom = remoteNorm.rooms?.[key]
    const localRoom = localNorm.rooms?.[key]
    if ((!remoteRoom || !Array.isArray(remoteRoom.log)) && (!localRoom || !Array.isArray(localRoom.log))) {
      continue
    }
    const remoteRevision = Math.max(0, Number(remoteRoom?.revision || 0))
    const localRevision = changed.has(key)
      ? Math.max(0, Number(writeRevision || 0))
      : Math.max(0, Number(localRoom?.revision || 0))
    const entries = new Map()
    for (const entry of [...(remoteRoom?.log || []), ...(localRoom?.log || [])]) {
      entries.set(groupChatSyncEntryKey(entry), entry)
    }

    // Identity fields (display name, membership, picture) follow the higher
    // revision; a tie unions members and prefers the local writer's fields.
    let identity
    let members
    let image
    if (localRevision > remoteRevision) {
      identity = localRoom
      members = [...(localRoom?.members || [])]
      image = localRoom?.image
    } else if (remoteRevision > localRevision) {
      identity = remoteRoom
      members = [...(remoteRoom?.members || [])]
      image = remoteRoom?.image
    } else {
      identity = localRoom || remoteRoom
      const byId = new Map()
      for (const member of [...(remoteRoom?.members || []), ...(localRoom?.members || [])]) {
        byId.set(groupChatSyncMemberKey(member), member)
      }
      members = [...byId.values()]
      image = Object.prototype.hasOwnProperty.call(localRoom || {}, 'image') ? localRoom.image : remoteRoom?.image
    }
    rooms[key] = {
      ...(identity?.name ? { name: identity.name } : {}),
      ...(identity?.roomId || (key.startsWith('id:') ? key.slice(3) : '')
        ? { roomId: identity?.roomId || key.slice(3) }
        : {}),
      log: [...entries.values()].sort((left, right) => {
        const byTime = Number(left?.at || 0) - Number(right?.at || 0)
        return byTime || groupChatSyncEntryKey(left).localeCompare(groupChatSyncEntryKey(right))
      }),
      members,
      revision: Math.max(remoteRevision, localRevision),
      ...(typeof image === 'string' && image ? { image } : {})
    }
  }

  for (const [key, deletedRevision] of Object.entries(deleted)) {
    if (key.startsWith('id:')) {
      // Tombstones for id-keyed rooms are FINAL: the roomId is minted once
      // and never reused (same-name recreation mints a fresh id), so a
      // resurrect-by-revision race is structurally impossible. Keep the
      // tombstone even when a lagging gateway's copy carries a higher
      // revision — that copy is the resurrection this exists to prevent.
      delete rooms[key]
    } else if (Number(deletedRevision || 0) >= Number(rooms[key]?.revision || 0)) {
      delete rooms[key]
    } else {
      delete deleted[key]
    }
  }

  return groupChatSyncEnvelope(rooms, deleted)
}

/** Assemble + size-bound a v3 envelope from already-compacted rooms. */
function groupChatSyncEnvelope(rooms, deleted = {}) {
  const boundedDeleted = Object.fromEntries(
    Object.entries(deleted)
      .sort(([, left], [, right]) => Number(right || 0) - Number(left || 0))
      .slice(0, 64)
  )
  const envelope = {
    version: 3,
    updatedAt: Date.now(),
    rooms,
    ...(Object.keys(boundedDeleted).length ? { deleted: boundedDeleted } : {})
  }
  const ranked = Object.entries(rooms).sort(([, left], [, right]) => {
    const leftAt = Number(left?.log?.[left.log.length - 1]?.at || 0)
    const rightAt = Number(right?.log?.[right.log.length - 1]?.at || 0)
    return leftAt - rightAt
  })
  for (const [key, room] of ranked) {
    while ((room.log?.length || 0) > 1 && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      room.log.shift()
    }
    if (room.image && groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete room.image
    }
    if (groupChatGatewayJsonSize(envelope) > GROUP_CHAT_SYNC_MAX_BYTES) {
      delete rooms[key]
    }
  }
  return envelope
}

/** Merge the gateway's bounded display projection into Desktop's richer room
 *  state without discarding local session/watermark/runtime fields. Missing
 *  remote rooms/messages are not deletions; only explicit tombstones remove a
 *  room, and a genuinely newer local message wins over a stale tombstone. */
function mergeRemoteGroupChatSnapshotIntoRooms(
  remote,
  current = $groupChats.get(),
  { preserveRooms = [], deletedRooms = [] } = {}
) {
  const remoteNorm = normalizeGroupChatSyncSnapshot(remote)
  const rooms = { ...(current || {}) }
  const preserved = new Set(preserveRooms)
  const locallyDeleted = new Set(deletedRooms)

  // Local rooms indexed by durable identity so an id-keyed projection room
  // finds its local twin even when the display name changed remotely.
  const localByRoomId = new Map()
  for (const [name, room] of Object.entries(rooms)) {
    if (typeof room?.roomId === 'string' && room.roomId) {
      localByRoomId.set(room.roomId, name)
    }
  }

  for (const [key, projected] of Object.entries(remoteNorm.rooms || {})) {
    if (!projected || !Array.isArray(projected.log)) {
      continue
    }
    const projectedRoomId = projected.roomId || (key.startsWith('id:') ? key.slice(3) : null)
    const localName = projectedRoomId && localByRoomId.has(projectedRoomId)
      ? localByRoomId.get(projectedRoomId)
      : (projected.name && rooms[projected.name] ? projected.name : null)
    const displayName = String(projected.name || localName || (key.startsWith('name:') ? key.slice(5) : key))
    if (locallyDeleted.has(displayName) || (localName && locallyDeleted.has(localName))) {
      // Mid-rename guard: the remote copy may still be under the OLD display
      // name while the local record was already re-keyed (same roomId, new
      // name). That old name sits in deletedRooms, but the local record is
      // the rename in flight — deleting it here would kill the renamed room.
      if (localName && localName !== displayName && !locallyDeleted.has(localName)) {
        continue
      }
      delete rooms[displayName]
      if (localName) {
        delete rooms[localName]
      }
      continue
    }
    const existing = (localName ? rooms[localName] : rooms[displayName]) || {}
    const remoteRevision = Math.max(0, Number(projected.revision || 0))
    const localRevision = Math.max(0, Number(existing.syncRevision || 0))
    const entries = new Map(
      (Array.isArray(existing.log) ? existing.log : []).map(entry => [groupChatSyncEntryKey(entry), entry])
    )
    const members = new Map(
      (Array.isArray(existing.members) ? existing.members : []).map(member => [groupChatSyncMemberKey(member), member])
    )

    for (const entry of projected.log) {
      const entryKey = groupChatSyncEntryKey(entry)
      // The projection is COMPACT (truncated text, no images). When the same
      // entry exists locally, the local rich copy is authoritative — merging
      // the compact twin over it would strip attachments and retrigger
      // watermark deltas for members that already saw it (phantom rounds).
      if (!entries.has(entryKey)) {
        entries.set(entryKey, entry)
      }
    }
    const isPreserved = preserved.has(displayName) || (localName && preserved.has(localName))
    if (!isPreserved) {
      if (remoteRevision > localRevision) {
        members.clear()
      }
      for (const member of Array.isArray(projected.members) ? projected.members : []) {
        members.set(groupChatSyncMemberKey(member), { ...member, remoteSource: true })
      }
    }

    const log = assignLegacyThreads(
      [...entries.values()].sort((left, right) => {
        const byTime = Number(left?.at || 0) - Number(right?.at || 0)
        return byTime || groupChatSyncEntryKey(left).localeCompare(groupChatSyncEntryKey(right))
      })
    )
    const bounded = trimGroupChatLog(log, existing.watermarks || {})

    // A remote rename with a higher revision moves the local record to the
    // new display name; local views keyed by the old name follow on the
    // next repaint (roster derives from $groupChats keys).
    const targetName = !isPreserved && remoteRevision > localRevision ? displayName : (localName || displayName)
    if (localName && targetName !== localName) {
      delete rooms[localName]
    }

    rooms[targetName] = {
      ...existing,
      log: bounded.log,
      watermarks: bounded.watermarks,
      sessions: existing.sessions && typeof existing.sessions === 'object' ? existing.sessions : {},
      stranded: existing.stranded && typeof existing.stranded === 'object' ? existing.stranded : {},
      members: [...members.values()],
      ...(projectedRoomId || existing.roomId ? { roomId: existing.roomId || projectedRoomId } : {}),
      image: isPreserved
        ? existing.image || null
        : remoteRevision >= localRevision && Object.prototype.hasOwnProperty.call(projected, 'image')
          ? projected.image || null
          : existing.image || null,
      syncRevision: isPreserved ? localRevision : Math.max(remoteRevision, localRevision),
      epoch: Number(existing.epoch || 0),
      running: Boolean(existing.running)
    }
  }

  for (const [key, deletedAt] of Object.entries(remoteNorm.deleted || {})) {
    const deletedRoomId = key.startsWith('id:') ? key.slice(3) : null
    const targetName = deletedRoomId && localByRoomId.has(deletedRoomId)
      ? localByRoomId.get(deletedRoomId)
      : key.startsWith('name:') ? key.slice(5) : null
    if (!targetName || preserved.has(targetName)) {
      continue
    }
    if (deletedRoomId) {
      // Id tombstones are final — the id is never reused, so there is no
      // legitimate higher-revision recreation to protect.
      delete rooms[targetName]
    } else {
      const deletedRevision = Math.max(0, Number(deletedAt || 0))
      if (deletedRevision >= Number(rooms[targetName]?.syncRevision || 0)) {
        delete rooms[targetName]
      }
    }
  }
  for (const name of locallyDeleted) {
    delete rooms[name]
  }

  return rooms
}

function durableGroupChatRooms(all = $groupChats.get()) {
  const durable = {}

  for (const [name, room] of Object.entries(all || {})) {
    if (!room || !Array.isArray(room.log)) {
      continue
    }
    // Disband tombstones are runtime-only coordination state (they hold the
    // epoch bump for an in-flight drive). Persisting one would resurrect the
    // room as an empty record on the next load AND keep its name "taken" for
    // same-name recreates. Mirrors updateGroupChat's inline durable map.
    if (room.tombstone) {
      continue
    }
    durable[name] = {
      log: room.log,
      watermarks: room.watermarks || {},
      sessions: room.sessions || {},
      stranded: room.stranded || {},
      members: Array.isArray(room.members) ? room.members : [],
      // Immutable room identity: without this, a room merged in via the
      // remote-sync path (the only caller of this function) loses its
      // roomId on the next cold hydrate and falls back to legacy
      // name-keyed identity — same field updateGroupChat's inline map
      // already carries.
      roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
      image: room.image || null,
      syncRevision: Math.max(0, Number(room.syncRevision || 0))
    }
  }

  return durable
}

function persistGroupChatRooms(all = $groupChats.get()) {
  try {
    return Promise.resolve(pluginCtx?.storage?.set?.('group-chats', durableGroupChatRooms(all))).catch(() => undefined)
  } catch {
    return Promise.resolve()
  }
}

// ── deleted-connection roster hygiene (#93492 root cause) ───────────────────
// Deleting a cloud/remote connection used to leave every persisted group-chat
// member descriptor that referenced it behind untouched. Those orphaned rows
// (remoteSource: true, connection gone) are exactly the shape that made
// render-path route lookups throw "Bot X has no connection owner" on every
// group open, permanently — the poisoned row lives in plugin storage. The
// sweep below runs on the registry's 'removed' push (and its annotate helper
// again at hydrate for rows orphaned before this build). It never hard-deletes
// user data: the member row is kept and marked, so panes render the existing
// degraded 'Gateway removed' botSourceStatus state instead of crashing.

/** Keep the member's identity; mark it so botSourceStatus reads
 *  'Gateway removed' and no render-path route lookup can throw on it. */
function markOrphanedGroupMemberDescriptor(member) {
  return {
    ...member,
    sourceMissing: true,
    sourceReachable: false
  }
}

function groupMemberReferencesConnection(member, connectionId) {
  const id = String(connectionId || '').trim()

  if (!id) {
    return false
  }

  return (
    String(member?.connectionId || '').trim() === id ||
    String(member?.route?.connectionId || '').trim() === id
  )
}

/** Register-removed sweep: annotate (not delete) every persisted group-chat
 *  member owned by the deleted connection, in the atom AND plugin storage.
 *  Writes ride updateGroupChat so the durable record keeps its full shape
 *  (sessionOwners, holds — durableGroupChatRooms would drop them).
 *  Returns whether anything changed. */
function sweepGroupChatMembersForRemovedConnection(connectionId) {
  const id = String(connectionId || '').trim()

  if (!id) {
    return false
  }

  let changed = false

  for (const [name, room] of Object.entries($groupChats.get())) {
    const members = Array.isArray(room?.members) ? room.members : []

    if (!members.some(member => groupMemberReferencesConnection(member, id) && !member?.sourceMissing)) {
      continue
    }

    changed = true
    updateGroupChat(name, current => ({
      ...current,
      members: (Array.isArray(current.members) ? current.members : []).map(member =>
        groupMemberReferencesConnection(member, id) ? markOrphanedGroupMemberDescriptor(member) : member
      )
    }))
  }

  return changed
}

/** Hydrate-time pass for rows orphaned BEFORE this build (the poisoned rows
 *  that made #93492 survive app restarts). Two shapes are annotated, never
 *  deleted:
 *  - a descriptor that already lost its connectionId (route unresolvable —
 *    exactly what a stale row looks like once its connection was deleted);
 *  - a descriptor whose connectionId is absent from the live registry, when
 *    the caller could obtain one (liveConnectionIds === null means "registry
 *    unavailable", which must NOT read as "everything is orphaned").
 *  Pure on the rooms map; returns { rooms, changed }. */
function annotateOrphanedGroupChatMembers(rooms, liveConnectionIds = null) {
  // Duck-typed, not instanceof: callers (and vm-based tests) may hand a Set
  // constructed in another realm.
  const live = liveConnectionIds && typeof liveConnectionIds.has === 'function' ? liveConnectionIds : null
  const next = {}
  let changed = false

  for (const [name, room] of Object.entries(rooms || {})) {
    const members = Array.isArray(room?.members) ? room.members : []
    const orphaned = member => {
      if (!member || member.sourceMissing) {
        return false
      }

      if (!member.sourceScoped && !member.remoteSource) {
        return false
      }

      const id = String(member.route?.connectionId || member.connectionId || '').trim()

      if (!id) {
        // Route unresolvable: this is the row shape that threw on render.
        return true
      }

      return live ? !live.has(id) : false
    }

    if (!members.some(orphaned)) {
      next[name] = room
      continue
    }

    changed = true
    next[name] = {
      ...room,
      members: members.map(member => (orphaned(member) ? markOrphanedGroupMemberDescriptor(member) : member))
    }
  }

  return { rooms: next, changed }
}

function groupChatSyncConnectionId() {
  return String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || '')
}

/** Route a sync job back to the gateway that was active when it was queued.
 *  A foreground switch during debounce must not write the old snapshot into
 *  the newly active gateway. */
async function groupChatSyncRequest(job, method, params) {
  if (job.connectionId && typeof host.profileRoutes === 'function' && typeof host.requestProfile === 'function') {
    const routes = await host.profileRoutes()
    const route = (Array.isArray(routes) ? routes : []).find(candidate => {
      const profile = String(candidate?.targetProfile || candidate?.profile || '')
      return String(candidate?.connectionId || '') === job.connectionId && profile === 'default'
    })

    if (route) {
      return host.requestProfile(route, method, params)
    }
  }

  const currentConnectionId = groupChatSyncConnectionId()
  if (job.connectionId && currentConnectionId && job.connectionId !== currentConnectionId) {
    throw new Error('Group chat gateway changed before sync')
  }
  return host.request(method, params)
}

async function groupChatRemoteSnapshot(job) {
  const result = await groupChatSyncRequest(job, 'profiles.list', { include_sessions: false })
  const profile = (Array.isArray(result?.profiles) ? result.profiles : []).find(row => row?.name === 'default')
  const snapshot = profile?.ui_meta?.[GROUP_CHAT_SYNC_META_KEY]
  const supportsCas = Boolean(profile && Object.prototype.hasOwnProperty.call(profile, 'ui_meta_revisions'))
  return {
    snapshot: snapshot && typeof snapshot === 'object' && !Array.isArray(snapshot) ? snapshot : null,
    revision: Math.max(0, Number(profile?.ui_meta_revisions?.[GROUP_CHAT_SYNC_META_KEY] || 0)),
    supportsCas
  }
}

/** Pull the shared room projection into this Desktop before it publishes any
 *  local state. This is the receive half of the client-only sync contract. */
async function pullGroupChatServerState(connectionId = groupChatSyncConnectionId()) {
  const { snapshot: remote } = await groupChatRemoteSnapshot({ connectionId })

  if (!remote) {
    return false
  }
  const pending = groupChatSyncPendingByConnection.get(String(connectionId || ''))
  const merged = mergeRemoteGroupChatSnapshotIntoRooms(remote, $groupChats.get(), {
    preserveRooms: pending?.changedRooms || [],
    deletedRooms: pending?.deletedRooms || []
  })
  $groupChats.set(merged)
  await persistGroupChatRooms(merged)
  return true
}

function groupChatSyncBackoff(connectionId) {
  const count = Number(groupChatSyncRetryCounts.get(connectionId) || 0)
  return Math.min(30000, 1000 * 2 ** Math.min(count, 5))
}

function mergeGroupChatSyncJobs(existing, incoming) {
  if (!existing || existing.connectionId !== incoming.connectionId) {
    return incoming
  }
  return {
    connectionId: incoming.connectionId,
    allowEmpty: Boolean(existing.allowEmpty || incoming.allowEmpty),
    changedRooms: [...new Set([...(existing.changedRooms || []), ...(incoming.changedRooms || [])])],
    deletedRooms: [...new Set([...(existing.deletedRooms || []), ...(incoming.deletedRooms || [])])]
  }
}

function groupChatSyncPayloadEqual(left, right) {
  return (
    JSON.stringify(left?.rooms || {}) === JSON.stringify(right?.rooms || {}) &&
    JSON.stringify(left?.deleted || {}) === JSON.stringify(right?.deleted || {})
  )
}

/** Every default-profile gateway route this Desktop can currently reach.
 *  The projection fans out to ALL of them, so any single gateway can die or
 *  be removed without losing the shared room state, and gateway-only
 *  clients (Hermes Go, headless backends) see rooms regardless of which
 *  gateway a Desktop was foregrounding when the room was used. */
async function groupChatSyncTargetConnections() {
  const targets = new Set()
  const active = groupChatSyncConnectionId()
  targets.add(String(active || ''))
  if (typeof host.profileRoutes === 'function' && typeof host.requestProfile === 'function') {
    try {
      const routes = await host.profileRoutes()
      for (const route of Array.isArray(routes) ? routes : []) {
        const profile = String(route?.targetProfile || route?.profile || '')
        const connectionId = String(route?.connectionId || '')
        if (profile === 'default' && connectionId) {
          targets.add(connectionId)
        }
      }
    } catch {
      // Route inventory unavailable — the active gateway alone still syncs.
    }
  }
  return [...targets]
}

async function flushGroupChatServerSync(connectionId) {
  if (connectionId === undefined) {
    // Drain every connection with pending work.
    for (const pendingId of [...groupChatSyncPendingByConnection.keys()]) {
      void flushGroupChatServerSync(pendingId)
    }
    return
  }
  const id = String(connectionId || '')
  if (groupChatSyncDisposed || groupChatSyncInFlightConnections.has(id) || !groupChatSyncPendingByConnection.has(id)) {
    return
  }
  const job = groupChatSyncPendingByConnection.get(id)
  groupChatSyncPendingByConnection.delete(id)
  groupChatSyncInFlightConnections.add(id)

  try {
    const remoteState = await groupChatRemoteSnapshot(job)
    const local = groupChatSyncSnapshot($groupChats.get())
    const writeRevision = remoteState.revision + 1
    const snapshot = mergeGroupChatSyncSnapshots(remoteState.snapshot, local, {
      changedRooms: job.changedRooms,
      deletedRooms: job.deletedRooms,
      writeRevision
    })

    // Reconnect/startup reconciliation often discovers that the gateway
    // already holds the exact merged projection. Avoid advancing a revision
    // merely because a view reopened.
    if (!(job.changedRooms || []).length && !(job.deletedRooms || []).length && groupChatSyncPayloadEqual(snapshot, remoteState.snapshot)) {
      if (remoteState.snapshot) {
        const pending = groupChatSyncPendingByConnection.get(id)
        const mergedRooms = mergeRemoteGroupChatSnapshotIntoRooms(remoteState.snapshot, $groupChats.get(), {
          preserveRooms: pending?.changedRooms || [],
          deletedRooms: pending?.deletedRooms || []
        })
        $groupChats.set(mergedRooms)
        await persistGroupChatRooms(mergedRooms)
      }
      groupChatSyncRetryCounts.delete(id)
      return
    }

    const configureParams = {
      name: 'default',
      ui_meta: { [GROUP_CHAT_SYNC_META_KEY]: snapshot }
    }
    if (remoteState.supportsCas) {
      configureParams.ui_meta_expected_revisions = { [GROUP_CHAT_SYNC_META_KEY]: remoteState.revision }
    }
    const result = await groupChatSyncRequest(job, 'profiles.configure', configureParams)

    if (result?.applied?.ui_meta !== true) {
      throw new Error('Gateway rejected group chat ui_meta')
    }
    if (
      remoteState.supportsCas &&
      Number(result?.applied?.ui_meta_revisions?.[GROUP_CHAT_SYNC_META_KEY] || 0) !== writeRevision
    ) {
      throw new Error('Gateway did not advance group chat ui_meta revision')
    }

    const confirmedState = await groupChatRemoteSnapshot(job)
    if (remoteState.supportsCas && confirmedState.revision < writeRevision) {
      throw new Error('Group chat ui_meta revision missing after read-back')
    }
    if (confirmedState.snapshot) {
      const pending = groupChatSyncPendingByConnection.get(id)
      const mergedRooms = mergeRemoteGroupChatSnapshotIntoRooms(confirmedState.snapshot, $groupChats.get(), {
        preserveRooms: pending?.changedRooms || [],
        deletedRooms: pending?.deletedRooms || []
      })
      $groupChats.set(mergedRooms)
      await persistGroupChatRooms(mergedRooms)
    }
    groupChatSyncRetryCounts.delete(id)
  } catch {
    if (!groupChatSyncDisposed) {
      const retries = Number(groupChatSyncRetryCounts.get(id) || 0) + 1
      // A gateway that was REMOVED (not just flaky) has no route anymore and
      // would otherwise retry forever. Give up after the backoff ladder tops
      // out; local storage remains authoritative and a future reconnect of
      // that gateway re-seeds it via the gateway-transition pull/publish.
      if (retries > 8) {
        groupChatSyncRetryCounts.delete(id)
        return
      }
      groupChatSyncPendingByConnection.set(id, mergeGroupChatSyncJobs(groupChatSyncPendingByConnection.get(id), job))
      groupChatSyncRetryCounts.set(id, retries)
      if (typeof setTimeout === 'function' && !groupChatSyncRetryTimers.has(id)) {
        groupChatSyncRetryTimers.set(id, setTimeout(() => {
          groupChatSyncRetryTimers.delete(id)
          void flushGroupChatServerSync(id)
        }, groupChatSyncBackoff(id)))
      }
    }
  } finally {
    groupChatSyncInFlightConnections.delete(id)
    if (groupChatSyncPendingByConnection.has(id) && !groupChatSyncRetryTimers.has(id) && !groupChatSyncDisposed) {
      void flushGroupChatServerSync(id)
    }
  }
}

function stopGroupChatServerSync() {
  groupChatSyncDisposed = true
  groupChatSyncPendingByConnection.clear()
  if (groupChatSyncTimer !== null) {
    clearTimeout(groupChatSyncTimer)
    groupChatSyncTimer = null
  }
  for (const timer of groupChatSyncRetryTimers.values()) {
    clearTimeout(timer)
  }
  groupChatSyncRetryTimers.clear()
  groupChatSyncRetryCounts.clear()
}

/** Debounced, pull-merge-write server mirror, fanned out to every reachable
 *  default-profile gateway. Local storage keeps the complete orchestration
 *  log; ui_meta is a bounded cross-client projection per gateway, each with
 *  its own CAS revision stream. */
function scheduleGroupChatServerSync(
  all = $groupChats.get(),
  { allowEmpty = false, changedRooms = [], deletedRooms = [] } = {}
) {
  // Browser shells provide timers; source-level VM tests and older embedded
  // hosts may not. Room persistence must never break the surrounding gateway
  // lifecycle when the optional mirror cannot be scheduled.
  if (typeof setTimeout !== 'function') {
    return
  }
  const snapshot = groupChatSyncSnapshot(all)
  // A newly installed Desktop has no local room cache. Publishing that empty
  // state on hydrate/reconnect would erase a valid mirror produced elsewhere.
  // Only an explicit final-room disband is allowed to clear the projection.
  if (Object.keys(snapshot.rooms).length === 0 && !allowEmpty) {
    return
  }
  if (groupChatSyncTimer !== null) {
    clearTimeout(groupChatSyncTimer)
  }
  // Queue on the ACTIVE gateway synchronously (tests and older hosts have no
  // async route inventory), then widen to every reachable gateway before the
  // debounce fires.
  const activeId = String(groupChatSyncConnectionId() || '')
  const queueFor = connectionId => {
    const id = String(connectionId || '')
    const retryTimer = groupChatSyncRetryTimers.get(id)
    if (retryTimer !== undefined) {
      clearTimeout(retryTimer)
      groupChatSyncRetryTimers.delete(id)
    }
    groupChatSyncPendingByConnection.set(id, mergeGroupChatSyncJobs(groupChatSyncPendingByConnection.get(id), {
      connectionId: id,
      allowEmpty,
      changedRooms,
      deletedRooms
    }))
  }
  queueFor(activeId)
  groupChatSyncTimer = setTimeout(() => {
    groupChatSyncTimer = null
    void groupChatSyncTargetConnections()
      .then(targets => {
        for (const target of targets) {
          if (String(target || '') !== activeId) {
            queueFor(target)
          }
        }
      })
      .catch(() => undefined)
      .then(() => flushGroupChatServerSync())
  }, 350)
}

function handleSessionsGatewayTransition() {
  // A gateway swap invalidates any in-flight room drive: bump every room's
  // epoch so running loops bail at their next member boundary.
  const rooms = { ...$groupChats.get() }

  for (const name of Object.keys(rooms)) {
    rooms[name] = { ...rooms[name], epoch: (rooms[name].epoch || 0) + 1, running: false }
  }

  $groupChats.set(rooms)
  // Pull before re-publishing so a reconnect or source swap never lets this
  // client's stale cache hide a room written by another Desktop/mobile client.
  void pullGroupChatServerState()
    .catch(() => false)
    .then(() => scheduleGroupChatServerSync($groupChats.get()))
}

// ── cross-connection bot relay ────────────────────────────────────────────
// Connections ARE the peer set: every gateway this Desktop holds a socket
// to (local, remote URL, SSH, Hermes Cloud, docker) must be able to find
// every other connection's agents and message them via message_agent. The
// Desktop is the relay — it owns every socket. Two loops:
//  - roster loop: pushes each gateway the union roster of agents on the
//    OTHER connections (bot_relay.roster.sync), so message_agent resolves
//    cross-connection targets and Bot Chat prompts list them;
//  - drain loop: collects queued envelopes from every gateway
//    (bot_relay.outbox.drain), delivers each on the target connection's
//    own socket (bot_relay.deliver), and posts the reply back to the
//    sender gateway (bot_relay.reply) where a waiter wakes the sender.
// Older backends without the RPCs fail per-call and are skipped — the
// relay degrades to whatever subset of connections supports it.
const RELAY_ROSTER_INTERVAL_MS = 60_000
// Backstop cadence only (#93594): the push path below carries envelope latency,
// so the interval poll exists for older backends and missed events — 30s
// matches LIVE_SESSION_STATUS_BACKSTOP_INTERVAL_MS. It was 4s back when the
// poll WAS the delivery path, which (before route retention) also meant a
// fresh WebSocket dial + teardown per registered connection every 4s.
const RELAY_DRAIN_INTERVAL_MS = 30_000
// Push path (#93091): the gateway broadcasts `bot_relay.outbox.pending` when
// an envelope lands on disk; a burst of signals inside this window collapses
// to ONE drain. The interval poll above stays as the backstop for older
// backends (and connections whose events don't reach the tap).
const RELAY_PUSH_DEBOUNCE_MS = 250
let relayDisposed = false
let relayRosterTimer = null
let relayDrainTimer = null
let relayRosterBusy = false
let relayDrainBusy = false
let relayPushUnsub = null
let relayPushDebounceTimer = null
// A push landing while a drain is ALREADY running would be lost forever —
// the gateway signature is monotone (one event per new envelope, never
// re-broadcast) — so remember it and re-schedule after the drain finishes.
let relayDrainRerun = false
// Relay-route socket retention (#93594): connection id → release fn. While
// the relay is active each registered connection's pooled socket is pinned
// open (host.retainProfileSocket) so drain RPCs reuse ONE persistent
// WebSocket instead of dialing and tearing down a fresh one per tick.
// Feature-detected — older shells lack the door and fall back to per-call
// leases. Local routes get a no-op release inside the host (idle-reaper
// exemption). stopBotRelay releases everything.
const relayRouteRetentions = new Map()

/** Reconcile retention with the CURRENT connection set: pin new connections,
 *  release removed ones. Runs on every drain/roster connection fetch. */
function syncRelayRetention(connections) {
  if (typeof host.retainProfileSocket !== 'function') {
    return
  }

  const live = new Set(connections.map(connection => connection.id))

  for (const [id, release] of [...relayRouteRetentions]) {
    if (!live.has(id)) {
      relayRouteRetentions.delete(id)
      try {
        release()
      } catch {
        // Never let a release failure break the relay loop.
      }
    }
  }

  if (relayDisposed) {
    return
  }

  for (const connection of connections) {
    if (!relayRouteRetentions.has(connection.id)) {
      relayRouteRetentions.set(connection.id, host.retainProfileSocket(connection.route))
    }
  }
}

/** Drop every relay pin — stop/dispose path. */
function releaseRelayRetention() {
  for (const release of relayRouteRetentions.values()) {
    try {
      release()
    } catch {
      // Disposer from an older shell shape — never break teardown.
    }
  }

  relayRouteRetentions.clear()
}

/** One representative route per reachable connection id. */
async function relayConnections() {
  if (typeof host.profileRoutes !== 'function' || typeof host.requestProfile !== 'function') {
    return []
  }

  try {
    const routes = await host.profileRoutes()
    const byConnection = new Map()

    for (const route of Array.isArray(routes) ? routes : []) {
      const id = String(route?.connectionId || '')

      if (id && !byConnection.has(id)) {
        byConnection.set(id, route)
      }
    }

    return [...byConnection.entries()].map(([id, route]) => ({ id, route }))
  } catch {
    return []
  }
}

/** The agents living on one connection, as relay roster rows.
 *  Returns null on FAILURE (transient RPC blip, slow socket) — distinct from
 *  a genuine empty profile list. Conflating the two would push a fresh union
 *  roster missing a LIVE connection's agents, and the gateway-side liveness
 *  check (bot_relay._target_liveness) reads "absent from a fresh roster" as
 *  definitively offline → false runtime_offline refusals (#93091 item 2). */
async function relayAgentsOn(connection) {
  try {
    const res = await host.requestProfile(connection.route, 'profiles.list', { include_sessions: false })
    const profiles = Array.isArray(res?.profiles) ? res.profiles : []
    const label = String(
      connection.route?.connectionLabel || connection.route?.label || connection.id
    )

    return profiles
      .map(profile => ({
        profile: String(profile?.name || ''),
        handle: botHandle(profile?.name, profile),
        connection_id: connection.id,
        connection_label: label,
        title: String(profile?.ui_meta?.['hermes-bots']?.title || profile?.display_name || ''),
        description: String(profile?.description || '')
      }))
      .filter(row => row.profile)
  } catch {
    return null
  }
}

/** Last good agent rows per connection id — reused when a fetch blips so a
 *  transient failure never reads as "everyone on that machine went away". */
const relayAgentsCache = new Map()

/** Push every gateway the union roster of agents on the OTHER connections. */
async function syncRelayRosters() {
  if (relayDisposed || relayRosterBusy) {
    return
  }

  relayRosterBusy = true

  try {
    const connections = await relayConnections()

    if (connections.length < 2) {
      return
    }

    const agentsByConnection = new Map()
    await Promise.all(
      connections.map(async connection => {
        const agents = await relayAgentsOn(connection)

        if (agents === null) {
          // Transient fetch failure: reuse the last good rows for this
          // connection (or contribute nothing this cycle) so the pushed
          // roster never drops a live machine's agents — absence from a
          // fresh roster means offline to the gateway-side fail-fast.
          agentsByConnection.set(connection.id, relayAgentsCache.get(connection.id) || [])
        } else {
          relayAgentsCache.set(connection.id, agents)
          agentsByConnection.set(connection.id, agents)
        }
      })
    )

    // Connections gone from profileRoutes are genuinely disconnected — drop
    // their cache so a later reconnect starts from live data.
    const liveIds = new Set(connections.map(connection => connection.id))
    for (const id of [...relayAgentsCache.keys()]) {
      if (!liveIds.has(id)) {
        relayAgentsCache.delete(id)
      }
    }

    await Promise.all(
      connections.map(async connection => {
        const others = []

        for (const [id, agents] of agentsByConnection) {
          if (id !== connection.id) {
            others.push(...agents)
          }
        }

        try {
          await host.requestProfile(connection.route, 'bot_relay.roster.sync', { agents: others })
        } catch {
          // Older backend without the relay RPCs — skip this connection.
        }
      })
    )
  } finally {
    relayRosterBusy = false
  }
}

/** Drain every gateway's outbox and deliver each envelope on the target
 *  connection's own socket; the reply (or error) is posted back to the
 *  sender gateway for its waiter. */
async function drainRelayOutboxes() {
  if (relayDisposed) {
    return
  }

  if (relayDrainBusy) {
    // A push signal raced an in-flight drain. The gateway never re-sends it
    // (monotone signature), so without this flag the envelope would wait out
    // the full poll interval — exactly the latency the push path removes.
    relayDrainRerun = true

    return
  }

  relayDrainBusy = true

  try {
    const connections = await relayConnections()

    // Retention follows the relay-eligible set: with fewer than two
    // connections there is nothing to relay, so nothing stays pinned.
    syncRelayRetention(connections.length >= 2 ? connections : [])

    if (connections.length < 2) {
      return
    }

    const byId = new Map(connections.map(connection => [connection.id, connection]))

    for (const sender of connections) {
      let envelopes = []

      try {
        const res = await host.requestProfile(sender.route, 'bot_relay.outbox.drain', {})
        envelopes = Array.isArray(res?.envelopes) ? res.envelopes : []
      } catch {
        continue
      }

      for (const envelope of envelopes) {
        if (relayDisposed) {
          return
        }

        const envelopeId = String(envelope?.id || '')
        const target = byId.get(String(envelope?.target_connection || ''))
        const postReply = async payload => {
          try {
            await host.requestProfile(sender.route, 'bot_relay.reply', { id: envelopeId, ...payload })
          } catch {
            // Sender gateway unreachable — its waiter times out with guidance.
          }
        }

        if (!envelopeId) {
          continue
        }

        if (!target) {
          await postReply({ error: `connection '${envelope?.target_connection}' is not connected to this Desktop right now` })
          continue
        }

        // Needs-attention hook (#93091 item 3): a delivered background DM is
        // this bot's "good turn"; a classified delivery failure badges it.
        const attentionKey = `${target.id}::${String(envelope?.target_profile || '')}`

        try {
          const res = await host.requestProfile(target.route, 'bot_relay.deliver', {
            profile: String(envelope?.target_profile || ''),
            message: String(envelope?.message || '')
          })
          clearBotAttention(attentionKey)
          await postReply({ reply: String(res?.reply || '') })
        } catch (error) {
          // #93091: bot_relay.deliver classifies the failed turn and ships the
          // typed code in the JSON-RPC error's `data.reason`; forward it into
          // the sender-side reply file so the waiter (and the sending agent)
          // get the machine-readable cause, and prefer it for the badge —
          // classified codes beat free-text re-parsing.
          const reason = String(error?.data?.reason || '').trim()
          noteBotAttention(attentionKey, reason || error?.message || error)
          await postReply({
            error: String(error?.message || error || 'delivery failed'),
            ...(reason ? { reason } : {})
          })
        }
      }
    }
  } finally {
    relayDrainBusy = false

    if (relayDrainRerun && !relayDisposed) {
      // Envelopes signaled mid-drain: schedule one follow-up pass (debounced)
      // instead of leaving them to the interval poll.
      relayDrainRerun = false
      scheduleRelayPushDrain()
    }
  }
}

/** Push-notified drain (#93091): collapse a burst of pending signals into
 *  one drain call ~RELAY_PUSH_DEBOUNCE_MS after the first signal. */
function scheduleRelayPushDrain() {
  if (relayDisposed || typeof setTimeout !== 'function') {
    return
  }

  if (relayPushDebounceTimer !== null) {
    return
  }

  relayPushDebounceTimer = setTimeout(() => {
    relayPushDebounceTimer = null
    void drainRelayOutboxes()
  }, RELAY_PUSH_DEBOUNCE_MS)
}

function startBotRelay() {
  relayDisposed = false

  // Source-shape test harnesses evaluate plugin.js without DOM timers —
  // the relay only runs where a real event loop exists.
  if (typeof setInterval !== 'function' || typeof clearInterval !== 'function') {
    return
  }

  if (relayRosterTimer === null) {
    relayRosterTimer = setInterval(() => void syncRelayRosters(), RELAY_ROSTER_INTERVAL_MS)
    void syncRelayRosters()
  }

  if (relayDrainTimer === null) {
    relayDrainTimer = setInterval(() => void drainRelayOutboxes(), RELAY_DRAIN_INTERVAL_MS)
  }

  // Push path: the gateway change watcher broadcasts when an envelope hits
  // the outbox; drain immediately (debounced) instead of waiting the poll
  // out. Feature-detected — older shells have no host.onEvent — and the 4s
  // poll above stays untouched as the backstop either way.
  if (relayPushUnsub === null && typeof host.onEvent === 'function') {
    relayPushUnsub = host.onEvent('bot_relay.outbox.pending', () => scheduleRelayPushDrain())
  }
}

function stopBotRelay() {
  relayDisposed = true
  // A rerun remembered mid-drain must not leak into the next start —
  // it would fire one stale drain after restart.
  relayDrainRerun = false
  // Unpin every relay-retained socket (#93594): with the relay stopped the
  // pooled entries return to dispose-at-refcount-0 semantics.
  releaseRelayRetention()

  if (relayRosterTimer !== null) {
    clearInterval(relayRosterTimer)
    relayRosterTimer = null
  }

  if (relayDrainTimer !== null) {
    clearInterval(relayDrainTimer)
    relayDrainTimer = null
  }

  if (relayPushDebounceTimer !== null) {
    clearTimeout(relayPushDebounceTimer)
    relayPushDebounceTimer = null
  }

  if (relayPushUnsub !== null) {
    try {
      relayPushUnsub()
    } catch {
      // Disposer from an older shell shape — never break teardown.
    }
    relayPushUnsub = null
  }
}

/** Per-bot appearance + display meta, persisted via ctx.storage:
 *  { [botName]: { shape, color, title } } */
const $botMeta = atom({})

function commitBotMetaV2(storage, snapshot) {
  const commit = botMetaV2Commit.then(async () => {
    if (typeof storage?.remove !== 'function' || typeof storage?.set !== 'function') {
      throw new Error('bot metadata v2 storage is unavailable')
    }

    const [previousSnapshot, previousMarker] = typeof storage.get === 'function'
      ? await Promise.all([
          storage.get(BOT_META_V2_KEY),
          storage.get(BOT_META_MIGRATION_KEY)
        ])
      : [null, null]
    const hasCommittedPrevious = previousMarker === true &&
      previousSnapshot &&
      typeof previousSnapshot === 'object' &&
      !Array.isArray(previousSnapshot)

    try {
      await storage.remove(BOT_META_MIGRATION_KEY)
      await storage.set(BOT_META_V2_KEY, snapshot)
      await storage.set(BOT_META_MIGRATION_KEY, true)
    } catch (error) {
      if (hasCommittedPrevious) {
        try {
          await storage.set(BOT_META_V2_KEY, previousSnapshot)
          await storage.set(BOT_META_MIGRATION_KEY, true)
        } catch {
          await Promise.allSettled([
            storage.remove(BOT_META_MIGRATION_KEY),
            storage.remove(BOT_META_V2_KEY)
          ])
        }
      } else {
        await Promise.allSettled(
          [BOT_META_MIGRATION_KEY, BOT_META_V2_KEY].map(key => storage.remove(key))
        )
      }
      throw error
    }
  })

  botMetaV2Commit = commit.catch(() => undefined)

  return commit
}

function botOwner(owner) {
  if (typeof owner === 'string') {
    const name = owner.trim()
    const route = migratedLocalRoutes.get(name)

    return {
      bot: route ? { name, sourceScoped: true, route } : { name },
      name,
      key: route ? botRouteKey(route) : name,
      route: route || null
    }
  }

  const name = String(owner?.name || '').trim()
  const route = botConnectionRoute(owner)

  return { bot: owner, name, key: route ? botRouteKey(route) : name, route }
}

/** Freshness fence for the server-meta overlay: a roster snapshot fetched
 * before the latest local/server metadata write must not overwrite it. */
const botMetaWriteAt = new Map()

function noteBotMetaWrite(key) {
  botMetaWriteAt.set(key, Date.now())
}

async function saveBotMeta(owner, patch) {
  const { bot, key, name, route } = botOwner(owner)
  const prevMeta = $botMeta.get()[key] || {}
  const next = { ...$botMeta.get(), [key]: { ...prevMeta, ...patch } }
  noteBotMetaWrite(key)
  $botMeta.set(next)

  // Local plugin storage: instant, and the fallback for older gateways.
  let localPersistence = Promise.resolve()
  try {
    const persisted = route || botMetaV2Active
      ? commitBotMetaV2(pluginCtx?.storage, next)
      : Promise.resolve(pluginCtx?.storage?.set?.(BOT_META_V1_KEY, next))
    localPersistence = persisted.catch(() => undefined)
  } catch {
    /* storage unavailable — look persists for this window only */
  }

  // Server-side (source of truth when supported): profile.yaml ui_meta,
  // namespaced under this plugin's id — every client machine sees the same
  // roster. Return the outcome so user-initiated saves can distinguish a
  // cross-machine save from a local-only fallback instead of reporting a
  // false success. Data-URL fields are stripped from ui_meta (64KB cap,
  // rides every profiles.list); the avatar IMAGE goes to the profile asset
  // store instead (profiles.set_asset), which is server-side and uncapped by
  // the list call — so pfps follow the profile across machines too.
  let serverRequest = null
  try {
    const { image, pet, ...rest } = next[key] || {}
    const request = route ? requestForBot(bot, 'profiles.configure', { name, ui_meta: { 'hermes-bots': rest } }) :
      host.request('profiles.configure', { name, ui_meta: { 'hermes-bots': rest } })
    serverRequest = Promise.resolve(request)
  } catch {
    /* older/unavailable gateway — the local fallback remains saved */
  }

  // Avatar image → profile asset store (feature-detected; local storage
  // remains the fallback rendering source on older gateways) — but only when
  // the image actually CHANGED. Every Edit Profile save sends the image key
  // (changed or not); a no-op `clear` from one machine can race another
  // machine's just-pushed avatar and wipe it server-side, and a no-op
  // `data` push re-uploads the full data URL for nothing.
  if ('image' in patch && patch.image !== (prevMeta.image ?? null)) {
    try {
      const req = patch.image
        ? (route
            ? requestForBot(bot, 'profiles.set_asset', { name, asset: 'avatar', data: patch.image })
            : host.request('profiles.set_asset', { name, asset: 'avatar', data: patch.image }))
        : (route
            ? requestForBot(bot, 'profiles.set_asset', { name, asset: 'avatar', clear: true })
            : host.request('profiles.set_asset', { name, asset: 'avatar', clear: true }))
      req.catch(() => undefined)
    } catch {
      /* older gateway */
    }
  }

  // Three-way outcome so callers can tell a REAL remote failure from the
  // documented legacy fallback ("older gateways reject the param shape;
  // that's fine, local wins"):
  //   'persisted'   — gateway confirmed applied.ui_meta === true
  //   'unsupported' — older gateway: request rejected, or response carries
  //                   no `applied` contract at all. Silent local fallback;
  //                   an error toast here would fire on EVERY save forever.
  //   'failed'      — gateway speaks the contract and explicitly reported
  //                   the ui_meta write did NOT apply.
  let serverOutcome = 'unsupported'
  if (serverRequest) {
    try {
      const result = await serverRequest
      if (result?.applied?.ui_meta === true) {
        serverOutcome = 'persisted'
      } else if (result && typeof result === 'object' && result.applied && typeof result.applied === 'object') {
        serverOutcome = 'failed'
      }
    } catch {
      /* older/unavailable gateway — the local fallback remains saved */
    }
    // Re-stamp now that the server write settled: a roster snapshot fetched
    // while profiles.configure was still in flight predates the new ui_meta
    // just as surely as one fetched before the local write.
    noteBotMetaWrite(key)
  }

  await localPersistence

  return { serverPersisted: serverOutcome === 'persisted', serverOutcome }
}

/** Migrate name-keyed appearance state only when the live registry proves
 * there is one local source. A v1 name cannot identify a machine in a
 * multi-source desktop, so the conservative result there is to retain v1 as
 * rollback data and leave remote rows unpainted. */
function hydrateBotMeta(snapshot, remap = null) {
  const next = { ...snapshot }

  for (const [key, meta] of Object.entries($botMeta.get())) {
    const target = remap?.get(key) || key
    next[target] = { ...(next[target] || {}), ...meta }
  }

  $botMeta.set(next)

  return next
}

async function migrateBotMeta(storage = pluginCtx?.storage) {
  let v1 = null
  let v2 = null
  let v2Committed = false

  try {
    ;[v1, v2, v2Committed] = await Promise.all([
      storage?.get?.(BOT_META_V1_KEY),
      storage?.get?.(BOT_META_V2_KEY),
      storage?.get?.(BOT_META_MIGRATION_KEY)
    ])
  } catch {
    return false
  }

  if (v2Committed === true && v2 && typeof v2 === 'object' && !Array.isArray(v2)) {
    hydrateBotMeta(v2)
    botMetaV2Active = true

    return true
  }

  if (!v1 || typeof v1 !== 'object' || Array.isArray(v1) || typeof host.agents !== 'function') {
    if (v1 && typeof v1 === 'object' && !Array.isArray(v1)) {
      hydrateBotMeta(v1)
    }

    return false
  }

  let union
  let routes

  try {
    union = await host.agents()
    routes = typeof host.profileRoutes === 'function' ? await host.profileRoutes() : []
  } catch {
    hydrateBotMeta(v1)

    return false
  }

  const sources = Array.isArray(union?.sources) ? union.sources : []
  const localAgents = (union?.agents || []).filter(agent => agent?.connectionKind === 'local')
  const soleLocal = sources.length === 1
    ? sources[0]?.kind === 'local'
    : sources.length === 0 && localAgents.length > 0 && (union?.agents || []).every(agent => agent?.connectionKind === 'local')

  if (!soleLocal) {
    hydrateBotMeta(v1)

    return false
  }

  const migrated = {}
  const pendingLocalRoutes = new Map()

  for (const [name, meta] of Object.entries(v1)) {
    const route = (routes || []).find(candidate => candidate?.mode === 'local' && candidate?.profile === name) ||
      (() => {
        const agent = localAgents.find(candidate => candidate.profile === name)

        return agent
          ? {
              connectionId: agent.connectionId,
              mode: 'local',
              profile: name,
              targetProfile: agent.targetProfile || name
            }
          : null
      })()

    if (!route?.connectionId) {
      // A missing route makes the topology proof unusable for this key. Keep
      // the v1 record intact rather than guessing a local/remote projection.
      hydrateBotMeta(v1)

      return false
    }

    const captured = {
      connectionId: route.connectionId,
      mode: 'local',
      profile: name,
      targetProfile: route.targetProfile || name
    }
    migrated[botRouteKey(captured)] = meta
    pendingLocalRoutes.set(name, captured)
  }

  const remap = new Map(
    [...pendingLocalRoutes].map(([name, route]) => [name, botRouteKey(route)])
  )
  const hydrated = { ...migrated }

  for (const [key, meta] of Object.entries($botMeta.get())) {
    const target = remap.get(key) || key
    hydrated[target] = { ...(hydrated[target] || {}), ...meta }
  }

  try {
    await commitBotMetaV2(storage, hydrated)
  } catch {
    botMetaV2Active = false
    hydrateBotMeta(v1)

    return false
  }

  migratedLocalRoutes.clear()
  for (const [name, route] of pendingLocalRoutes) {
    migratedLocalRoutes.set(name, route)
  }
  hydrateBotMeta(hydrated)
  botMetaV2Active = true

  return true
}

// ── hidden bots (right-click → Hide Bot) ────────────────────────────────────
// Hiding is a ROSTER-DISPLAY concern only: a hidden bot keeps working,
// remains mentionable, keeps group membership, and any open chat stays open.

/** Session-only view toggle: reveal hidden bots (dimmed) in the roster. */
const $showHiddenBots = atom(false)

function isBotHidden(bot, metaByName) {
  return Boolean(botRosterMeta(bot, metaByName)?.hidden)
}

function isBotPinned(bot, metaByName) {
  return Boolean(botRosterMeta(bot, metaByName)?.pinned)
}

/** Hiding the selected bot re-homes the selection to the next visible owner. */
function fallbackSelectionAfterHide(name) {
  if ($selectedBot.get() !== name) {
    return
  }

  const meta = $botMeta.get()
  const visible = $lastRoster
    .get()
    .filter(bot => botSelectionKey(bot) !== name && !botRosterMeta(bot, meta)?.hidden)

  if (visible.length) {
    $selectedBot.set(botSelectionKey(visible[0]))
    return
  }

  const defaultBot = $lastRoster.get().find(bot => isDefaultBot(bot) && !botRosterMeta(bot, meta)?.hidden)
  if (defaultBot && botSelectionKey(defaultBot) !== name) {
    $selectedBot.set(botSelectionKey(defaultBot))
  } else if (!$lastRoster.get().some(isDefaultBot)) {
    // Legacy sole-local rosters can transiently omit the default row. Keep the
    // historic fallback rather than leaving Routines attached to a hidden bot.
    $selectedBot.set('default')
  }
}

/** One-time reconciliation: Bot Mode sessions are always hidden, but rooms
 *  and Bot Chats created before this policy (or while the old pref was off)
 *  left visible rows behind. On every plugin load, sweep the session ids we
 *  own by id (each group room's member sessions) through the core
 *  session.set_hidden RPC, then run the TITLE-based ownership sweep for
 *  everything else — canonical Bot Chats are identified by name (the
 *  registry row titled "Bot Chat"), so the title sweep is what hides them;
 *  no stored-id pointer is consulted. Idempotent (the DB setter is a no-op
 *  on already-hidden rows) and feature-detected: older gateways lack
 *  session.set_hidden and simply keep the rows visible. */
function hideOwnedBotSessions() {
  const roomEntries = Object.values($groupChats.get()).flatMap(room =>
    Object.entries(room?.sessions || {})
      .map(([key, id]) => {
        if (!id || id === true) {
          return null
        }

        const persisted = room?.sessionOwners?.[key]
        const derived = (room?.members || []).find(member => groupMemberKey(member) === key)
        // Bare keys are legacy local rooms. A source-qualified key without its
        // immutable owner is unsafe: never let it fall through ambient routing.
        const owner = persisted || derived || (!key.includes('::') ? { name: key } : null)

        if (key.includes('::')) {
          const route = owner?.route
          const sourceMarked = owner?.sourceScoped || owner?.remoteSource
          const routeKey = route?.connectionId && route?.profile
            ? `${route.connectionId}::${route.profile}`
            : ''

          if (!sourceMarked || !route?.targetProfile || routeKey !== key) {
            return null
          }
        }

        return owner ? { owner, id, dedupe: `${key}\u0000${id}` } : null
      })
      .filter(Boolean)
  )

  // The same member session can appear in several rooms (and legacy rooms can
  // share ids) — hide each (owner, id) pair exactly once.
  const rooms = [...new Map(roomEntries.map(entry => [entry.dedupe, entry])).values()]

  const known = Promise.all(
    rooms.map(({ owner, id }) =>
      Promise.resolve(requestForBot(owner, 'session.set_hidden', { session_id: id, hidden: true })).catch(
        () => undefined
      )
    )
  )

  return Promise.all([known, sweepBotProfileSessions().catch(() => undefined)])
}

// Titles Bot Mode itself mints for its plumbing sessions. Bot-to-bot CLI
// handoffs (`hermes -p <bot> chat --in ~ -c "Bot Chat" --create-if-missing`)
// create sessions with EXACTLY these titles; the "Group: " prefix is the
// member-session title ensureGroupChatSession has
// used since group chats shipped. Exact/prefix matching is deliberate — a
// user's real conversation inside a bot profile keeps whatever title the
// user gave it and is never touched.
const BOT_MODE_SWEEP_TITLES = new Set(['Bot Chat', 'Agent Inbox'])
const BOT_MODE_SWEEP_MIN_AGE_SECONDS = 5 * 60

function isBotModeSweepTitle(title) {
  const t = String(title || '').trim()
  return BOT_MODE_SWEEP_TITLES.has(t) || t.startsWith('Group: ')
}

function isBotModeSweepCandidate(row, nowSeconds = Date.now() / 1000) {
  const startedAt = Number(row?.started_at)
  return (
    row &&
    row.id &&
    isBotModeSweepTitle(row.title) &&
    Number.isFinite(startedAt) &&
    startedAt > 0 &&
    nowSeconds - startedAt >= BOT_MODE_SWEEP_MIN_AGE_SECONDS
  )
}

/** Ownership-based sweep: the id-based sweep above only covers sessions the
 *  plugin recorded ($botMeta canonical chats, $groupChats member sids), but
 *  Bot Mode sessions are ALSO minted outside the plugin — bot-to-bot CLI
 *  handoffs ("Agent Inbox" / extra "Bot Chat" rows born visible in a bot's
 *  profile) — and those ids the plugin never learns. So: enumerate each
 *  roster bot's OWN profile sessions (only bot profiles — a non-bot profile
 *  is never listed, so its sessions are never touched) and hide any VISIBLE
 *  row whose title is Bot Mode plumbing and whose creation grace period has
 *  elapsed. The grace period protects a new desktop draft while its first-turn
 *  title is pending; after five minutes an unchanged plumbing title is treated
 *  as Bot Mode-owned. session.list supplies epoch seconds; missing, malformed,
 *  millisecond, or future timestamps fail closed and stay visible. session.list
 *  without include_hidden returns only visible rows, which keeps the sweep
 *  naturally idempotent.
 *  Remote-source bots route to their own connection via requestForBot.
 *  Feature-detected + fire-and-forget: older gateways without per-profile
 *  session.list / session.set_hidden simply reject and the sweep no-ops. */
async function sweepBotProfileSessions(nowSeconds = Date.now() / 1000) {
  const cached = $lastRoster.get()
  let roster = Array.isArray(cached) && cached.length ? cached : null

  if (!roster) {
    // Plugin load can run before the Bots pane hydrates $lastRoster — fall
    // back to the active gateway's own profile list (local bots; remote
    // sources get covered by the next sweep once the roster cache exists).
    try {
      const activeBot = { name: String(host.state.profile?.get?.() || 'default').trim() || 'default' }
      const res = await requestForBot(activeBot, 'profiles.list', {})
      roster = Array.isArray(res?.profiles) ? res.profiles : []
    } catch {
      return
    }
  }

  await Promise.all(
    roster.map(async bot => {
      const name = String(bot?.name || '').trim()

      if (!name) {
        return
      }

      try {
        const res = await requestForBot(bot, 'session.list', { profile: name, limit: PROFILE_SESSION_LIST_LIMIT })
        const rows = Array.isArray(res?.sessions) ? res.sessions : []

        await Promise.all(
          rows
            .filter(row => isBotModeSweepCandidate(row, nowSeconds))
            .map(row =>
              Promise.resolve(
                requestForBot(bot, 'session.set_hidden', { session_id: row.id, hidden: true, profile: name })
              ).catch(() => undefined)
            )
        )
      } catch {
        /* older gateway / unreachable source — leave this profile alone */
      }
    })
  )
}

/** Fetch server-side avatars for roster rows flagged has_avatar when the
 *  local cache doesn't already have an image for them. Fire-and-forget. */
const avatarFetchInflight = new Set()

const avatarPushInflight = new Set()

/** Backfill: local meta has art the server lacks -> profiles.set_asset.
 *  Server-side avatars power the inter-agent notice pfp (core #85855) and
 *  cross-machine roster art, so local-only images are a bug, not a state. */
function pushLocalAvatars(roster) {
  for (const bot of roster) {
    const key = botMetaKey(bot)

    if (bot.has_avatar || avatarPushInflight.has(key)) {
      continue
    }

    const image = $botMeta.get()[key]?.image

    if (image && typeof image === 'string' && image.startsWith('data:')) {
      avatarPushInflight.add(key)
      const request = bot.sourceScoped
        ? requestForBot(bot, 'profiles.set_asset', { name: bot.name, asset: 'avatar', data: image })
        : host.request('profiles.set_asset', { name: bot.name, asset: 'avatar', data: image })
      Promise.resolve(request)
        .then(() => queryClient.invalidateQueries({ queryKey: ['hermes-bots', 'roster'] }))
        .catch(() => avatarPushInflight.delete(key))
      continue
    }

    // Vector shape/color face: no image exists anywhere — rasterize the
    // live SVG (tagged data-bot-face) to a PNG and push that, so the
    // inter-agent notices (core #85855/#85888) can show the real pfp.
    const svg = document.querySelector('svg[data-bot-face=' + JSON.stringify(bot.name) + ']')

    if (!svg) {
      continue
    }

    avatarPushInflight.add(key)
    rasterizeSvgToPng(svg, 160)
      .then(png =>
        png
          ? (bot.sourceScoped
              ? requestForBot(bot, 'profiles.set_asset', { name: bot.name, asset: 'avatar', data: png })
              : host.request('profiles.set_asset', { name: bot.name, asset: 'avatar', data: png }))
              .then(() => queryClient.invalidateQueries({ queryKey: ['hermes-bots', 'roster'] }))
          : Promise.reject(new Error('rasterize failed'))
      )
      .catch(() => avatarPushInflight.delete(key))
  }
}

/** Serialize an inline SVG and draw it to a canvas -> PNG data URL. */
function rasterizeSvgToPng(svgEl, size) {
  return new Promise(resolve => {
    try {
      const clone = svgEl.cloneNode(true)
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      clone.setAttribute('width', String(size))
      clone.setAttribute('height', String(size))
      const markup = new XMLSerializer().serializeToString(clone)
      const url = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(markup)
      const img = new Image()

      img.onload = () => {
        try {
          const canvas = document.createElement('canvas')
          canvas.width = size
          canvas.height = size
          canvas.getContext('2d').drawImage(img, 0, 0, size, size)
          resolve(canvas.toDataURL('image/png'))
        } catch {
          resolve(null)
        }
      }
      img.onerror = () => resolve(null)
      img.src = url
    } catch {
      resolve(null)
    }
  })
}

/** The roster backfill draws the live SVG at 160x160. Pets are 96x104
 *  and uploads are 256. Use that to tell a still face-copy from a real picture. */
function isBackfilledFacePng(dataUrl) {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/png;base64,')) {
    return false
  }

  try {
    const bin = atob(dataUrl.slice('data:image/png;base64,'.length).slice(0, 48))
    if (bin.length < 24) {
      return false
    }
    const w = (bin.charCodeAt(16) << 24) | (bin.charCodeAt(17) << 16) | (bin.charCodeAt(18) << 8) | bin.charCodeAt(19)
    const h = (bin.charCodeAt(20) << 24) | (bin.charCodeAt(21) << 16) | (bin.charCodeAt(22) << 8) | bin.charCodeAt(23)
    return w === 160 && h === 160
  } catch {
    return false
  }
}

function pullServerAvatars(roster) {
  pushLocalAvatars(roster)

  for (const bot of roster) {
    const key = botMetaKey(bot)

    if (!bot.has_avatar || avatarFetchInflight.has(key)) {
      continue
    }

    if ($botMeta.get()[key]?.image) {
      continue
    }

    avatarFetchInflight.add(key)
    const assetRequest = bot.sourceScoped
      ? requestForBot(bot, 'profiles.get_asset', { name: bot.name, asset: 'avatar' })
      : host.request('profiles.get_asset', { name: bot.name, asset: 'avatar' })
    Promise.resolve(assetRequest)
      .then(res => {
        if (res?.found && res.data) {
          const current = $botMeta.get()
          const mine = current[key] || {}
          // A 160px raster of the vector face is only for inter-agent
          // notices. Do not park it on the roster or the live face dies.
          if (isBackfilledFacePng(res.data) && mine.imageKind !== 'photo' && !mine.pet) {
            return
          }
          $botMeta.set({ ...current, [key]: { ...mine, image: res.data } })
          persistBotMetaSnapshot($botMeta.get(), Boolean(bot.sourceScoped))
        }
      })
      .catch(() => undefined)
      .finally(() => avatarFetchInflight.delete(key))
  }
}

/** Server ui_meta (per roster row) beats local storage for the compact
 *  fields it carries; local-only fields (avatar image data URL, extracted
 *  pet icon) are PRESERVED — the server copy never includes them, so a
 *  naive replace would wipe a just-saved image avatar on the next roster
 *  paint. When server bot metadata exists, an omitted chat is authoritative
 *  deletion; local still fills all gaps for older gateways with no metadata.
 *
 *  `fetchedAt` (the snapshot's issue time) fences the overlay: a bot whose
 *  local meta was written AFTER the snapshot was fetched is skipped — that
 *  snapshot's ui_meta predates the write, and spreading it back over local
 *  meta resurrects state the user just changed (a disbanded group's
 *  membership reappearing as an empty roster row, a rename reverting, an
 *  unpin undoing). The next roster fetch post-dates the write and overlays
 *  normally, so server truth still gets the last word. */
function mergeServerMeta(roster, fetchedAt = 0) {
  const local = $botMeta.get()
  let changed = false
  const next = { ...local }

  for (const bot of roster) {
    const server = bot.ui_meta?.['hermes-bots']
    if (server && typeof server === 'object') {
      const key = botMetaKey(bot)
      if (fetchedAt && fetchedAt < (botMetaWriteAt.get(key) || 0)) {
        continue
      }
      const mine = next[key] || {}
      const merged = { ...mine, ...server }

      // Local-only fields survive the server overlay.
      if (mine.image) {
        merged.image = mine.image
      }

      // Legacy canonical-chat pointers (meta.chat) are dead: identity is the
      // profile's "Bot Chat" registry row, resolved by name. Drop the key on
      // sight so old ui_meta can never look meaningful again.
      delete merged.chat

      // Canonical multi-group metadata is authoritative for the compatibility
      // scalar too. A server-side `group: null` is represented by omission,
      // so retaining the local scalar would resurrect a membership that another
      // desktop just removed.
      if (
        Array.isArray(server.groups) &&
        Object.prototype.hasOwnProperty.call(mine, 'group') &&
        !Object.prototype.hasOwnProperty.call(server, 'group')
      ) {
        delete merged.group
      }

      if (JSON.stringify(next[key] || null) !== JSON.stringify(merged)) {
        next[key] = merged
        changed = true
      }
    }
  }

  if (changed) {
    $botMeta.set(next)

    // Persist server reconciliation so a relaunch cannot rehydrate stale
    // local fields that the server intentionally removed.
    try {
      persistBotMetaSnapshot(next, roster.some(bot => bot.sourceScoped))
    } catch {
      /* storage unavailable — reconciliation lasts for this window only */
    }
  }
}

/** Clone a bot: profile (config/skills/SOUL/memory via clone_from) + look.
 *  Name is "<base>-2", "-3", … — first free slot against the live roster. */
async function duplicateBot(bot, roster) {
  await ensureBotMetadata(bot)
  const base = bot.name
  const ownerRoute = botConnectionRoute(bot)
  const ownerKey = ownerRoute ? botRouteKey(ownerRoute) : null
  let name = null
  for (let n = 2; n < 100; n++) {
    // Truncate the BASE, never the suffix — slicing the joined string chops
    // the "-2" off a max-length name and the candidate collides with the
    // base forever (#19).
    const suffix = `-${n}`
    const candidate = base.slice(0, 64 - suffix.length) + suffix
    if (!roster.some(b => b.name === candidate && (!ownerKey || botMetaKey(b)?.startsWith(`${ownerRoute.connectionId}::`)))) {
      name = candidate
      break
    }
  }

  if (!name) {
    throw new Error('No free name for the duplicate.')
  }

  await requestForBot(bot, 'profiles.create', {
    name,
    clone_from: base,
    description: bot.description || ''
  })

  // Same look: avatar shape/color/image and a "(copy)" title so the two
  // are tellable apart in the roster until the user renames. Do not copy
  // chat or created. Those belong to the original bot.
  const meta = $botMeta.get()[botMetaKey(bot)]
  if (meta) {
    const { chat, created, ...look } = meta
    await saveBotMeta({ ...bot, name, route: ownerRoute, sourceScoped: Boolean(ownerRoute) }, {
      ...look,
      title: meta.title ? `${meta.title} (copy)` : ''
    })
  }

  return name
}

/** Permanently delete a bot's Hermes profile, then remove plugin-local state
 * that would otherwise leave stale appearance/unread data behind.
 *
 * Prefer the SDK's `host.deleteProfile` when this Desktop build ships it: it
 * routes through the Electron-intercepted REST delete, which tears down the
 * bot's pool backend FIRST and routes the next request away from it. The
 * older `cli.exec` path bypasses that interception, so a backend that the
 * roster's hover pre-warm just woke (right-click hovers the row!) holds the
 * profile dir open — the CLI's rmtree races the live backend and the
 * renderer's socket reconnect respawns it mid-delete, resurrecting the
 * directory (hermes-agent#52279). That is the "can't delete a bot" error. */
async function deleteBot(bot) {
  const route = botConnectionRoute(bot)

  if (isDefaultBot(bot) || String(route?.targetProfile || '').toLowerCase() === 'default') {
    throw new Error('The default profile cannot be deleted.')
  }

  if (typeof host.deleteProfile === 'function') {
    if (route) {
      await host.deleteProfile(route)
    } else {
      await host.deleteProfile(bot.name)
    }
  } else {
    // Older desktop without the SDK verb — source-scoped rows fail closed.
    if (route) {
      throw new Error('Source-scoped profile deletion requires host.deleteProfile.')
    }

    const result = await host.request('cli.exec', {
      argv: ['profile', 'delete', bot.name, '--yes']
    })

    if (result?.blocked || result?.code !== 0) {
      throw new Error(result?.hint || result?.output || `Could not delete profile ${bot.name}.`)
    }
  }

  const meta = { ...$botMeta.get() }
  delete meta[botMetaKey(bot)]
  $botMeta.set(meta)

  try {
    if (route) {
      await commitBotMetaV2(pluginCtx?.storage, meta)
    } else {
      await Promise.resolve(pluginCtx?.storage?.set?.(BOT_META_V1_KEY, meta))
    }
  } catch {
    /* profile is deleted; stale local appearance is harmless if storage fails */
  }

  const unread = { ...$botUnread.get() }
  delete unread[botSelectionKey(bot)]
  $botUnread.set(unread)
  rosterWatermarks.delete(botSelectionKey(bot))
  avatarFetchInflight.delete(botMetaKey(bot))
  avatarPushInflight.delete(botMetaKey(bot))

  if ($selectedBot.get() === botSelectionKey(bot)) {
    $selectedBot.set('default')
  }
  clearSelectedRosterBot(bot)

  if ($openBotChat.get()?.key === botRosterKey(bot)) {
    $openBotChat.set(null)
    syncBotsHomeWorkspace()
  }

  queryClient.invalidateQueries({ queryKey: ROSTER_KEY })

  const activeOwner = focusedRosterOwner($focusedBotOwner.get?.())
  const deletedOwnerIsActive = route
    ? activeOwner?.connectionId === route.connectionId && activeOwner?.name === route.profile
    : activeOwner?.name === bot.name

  if (deletedOwnerIsActive && typeof host.newChat === 'function') {
    host.newChat('default')
  }
}

// ── avatars (shape + color + eyes) ──────────────────────────────────────────

// The original flat shapes. Sigils ('sigil-N') and platonic
// solids remain render-only so any bot that picked one during the experiments
// keeps its look.
// Radix ScrollArea's viewport wraps children in a display:table div that
// sizes to content — unbounded width means `truncate` below it never fires
// and previews run through the panel edge. Scope-limited corrective.
//
// A second Radix quirk bites in the dialogs: the viewport is height:100%,
// which computes to auto when the root only has max-height (no definite
// height anywhere up the chain) — the viewport grows to full content height,
// the root's overflow:hidden clips it, and NOTHING scrolls (#88). Capping
// the viewport itself (inheriting the root's max-height) makes it the real
// scroll container; lists shorter than the cap still shrink to fit.
if (typeof document !== 'undefined' && !document.getElementById('hermes-bots-roster-css')) {
  const style = document.createElement('style')
  style.id = 'hermes-bots-roster-css'
  style.textContent =
    '.hermes-bots-roster [data-radix-scroll-area-viewport] > div {' +
    ' display: block !important; width: 100%; min-width: 0; }' +
    '.hermes-scroll-cap > [data-radix-scroll-area-viewport] { max-height: inherit; }' +
    '@keyframes hermes-bots-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }' +
    '.hermes-bots-pulse { animation: hermes-bots-pulse 1.2s ease-in-out infinite; }'
  document.head.appendChild(style)
}

const AVATAR_SHAPES = ['circle', 'squircle', 'pill', 'triangle', 'hexagon', 'cloud', 'drop']
const AVATAR_PICKER_SHAPES = ['circle', 'blob', 'squircle', 'pill', 'triangle', 'hexagon', 'cloud', 'drop']

/** xorshift PRNG seeded from a string — stable across sessions/platforms. */
function sigilRng(text) {
  let h = 2166136261
  for (const ch of text) {
    h ^= ch.charCodeAt(0)
    h = Math.imul(h, 16777619)
  }
  let state = h >>> 0 || 88675123
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 4294967296
  }
}

/**
 * Angular hermetic sigil: strokes on the left half of a 5-column grid,
 * mirrored right, plus a chance of a diamond ring. Returns SVG path strings.
 */
function sigilGeometry(name, seed) {
  const rng = sigilRng(`${name}::${seed}`)
  const gx = i => 6 + i * 7 // 5 cols: 6..34
  const gy = j => 8 + j * 6 // 5 rows: 8..32
  const strokes = []
  const segments = 4 + Math.floor(rng() * 3)

  for (let k = 0; k < segments; k++) {
    const x1 = Math.floor(rng() * 3) // left half incl. center
    const y1 = Math.floor(rng() * 5)
    const x2 = Math.min(2, Math.max(0, x1 + (rng() > 0.5 ? 1 : -1)))
    const y2 = Math.min(4, Math.max(0, y1 + Math.floor(rng() * 3) - 1))

    strokes.push(`M${gx(x1)} ${gy(y1)} L${gx(x2)} ${gy(y2)}`)
    // mirror (col i → col 4-i)
    strokes.push(`M${gx(4 - x1)} ${gy(y1)} L${gx(4 - x2)} ${gy(y2)}`)

    // occasional cross-tie through the axis for connectedness
    if (rng() > 0.6) {
      strokes.push(`M${gx(x2)} ${gy(y2)} L${gx(4 - x2)} ${gy(y2)}`)
    }
  }

  // spine down the axis grounds every variant
  strokes.push(`M20 ${gy(0)} L20 ${gy(4)}`)

  const ring = rng() > 0.45 ? 'M20 4 L36 20 L20 36 L4 20 Z' : null
  return { strokes: strokes.join(' '), ring }
}

const AVATAR_COLORS = [
  '#f5f5f4', // white
  '#8d6748', // brown
  '#ef4444', // red
  '#f97316', // orange
  '#14b8a6', // teal
  '#38bdf8', // cyan
  '#3b40c8', // royal blue
  '#8b5cf6', // violet
  '#ec4899', // magenta
  '#9ca3af' // silver
]

/** Perceptual luminance — eyes/pupils flip light on dark bodies (ink, oxblood). */
function isDarkColor(hex) {
  try {
    const n = parseInt(hex.slice(1), 16)
    const r = (n >> 16) & 255
    const g = (n >> 8) & 255
    const b = n & 255
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 110
  } catch {
    return false
  }
}

function defaultShapeFor(name) {
  let hash = 0
  for (const ch of name) {
    hash = (hash * 31 + ch.charCodeAt(0)) >>> 0
  }
  return AVATAR_SHAPES[hash % AVATAR_SHAPES.length]
}

// ── blobatar shapes mode (default for new agents) ───────────────────────────
// Deterministic soft-body faces drawn from a string. Shape strings:
//   'blobatar'                — the face follows the bot's NAME (renaming the
//                               bot re-rolls the face, live in the dialog)
//   'blobatar:<seed>'         — seed locked (the 🔒 lock / 🎲 randomize picks)
//   'blobatar:<seed>:<kind>'  — plus one of the ten silhouettes pinned
//   'blobatar::<kind>'        — silhouette pinned, seed still follows the name
// Bot names are slugs (NAME_RE) and generated seeds are base36, so ':' never
// appears inside a segment. Colors come from the library's own name-derived
// palette (contrast-guaranteed) — the classic color swatches don't apply.

const BLOB_KINDS = ['round', 'organic', 'boxy', 'capsule', 'nub', 'cloud', 'droplet', 'hexagon', 'sun', 'triangle']

// Trait positions at the center of each silhouette band. Band thresholds are
// frozen per blobatar major (gen2: 0.22 / 0.48 / 0.60 / 0.70 / 0.79 / 0.86 /
// 0.915 / 0.95 / 0.98).
const BLOB_KIND_TRAIT = {
  round: 0.11, organic: 0.35, boxy: 0.54, capsule: 0.65, nub: 0.745,
  cloud: 0.825, droplet: 0.8875, hexagon: 0.9325, sun: 0.965, triangle: 0.99
}

function isBlobShape(shape) {
  return shape === 'blobatar' || (typeof shape === 'string' && shape.startsWith('blobatar:'))
}

function parseBlobShape(shape, name) {
  const parts = typeof shape === 'string' ? shape.split(':') : []
  const seedPart = parts[1] || ''
  const kind = BLOB_KINDS.includes(parts[2]) ? parts[2] : ''
  return { seed: seedPart || name || 'agent', seedPart, kind }
}

function blobShapeString(seedPart, kind) {
  if (kind) {
    return `blobatar:${seedPart}:${kind}`
  }
  return seedPart ? `blobatar:${seedPart}` : 'blobatar'
}

/** Static SVG markup for a blob face, tagged data-bot-face so the roster's
 *  PNG backfill (pushLocalAvatars → rasterizeSvgToPng) still finds it. */
function blobMarkup(shape, name, size) {
  if (!blobatarSvg) {
    return null
  }

  const { seed, kind } = parseBlobShape(shape, name)
  const opts = { size }

  if (kind) {
    opts.traits = { shape: BLOB_KIND_TRAIT[kind] }
  }

  try {
    return blobatarSvg(seed, opts).replace('<svg ', '<svg data-bot-face=' + JSON.stringify(name) + ' ')
  } catch {
    return null
  }
}

/** The colored body of the avatar (no eyes). Platonic solids are a filled
 *  silhouette + translucent internal edge lines (the projected wireframe);
 *  legacy flat shapes keep their old geometry so stored picks still render. */
function shapeNode(shape, color, botName = 'agent') {
  if (shape.startsWith('sigil-')) {
    const seed = Number(shape.slice(6)) || 0
    const { strokes, ring } = sigilGeometry(botName, seed)
    const sw = { fill: 'none', stroke: color, strokeWidth: 2.2, strokeLinecap: 'round', strokeLinejoin: 'round' }
    return jsxs('g', {
      children: [
        ring ? jsx('path', { d: ring, fill: 'none', stroke: color, strokeWidth: 1.2, opacity: 0.5 }) : null,
        jsx('path', { d: strokes, ...sw })
      ]
    })
  }

  const stroke = { fill: color, stroke: color, strokeWidth: 7, strokeLinejoin: 'round' }
  const edge = { fill: 'none', stroke: 'rgba(0,0,0,0.4)', strokeWidth: 1.4, strokeLinejoin: 'round', strokeLinecap: 'round' }
  const face = { fill: color, stroke: 'rgba(0,0,0,0.4)', strokeWidth: 1.4, strokeLinejoin: 'round' }

  switch (shape) {
    // ── platonic solids ──
    case 'tetrahedron':
      return jsxs('g', {
        children: [
          jsx('path', { d: 'M20 5 L36 33 L4 33 Z', ...face }),
          jsx('path', { d: 'M20 5 L20 25 M4 33 L20 25 M36 33 L20 25', ...edge })
        ]
      })
    case 'cube':
      return jsxs('g', {
        children: [
          jsx('path', { d: 'M20 4 L33 11 L33 29 L20 36 L7 29 L7 11 Z', ...face }),
          jsx('path', { d: 'M7 11 L20 18 L33 11 M20 18 L20 36', ...edge })
        ]
      })
    case 'octahedron':
      return jsxs('g', {
        children: [
          jsx('path', { d: 'M20 3 L36 20 L20 37 L4 20 Z', ...face }),
          jsx('path', { d: 'M4 20 L36 20 M20 3 L20 37', ...edge })
        ]
      })
    case 'dodecahedron':
      return jsxs('g', {
        children: [
          jsx('path', {
            d: 'M20 3 L30 6.2 L36.2 14.7 L36.2 25.3 L30 33.8 L20 37 L10 33.8 L3.8 25.3 L3.8 14.7 L10 6.2 Z',
            ...face
          }),
          jsx('path', {
            d:
              'M20 12 L27.6 17.5 L24.7 26.5 L15.3 26.5 L12.4 17.5 Z ' +
              'M20 12 L20 3 M27.6 17.5 L36.2 14.7 M24.7 26.5 L30 33.8 M15.3 26.5 L10 33.8 M12.4 17.5 L3.8 14.7',
            ...edge
          })
        ]
      })
    case 'icosahedron':
      return jsxs('g', {
        children: [
          jsx('path', { d: 'M20 3 L34.7 11.5 L34.7 28.5 L20 37 L5.3 28.5 L5.3 11.5 Z', ...face }),
          jsx('path', {
            d:
              'M20 11 L27.8 24.5 L12.2 24.5 Z ' +
              'M20 11 L20 3 M20 11 L34.7 11.5 M20 11 L5.3 11.5 ' +
              'M27.8 24.5 L34.7 11.5 M27.8 24.5 L34.7 28.5 M27.8 24.5 L20 37 ' +
              'M12.2 24.5 L5.3 11.5 M12.2 24.5 L5.3 28.5 M12.2 24.5 L20 37',
            ...edge
          })
        ]
      })

    // ── legacy flat shapes (stored picks from earlier versions) ──
    case 'squircle':
      return jsx('rect', { x: 3, y: 3, width: 34, height: 34, rx: 11, fill: color })
    case 'pill':
      return jsx('rect', { x: 2, y: 7, width: 36, height: 26, rx: 13, fill: color })
    case 'triangle':
      return jsx('path', { d: 'M20 5.5 L36 33.5 L4 33.5 Z', ...stroke })
    case 'hexagon':
      return jsx('path', { d: 'M20 3.5 L34.5 11.75 L34.5 28.25 L20 36.5 L5.5 28.25 L5.5 11.75 Z', ...stroke })
    case 'cloud':
      return jsx('path', {
        d: 'M11 32 a7.5 7.5 0 0 1 -1 -14.9 A9.5 9.5 0 0 1 29 12.5 A7 7 0 0 1 30 32 Z',
        fill: color
      })
    case 'drop':
      return jsx('path', { d: 'M20 3 C20 3 6 20 6 27 a14 13.5 0 0 0 28 0 C34 20 20 3 20 3 Z', fill: color })
    default:
      return jsx('circle', { cx: 20, cy: 20, r: 17.5, fill: color })
  }
}

const EYE_Y = {
  // solids: eyes sit on the upper face region, clear of the busiest edges
  tetrahedron: 26,
  cube: 22.5,
  octahedron: 14.5,
  dodecahedron: 20,
  icosahedron: 17.5,
  // legacy
  circle: 17,
  squircle: 17,
  pill: 20,
  triangle: 25,
  hexagon: 17,
  cloud: 22,
  drop: 24
}

// Solids draw eyes slightly tighter so they read as ON a face.
const EYE_X = {
  tetrahedron: [16.5, 23.5],
  cube: [15, 25],
  octahedron: [16, 24],
  dodecahedron: [16.5, 23.5],
  icosahedron: [16.5, 23.5]
}

function cubicAt(p0, p1, p2, p3, t) {
  const u = 1 - t
  return [
    u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
    u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
  ]
}

/** Same outline as the old GitHub drop path, so it stays a fat water drop. */
function sampleDropRing(steps) {
  const pts = []
  const n = Math.max(8, Math.floor(steps / 3))

  for (let i = 0; i < n; i++) {
    pts.push(cubicAt([20, 3], [20, 3], [6, 20], [6, 27], i / n))
  }

  for (let i = 0; i <= n; i++) {
    const t = (i / n) * Math.PI
    pts.push([20 - 14 * Math.cos(t), 27 + 13.5 * Math.sin(t)])
  }

  for (let i = 1; i <= n; i++) {
    pts.push(cubicAt([34, 27], [34, 20], [20, 3], [20, 3], i / n))
  }

  return pts
}

function svgArc(x1, y1, rx, ry, fa, fs, x2, y2) {
  const dx = (x1 - x2) / 2
  const dy = (y1 - y2) / 2
  let rx2 = rx * rx
  let ry2 = ry * ry
  const lam = (dx * dx) / rx2 + (dy * dy) / ry2
  if (lam > 1) {
    const s = Math.sqrt(lam)
    rx *= s
    ry *= s
    rx2 = rx * rx
    ry2 = ry * ry
  }
  const num = rx2 * ry2 - rx2 * dy * dy - ry2 * dx * dx
  const den = rx2 * dy * dy + ry2 * dx * dx
  let sq = Math.sqrt(Math.max(0, num / den))
  if (fa === fs) {
    sq = -sq
  }
  const cx = sq * (rx * dy / ry) + (x1 + x2) / 2
  const cy = sq * (-ry * dx / rx) + (y1 + y2) / 2
  const ang = (ux, uy, vx, vy) => {
    const n = Math.hypot(ux, uy) * Math.hypot(vx, vy) || 1
    let a = Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / n)))
    if (ux * vy - uy * vx < 0) {
      a = -a
    }
    return a
  }
  const theta1 = ang(1, 0, (x1 - cx) / rx, (y1 - cy) / ry)
  let dtheta = ang((x1 - cx) / rx, (y1 - cy) / ry, (x2 - cx) / rx, (y2 - cy) / ry)
  if (!fs && dtheta > 0) {
    dtheta -= Math.PI * 2
  }
  if (fs && dtheta < 0) {
    dtheta += Math.PI * 2
  }
  return { cx, cy, rx, ry, theta1, dtheta }
}

function sampleArc(arc, n) {
  const pts = []
  for (let i = 0; i < n; i++) {
    const th = arc.theta1 + arc.dtheta * (i / n)
    pts.push([arc.cx + arc.rx * Math.cos(th), arc.cy + arc.ry * Math.sin(th)])
  }
  return pts
}

/** Same outline as the old GitHub cloud path: three puffs and a flat floor. */
function sampleCloudRing(steps) {
  const a1 = svgArc(11, 32, 7.5, 7.5, 0, 1, 10, 17.1)
  const a2 = svgArc(10, 17.1, 9.5, 9.5, 0, 1, 29, 12.5)
  const a3 = svgArc(29, 12.5, 7, 7, 0, 1, 30, 32)
  const len1 = Math.abs(a1.dtheta) * a1.rx
  const len2 = Math.abs(a2.dtheta) * a2.rx
  const len3 = Math.abs(a3.dtheta) * a3.rx
  const len4 = 19
  const total = len1 + len2 + len3 + len4
  const n = Math.max(64, steps)
  const n1 = Math.max(8, Math.round(n * len1 / total))
  const n2 = Math.max(10, Math.round(n * len2 / total))
  const n3 = Math.max(10, Math.round(n * len3 / total))
  const n4 = Math.max(4, n - n1 - n2 - n3)
  const pts = []
  pts.push(...sampleArc(a1, n1))
  pts.push(...sampleArc(a2, n2))
  pts.push(...sampleArc(a3, n3))
  for (let i = 0; i < n4; i++) {
    pts.push([30 + (11 - 30) * (i / n4), 32])
  }
  return pts
}

/** Outline of a face in a 40x40 box. Same family as Grok Bot
 *  (blob / squircle / pebble / \u2026) but sampled from formulas, not
 *  a dumped point cloud. */
function sampleFaceRing(shape, steps = 52) {
  const kind = (shape || '').startsWith('sigil-') ? 'circle' : shape

  if (kind === 'drop' || kind === 'teardrop') {
    return sampleDropRing(steps)
  }
  if (kind === 'cloud') {
    return sampleCloudRing(steps)
  }
  const pts = []

  for (let i = 0; i < steps; i++) {
    const a = (i / steps) * Math.PI * 2 - Math.PI / 2
    const c = Math.cos(a)
    const s = Math.sin(a)
    let rx = 16
    let ry = 16
    if (kind === 'circle') {
      rx = ry = 16.2
    } else if (kind === 'blob') {
      rx = ry = 16 + 1.7 * Math.sin(3 * a) + 0.7 * Math.cos(5 * a)
    } else if (kind === 'squircle') {
      const p = 5
      const d = Math.pow(Math.abs(c) ** p + Math.abs(s) ** p, 1 / p) || 1
      rx = ry = 16.2 / d
    } else if (kind === 'pill') {
      const d = Math.pow(Math.abs(c) ** 8 + Math.abs(s / 0.72) ** 8, 1 / 8) || 1
      rx = ry = 16 / d
    } else if (kind === 'triangle' || kind === 'tetrahedron' || kind === 'wedge') {
      const u = (a + Math.PI / 2 + Math.PI * 2) % (Math.PI * 2)
      const sector = (u / (Math.PI * 2 / 3)) % 1
      rx = ry = 13.5 / Math.max(0.42, Math.cos((sector - 0.5) * 1.9))
    } else if (kind === 'hexagon' || kind === 'hex' || kind === 'icosahedron' || kind === 'dodecahedron') {
      const seg = Math.PI / 3
      const hex = Math.cos(seg / 2) / Math.cos(a - seg * Math.round(a / seg))
      rx = ry = 16.2 * hex
    } else if (kind === 'cube' || kind === 'octahedron') {
      const p = 3.1
      const d = Math.pow(Math.abs(c) ** p + Math.abs(s) ** p, 1 / p) || 1
      rx = ry = 16 / d
    } else if (kind === 'pebble') {
      rx = 16.4 * (1.04 - 0.14 * Math.cos(2 * a))
      ry = 15.2 * (1.06 + 0.08 * Math.sin(2 * a))
    } else {
      rx = ry = 16.2
    }

    pts.push([20 + rx * c, 20 + ry * s])
  }

  return pts
}

function projectFacePoint(x, y, turn, tilt, roll) {
  const dx = x - 20
  const dy = y - 20
  const r = (roll * Math.PI) / 180
  const xr = dx * Math.cos(r) - dy * Math.sin(r)
  const yr = dx * Math.sin(r) + dy * Math.cos(r)
  const sx = 0.74 + 0.26 * Math.abs(Math.cos((turn * Math.PI) / 180))
  const sy = 0.8 + 0.2 * Math.abs(Math.cos((tilt * Math.PI) / 180))
  return [20 + xr * sx, 20 + yr * sy]
}

function ringToPath(pts) {
  if (!pts.length) {
    return ''
  }

  let d = `M${pts[0][0].toFixed(2)} ${pts[0][1].toFixed(2)}`

  for (let i = 1; i < pts.length; i++) {
    d += `L${pts[i][0].toFixed(2)} ${pts[i][1].toFixed(2)}`
  }

  return d + 'Z'
}

/** Grok-style pose. thinking/working lean and sway. idle is a small sine. */
function facePose(mood, t) {
  if (mood === 'work') {
    return {
      turn: -11 + Math.sin(t * 0.48) * 8,
      tilt: Math.sin(t * 0.42) * 8 + Math.sin(t * 1.1) * 1.6,
      roll: Math.sin(t * 0.75) * 4.2,
      gazeX: Math.sin(t * 0.55) * 3.6,
      gazeY: -1.6 + Math.sin(t * 0.38) * 2,
      blink: t % 1.45 > 1.26,
      d0: 0.2 + 0.8 * Math.max(0, Math.sin(t * 2.6)),
      d1: 0.2 + 0.8 * Math.max(0, Math.sin(t * 2.6 - 0.7)),
      d2: 0.2 + 0.8 * Math.max(0, Math.sin(t * 2.6 - 1.4))
    }
  }

  return {
    turn: Math.sin(t * 0.5) * 1.5,
    tilt: Math.sin(t * 0.27),
    roll: Math.sin(t * 0.85) * 1.2,
    gazeX: 0,
    gazeY: 0,
    blink: t % 3.2 > 3.02,
    d0: 0,
    d1: 0,
    d2: 0
  }
}

function paintMathFace(svg, t) {
  const mood = svg.getAttribute('data-hb-mood') || 'idle'
  const shape = svg.getAttribute('data-hb-shape') || 'circle'
  const pose = facePose(mood, t)
  const body = svg.querySelector('[data-hb-body]')
  const open = svg.querySelector('[data-hb-open]')
  const shut = svg.querySelector('[data-hb-shut]')
  const el = svg.querySelector('[data-hb-el]')
  const er = svg.querySelector('[data-hb-er]')
  const dots = svg.querySelectorAll('[data-hb-dot]')

  if (body) {
    if (shape === 'cloud') {
      body.setAttribute('d', 'M11 32 a7.5 7.5 0 0 1 -1 -14.9 A9.5 9.5 0 0 1 29 12.5 A7 7 0 0 1 30 32 Z')
    } else {
      const ring = sampleFaceRing(shape).map(([x, y]) => projectFacePoint(x, y, pose.turn, pose.tilt, pose.roll))
      body.setAttribute('d', ringToPath(ring))
    }
  }

  const eyeY = (shape === 'cloud' ? 22 : 17.2) + pose.gazeY
  const eyeL = 15.4 + pose.gazeX
  const eyeR = 24.6 + pose.gazeX

  if (el) {
    el.setAttribute('cx', eyeL)
    el.setAttribute('cy', eyeY)
  }

  if (er) {
    er.setAttribute('cx', eyeR)
    er.setAttribute('cy', eyeY)
  }

  // Catchlights ride the pupils (upper-left offset) — without this they
  // stay at the circle-face position and drift outside e.g. the cloud's
  // lower-set eyes.
  const hl = svg.querySelector('[data-hb-hl-l]')
  const hr = svg.querySelector('[data-hb-hl-r]')

  if (hl) {
    hl.setAttribute('cx', eyeL - 0.6)
    hl.setAttribute('cy', eyeY - 0.7)
  }

  if (hr) {
    hr.setAttribute('cx', eyeR - 0.6)
    hr.setAttribute('cy', eyeY - 0.7)
  }

  if (open) {
    open.setAttribute('opacity', pose.blink ? '0' : '1')
  }

  if (shut) {
    shut.setAttribute('d', `M${eyeL - 2.6} ${eyeY} L${eyeL + 2.6} ${eyeY} M${eyeR - 2.6} ${eyeY} L${eyeR + 2.6} ${eyeY}`)
    shut.setAttribute('opacity', pose.blink ? '1' : '0')
  }

  dots.forEach((dot, i) => {
    const o = i === 0 ? pose.d0 : i === 1 ? pose.d1 : pose.d2
    dot.setAttribute('opacity', String(o))
  })

  svg.style.transform = `rotate(${pose.tilt}deg)`
  svg.style.transformOrigin = '50% 70%'
}

function walkMathFaces(root, acc) {
  if (!root || !root.querySelectorAll) {
    return acc
  }

  root.querySelectorAll('svg[data-hb-math]').forEach(node => acc.push(node))
  root.querySelectorAll('*').forEach(el => {
    if (el.shadowRoot) {
      walkMathFaces(el.shadowRoot, acc)
    }
  })
  return acc
}

function startFaceClock() {
  if (typeof window === 'undefined') {
    return
  }

  if (window.__hbFaceClock) {
    // Already initialized (possibly parked) — make sure it's awake. BotFace
    // renders route here, so a face mounting is what wakes a dormant clock.
    window.__hbFaceClock.wake()

    return
  }

  const t0 = performance.now()
  // A large roster can mount hundreds of faces. Observe the cached nodes so
  // off-screen cards do not consume a full animation frame by themselves.
  let faces = []
  let lastScan = -Infinity
  const visibleFaces = new Set()
  const observedFaces = new Set()
  const observer =
    typeof IntersectionObserver === 'function'
      ? new IntersectionObserver(entries => {
          let becameVisible = false

          for (const entry of entries) {
            if (entry.isIntersecting) {
              visibleFaces.add(entry.target)
              becameVisible = true
            } else {
              visibleFaces.delete(entry.target)
            }
          }

          // A parked clock (no visible faces) resumes when one scrolls in.
          if (becameVisible) {
            window.__hbFaceClock?.wake()
          }
        })
      : null

  const scanFaces = () => {
    faces = walkMathFaces(document, [])

    if (!observer) {
      return
    }

    const currentFaces = new Set(faces)

    for (const svg of observedFaces) {
      if (!currentFaces.has(svg)) {
        observer.unobserve(svg)
        observedFaces.delete(svg)
        visibleFaces.delete(svg)
      }
    }

    for (const svg of faces) {
      if (!observedFaces.has(svg)) {
        observedFaces.add(svg)
        observer.observe(svg)
      }
    }
  }

  // Shared painting body for both scheduling paths: 1Hz document rescans,
  // paint only visible faces (all cached faces when IO is unavailable).
  const paint = now => {
    if (now - lastScan > 1000) {
      scanFaces()
      lastScan = now
    }
    const t = (now - t0) / 1000
    const facesToPaint = observer ? visibleFaces : faces

    for (const svg of facesToPaint) {
      if (svg.isConnected) {
        paintMathFace(svg, t)
      }
    }
  }

  // Nothing worth animating: no faces mounted (BotFace wakes us on the next
  // mount) or none visible (the observer wakes us when one scrolls in).
  const idle = () => faces.length === 0 || (observer && visibleFaces.size === 0)

  const teardownCaches = () => {
    if (observer) {
      observer.disconnect()
    }

    visibleFaces.clear()
    observedFaces.clear()
    faces = []
    delete window.__hbFaceClock
  }

  // Newer desktops: the SDK's budgeted loop owns scheduling (15fps budget,
  // hidden/minimized/unfocused pause, dormancy, teardown). typeof-guarded so
  // older shells and the vm test harness use the hand-rolled path below.
  if (typeof createBudgetedLoop === 'function' && createBudgetedLoop) {
    const loop = createBudgetedLoop(paint, { fps: 15, idleWhen: idle })

    window.__hbFaceClock = {
      stop: () => {
        loop.dispose()
        teardownCaches()
      },
      wake: () => {
        // Faces may have mounted/unmounted while parked — rescan on wake.
        lastScan = -Infinity
        loop.wake()
      }
    }

    return
  }

  // Fallback scheduling for desktops whose SDK predates createBudgetedLoop.
  let lastPaint = -Infinity
  let rafId = 0
  let dormant = false
  let stopped = false

  const tick = now => {
    if (stopped) {
      return
    }

    rafId = 0
    // 15fps is smooth at avatar scale and bounds SVG/DOM churn. The clock
    // still uses rAF so Chromium can pause it when the window is occluded.
    if (!document.hidden && now - lastPaint >= 1000 / 15) {
      paint(now)
      lastPaint = now
    }

    // Park instead of burning frames + 1Hz whole-document shadow walks.
    if (idle()) {
      dormant = true

      return
    }

    rafId = window.requestAnimationFrame(tick)
  }

  const wake = () => {
    if (stopped || !dormant) {
      return
    }

    dormant = false
    // Faces may have mounted/unmounted while parked — rescan on first tick.
    lastScan = -Infinity
    rafId = window.requestAnimationFrame(tick)
  }

  const stop = () => {
    stopped = true

    if (rafId) {
      window.cancelAnimationFrame(rafId)
      rafId = 0
    }

    teardownCaches()
  }

  window.__hbFaceClock = { stop, wake }
  rafId = window.requestAnimationFrame(tick)
}

/** Tear the face clock down (plugin disable/reload) — cancels the animation
 *  frame, disconnects the visibility observer, and drops all cached nodes. */
function stopFaceClock() {
  if (typeof window !== 'undefined' && window.__hbFaceClock) {
    window.__hbFaceClock.stop()
  }
}

/**
 * Live math face. Photos still use <img>. Shape avatars stay SVG so
 * the clock can move them (a baked PNG cannot).
 */
function BotFace({ shape, color, image, size = 36, name = 'agent', mood = 'idle' }) {
  startFaceClock()

  if (image) {
    return jsx('img', {
      src: image,
      alt: '',
      'aria-hidden': true,
      style: { width: size, height: size, borderRadius: '22%', objectFit: 'cover', display: 'block' }
    })
  }

  // Blobatar shapes: the library draws the whole face (body + eyes + its own
  // name-derived palette). Inline SVG via innerHTML so the roster PNG
  // backfill's `svg[data-bot-face=…]` query still finds it; the math clock
  // ignores it (no data-hb-math). Falls back to the legacy math face when the
  // SDK predates the export.
  if (isBlobShape(shape)) {
    const markup = blobMarkup(shape, name, size)

    if (markup) {
      return jsx('span', {
        'aria-hidden': true,
        style: { width: size, height: size, display: 'block', lineHeight: 0 },
        dangerouslySetInnerHTML: { __html: markup }
      })
    }

    // Older SDK without blobatar: legacy deterministic shape from the name.
    shape = defaultShapeFor(name)
  }

  // Sigils are line art (no filled body) — the math clock rebuilds filled
  // outlines, which would turn a stored sigil pick into a blank circle.
  // Keep the legacy static render for them so old picks still draw.
  if (shape.startsWith('sigil-')) {
    const eyes = jsxs('g', {
      children: [
        jsx('circle', { cx: 16, cy: 14, r: 2.4, fill: color }),
        jsx('circle', { cx: 24, cy: 14, r: 2.4, fill: color })
      ]
    })
    return jsxs('svg', {
      'data-bot-face': name,
      viewBox: '0 0 40 40',
      width: size,
      height: size,
      'aria-hidden': true,
      children: [shapeNode(shape, color, name), eyes]
    })
  }

  const working = mood === 'work'
  const eyeFill = isDarkColor(color) ? 'rgba(232,220,195,0.95)' : 'rgba(0,0,0,0.85)'
  // Catchlight contrast follows the pupil, not the body: dark pupils get the
  // white sparkle, light (cream) pupils on dark bodies get a dark one — a
  // white dot on a cream pupil is invisible, which read as "no eye dots" on
  // maroon/ink/oxblood avatars.
  const hlFill = isDarkColor(color) ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.85)'
  const ring = sampleFaceRing(shape)
  const rest = facePose(working ? 'work' : 'idle', 0)
  // Shape-aware initial eye line — the cloud body sits lower, so its eyes
  // (and their catchlights) start at the cloud position instead of jumping
  // there on the first clock paint.
  const eyeY0 = shape === 'cloud' ? 22 : 17.2

  return jsxs('svg', {
    'data-bot-face': name,
    'data-hb-math': '1',
    'data-hb-mood': working ? 'work' : 'idle',
    'data-hb-shape': shape || 'circle',
    viewBox: '0 0 40 44',
    width: size,
    height: size,
    'aria-hidden': true,
    style: { overflow: 'visible', display: 'block' },
    children: [
      jsx('path', {
        'data-hb-body': '1',
        d: shape === 'cloud'
          ? 'M11 32 a7.5 7.5 0 0 1 -1 -14.9 A9.5 9.5 0 0 1 29 12.5 A7 7 0 0 1 30 32 Z'
          : ringToPath(ring),
        fill: color
      }),
      jsxs('g', {
        'data-hb-open': '1',
        children: [
          jsx('ellipse', { 'data-hb-el': '1', cx: 15.4, cy: eyeY0, rx: 2.2, ry: working ? 2.6 : 2.3, fill: eyeFill }),
          jsx('ellipse', { 'data-hb-er': '1', cx: 24.6, cy: eyeY0, rx: 2.2, ry: working ? 2.6 : 2.3, fill: eyeFill }),
          jsx('circle', { 'data-hb-hl-l': '1', cx: 14.8, cy: eyeY0 - 0.7, r: 0.65, fill: hlFill }),
          jsx('circle', { 'data-hb-hl-r': '1', cx: 24, cy: eyeY0 - 0.7, r: 0.65, fill: hlFill })
        ]
      }),
      jsx('path', {
        'data-hb-shut': '1',
        d: `M12.8 ${eyeY0} L18 ${eyeY0} M22 ${eyeY0} L27.2 ${eyeY0}`,
        stroke: eyeFill,
        strokeWidth: 2,
        strokeLinecap: 'round',
        fill: 'none',
        opacity: 0
      }),
      working
        ? jsxs('g', {
            children: [
              jsx('circle', { 'data-hb-dot': '1', cx: 16.4, cy: 41.2, r: 1.15, fill: color, opacity: rest.d0 }),
              jsx('circle', { 'data-hb-dot': '1', cx: 20, cy: 41.2, r: 1.15, fill: color, opacity: rest.d1 }),
              jsx('circle', { 'data-hb-dot': '1', cx: 23.6, cy: 41.2, r: 1.15, fill: color, opacity: rest.d2 })
            ]
          })
        : null
    ]
  })
}

// -- inline MCP setup (per-profile), driven by the mcp.servers.* gateway RPCs --
// Feature-detected: if the gateway predates those RPCs the setup button hides
// and the row falls back to the "run hermes mcp / Settings" hint. profile is
// the target bot's profile name (its config is what we write).

async function mcpRpc(method, params) {
  // Returns { ok, result } or { ok:false, unsupported:true } when the gateway
  // doesn't know the method (older backend) vs a real error.
  try {
    const res = await host.request(method, params)
    return { ok: true, result: res }
  } catch (err) {
    const msg = String((err && err.message) || err || '')
    if (/unknown method/i.test(msg)) {
      return { ok: false, unsupported: true }
    }
    return { ok: false, error: msg }
  }
}

// Probe whether the new lifecycle RPCs exist on this gateway (cached per session).
let _mcpRpcSupported = null
async function mcpSetupSupported() {
  if (_mcpRpcSupported !== null) {
    return _mcpRpcSupported
  }
  const r = await mcpRpc('mcp.servers.list', {})
  _mcpRpcSupported = !(r.ok === false && r.unsupported)
  return _mcpRpcSupported
}

function McpSetupButton({ profile, entry, onDone, ensureProfile }) {
  // entry: { name, requires:[env keys], auth?, fromCatalog, installed }
  // profile may be null at first (New Bot: the profile isn't created yet).
  // ensureProfile() lazily creates it on the first setup action and returns the
  // slug, so OAuth / API-key setup works DURING creation, not only in Edit.
  const [phase, setPhase] = useState('idle') // idle | keys | oauth | busy | done | error
  const [supported, setSupported] = useState(null)
  const [keyValues, setKeyValues] = useState({})
  const [message, setMessage] = useState('')
  const pollRef = useRef(null)
  const profileRef = useRef(profile || null)

  useEffect(() => {
    if (profile) {
      profileRef.current = profile
    }
  }, [profile])

  // Resolve the target profile, creating it on demand for the New Bot flow.
  const resolveProfile = async () => {
    if (profileRef.current) {
      return profileRef.current
    }
    if (ensureProfile) {
      const created = await ensureProfile()
      if (created) {
        profileRef.current = created
      }
      return created
    }
    return null
  }

  useEffect(() => {
    let alive = true
    mcpSetupSupported().then(ok => {
      if (alive) setSupported(ok)
    })
    return () => {
      alive = false
      if (pollRef.current) {
        clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [])

  const isOAuth = (entry.auth || '').toLowerCase() === 'oauth'
  const requires = entry.requires || []

  const beginKeys = async () => {
    // Ensure the server exists in the target profile first (add from catalog).
    setPhase('busy')
    setMessage('')
    const profile = await resolveProfile()
    if (!profile) {
      setPhase('idle')
      return
    }
    if (entry.fromCatalog && !entry.installed) {
      const add = await mcpRpc('mcp.servers.add', { profile, name: entry.name, preset: entry.name })
      if (!add.ok) {
        setPhase('error')
        setMessage(add.error || 'Could not add server')
        return
      }
    }
    setPhase(isOAuth ? 'oauth' : 'keys')
  }

  const submitKeys = async () => {
    setPhase('busy')
    const profile = profileRef.current
    if (!profile) {
      setPhase('error')
      setMessage('No target profile')
      return
    }
    for (const k of requires) {
      const val = (keyValues[k] || '').trim()
      if (!val) {
        continue
      }
      const r = await mcpRpc('mcp.servers.set_api_key', { profile, name: entry.name, env_var: k, value: val })
      if (!r.ok) {
        setPhase('error')
        setMessage(r.error || ('Failed to set ' + k))
        return
      }
    }
    // Verify via test.
    const t = await mcpRpc('mcp.servers.test', { profile, name: entry.name })
    if (t.ok && t.result && (t.result.ok || (t.result.result && t.result.result.ok))) {
      setPhase('done')
      host.notify({ kind: 'success', message: entry.name + ' configured' })
      onDone && onDone()
    } else {
      setPhase('error')
      setMessage((t.result && (t.result.error || (t.result.result && t.result.result.error))) || 'Server test failed after setup')
    }
  }

  const beginOAuth = async () => {
    // A second click (retry, impatient double-click) must not orphan the
    // previous poll interval — an overwritten pollRef leaks a 2s poller that
    // runs until unmount and can flip phase from a stale OAuth session.
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    setPhase('busy')
    setMessage('')
    const profile = await resolveProfile()
    if (!profile) {
      setPhase('idle')
      return
    }
    if (entry.fromCatalog && !entry.installed) {
      const add = await mcpRpc('mcp.servers.add', { profile, name: entry.name, preset: entry.name })
      if (!add.ok) {
        setPhase('error')
        setMessage(add.error || 'Could not add server')
        return
      }
    }
    const start = await mcpRpc('mcp.servers.oauth.start', { profile, name: entry.name })
    const payload = start.result && (start.result.result || start.result)
    const authUrl = payload && (payload.auth_url || payload.verification_url)
    const sessionId = payload && payload.session_id
    if (!start.ok || !authUrl || !sessionId) {
      setPhase('error')
      setMessage((start.error) || 'Could not start OAuth')
      return
    }
    // Open the auth URL in the native browser, same as provider OAuth.
    try {
      if (host.openExternal) {
        host.openExternal(authUrl)
      } else if (typeof window !== 'undefined' && window.hermesDesktop && window.hermesDesktop.openExternal) {
        window.hermesDesktop.openExternal(authUrl)
      } else {
        window.open(authUrl, '_blank')
      }
    } catch {
      /* fall through to poll; user can open the URL from the toast */
    }
    setPhase('oauth')
    setMessage('Complete sign-in in your browser...')
    pollRef.current = setInterval(async () => {
      const poll = await mcpRpc('mcp.servers.oauth.poll', { profile, name: entry.name, session_id: sessionId })
      const pd = poll.result && (poll.result.result || poll.result)
      const status = pd && pd.status
      if (status === 'approved') {
        clearInterval(pollRef.current)
        pollRef.current = null
        setPhase('done')
        host.notify({ kind: 'success', message: entry.name + ' authenticated' })
        onDone && onDone()
      } else if (status === 'error') {
        clearInterval(pollRef.current)
        pollRef.current = null
        setPhase('error')
        setMessage((pd && pd.error_message) || 'OAuth failed')
      }
    }, 2000)
  }

  if (supported === false) {
    return jsx('span', {
      className: 'ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)',
      children: 'needs setup (' + requires.join(', ') + ') \u2014 restart the gateway to enable in-app setup'
    })
  }
  if (phase === 'done') {
    return jsx('span', { className: 'ml-1.5 text-[0.65rem] text-(--ui-success,#22c55e)', children: 'set up \u2713' })
  }
  if (phase === 'keys') {
    return jsxs('div', {
      className: 'mt-1 grid gap-1',
      children: [
        ...requires.map(k =>
          jsx(Input, {
            key: k,
            type: 'password',
            className: 'h-6 text-[0.7rem]',
            placeholder: k,
            value: keyValues[k] || '',
            onChange: e => setKeyValues(prev => ({ ...prev, [k]: e.target.value }))
          }, k)
        ),
        jsxs('div', {
          className: 'flex gap-1',
          children: [
            jsx(Button, { size: 'xs', variant: 'secondary', onClick: () => void submitKeys(), children: 'Save & test' }),
            jsx(Button, { size: 'xs', variant: 'ghost', onClick: () => setPhase('idle'), children: 'Cancel' })
          ]
        })
      ]
    })
  }
  if (phase === 'oauth') {
    return jsx('span', { className: 'ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)', children: message || 'Authorizing\u2026' })
  }
  if (phase === 'busy') {
    return jsx('span', { className: 'ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)', children: 'Working\u2026' })
  }
  if (phase === 'error') {
    return jsxs('span', {
      className: 'ml-1.5 text-[0.65rem] text-(--ui-danger,#ef4444)',
      children: [(message || 'Setup failed') + ' ', jsx('button', { className: 'underline', onClick: () => setPhase('idle'), children: 'retry' })]
    })
  }
  // idle
  return jsx('button', {
    className: 'ml-1.5 text-[0.65rem] text-(--ui-accent,#4f9cf9) underline',
    onClick: () => void (isOAuth ? beginOAuth() : beginKeys()),
    children: isOAuth ? 'Sign in\u2026' : 'Set up\u2026'
  })
}

function botAppearance(name, meta) {
  // The primary profile is literally named "default"; the SDK's profileColor
  // can hand it a near-black that renders as an ugly black square, and any
  // auto-seeded color in local bot-meta would otherwise stick. Give the
  // primary a nice fixed generic look (a friendly violet squircle). A user's
  // EXPLICIT customization still wins: an uploaded/generated/pet image, or a
  // shape/color they set via the editor (tracked by meta.custom === true).
  const isPrimary = (name || '').trim().toLowerCase() === 'default'
  const userCustomized = Boolean(meta?.custom)
  if (isPrimary && !userCustomized) {
    return { shape: 'squircle', color: '#8b5cf6', image: meta?.image || null }
  }
  return {
    shape: meta?.shape || defaultShapeFor(name),
    color: meta?.color || profileColor(name),
    image: meta?.image || null
  }
}

// ── image avatars: upload from device + generate via image.generate ─────────

/** Downscale to a small square so plugin storage stays light. */
function normalizeAvatarImage(dataUrl, edge = 256) {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = edge
        canvas.height = edge
        const ctx2d = canvas.getContext('2d')
        const side = Math.min(img.width, img.height)
        ctx2d.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, edge, edge)
        resolve(canvas.toDataURL('image/png'))
      } catch {
        resolve(dataUrl)
      }
    }
    img.onerror = () => resolve(dataUrl)
    img.src = dataUrl
  })
}

function pickImageFromDevice() {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/png,image/jpeg,image/webp,image/gif'
    input.onchange = () => {
      const file = input.files?.[0]

      if (!file) {
        return resolve(null)
      }

      if (file.size > 15_000_000) {
        host.notify({ kind: 'error', message: 'Image too large (max 15MB).' })
        return resolve(null)
      }

      const reader = new FileReader()
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null)
      reader.onerror = () => resolve(null)
      reader.readAsDataURL(file)
    }
    input.click()
  })
}

// ── group-chat attachments: pick/paste/drop files the room's members see ────

/** Classify a picked file for the group-attachment pipeline. */
function groupAttachmentKind(file) {
  if (/^image\//.test(file.type || '')) {
    return 'image'
  }

  if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name || '')) {
    return 'pdf'
  }

  return 'file'
}

/** File objects → [{ name, data, kind }] (data URLs), oversized files skipped
 *  with a toast. Images are downscaled; PDFs and other files ride as raw data
 *  URLs for the gateway's pdf.attach / file.attach staging. Shared by the
 *  picker button, the composer paste handler, and room drag & drop. */
async function filesToGroupAttachments(files) {
  const picked = []

  for (const file of [...(files || [])]) {
    if (!file) {
      continue
    }

    if (file.size > 15_000_000) {
      host.notify({ kind: 'error', message: `${file.name || 'attachment'}: too large (max 15MB).` })
      continue
    }

    const data = await new Promise(done => {
      const reader = new FileReader()
      reader.onload = () => done(typeof reader.result === 'string' ? reader.result : null)
      reader.onerror = () => done(null)
      reader.readAsDataURL(file)
    })

    if (!data) {
      continue
    }

    const kind = groupAttachmentKind(file)
    picked.push({
      name: file.name || (kind === 'image' ? 'pasted image' : 'attachment'),
      data: kind === 'image' ? await normalizeGroupAttachment(data) : data,
      kind
    })
  }

  return picked
}

/** Multi-file picker for the group composer — any file type; kind decides
 *  the staging RPC. Resolves to [{ name, data, kind }]. */
function pickGroupAttachments() {
  return new Promise(resolve => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.onchange = () => resolve(filesToGroupAttachments(input.files))
    input.click()
  })
}

/** Bound a group attachment's long edge so room logs (persisted with the
 *  plugin's other durable state) stay light while screenshots keep enough
 *  resolution for vision models to read text. No-op for small images or
 *  anything the canvas can't decode. */
function normalizeGroupAttachment(dataUrl, maxEdge = 1568) {
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => {
      try {
        const long = Math.max(img.width, img.height)

        if (!long || long <= maxEdge) {
          return resolve(dataUrl)
        }

        const scale = maxEdge / long
        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.round(img.width * scale))
        canvas.height = Math.max(1, Math.round(img.height * scale))
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
        resolve(canvas.toDataURL('image/png'))
      } catch {
        resolve(dataUrl)
      }
    }
    img.onerror = () => resolve(dataUrl)
    img.src = dataUrl
  })
}

/** Cached probe: does the gateway have an image backend? A `false` answer
 *  is re-checked on every dialog open — the gateway may have been restarted
 *  (picking up image.generate) or a backend enabled since the last probe.
 *  Only `true` is sticky. */
const $imagenAvailable = atom(null)
let imagenProbeInflight = null

function probeImagen() {
  if (imagenProbeInflight) {
    return imagenProbeInflight
  }

  imagenProbeInflight = host
    .request('image.generate', { probe: true })
    .then(res => $imagenAvailable.set(Boolean(res?.available)))
    .catch(() => $imagenAvailable.set(false))
    .finally(() => {
      imagenProbeInflight = null
    })

  return imagenProbeInflight
}

async function generateAvatarImage(bot, title, description) {
  const who = [title || bot, description].filter(Boolean).join(' — ')
  const res = await host.request('image.generate', {
    prompt:
      `Cute minimal robot avatar for an AI agent named "${who}". ` +
      'Friendly simple mascot face, bold flat vector style, solid color background, centered, no text.',
    aspect_ratio: 'square'
  })

  if (!res?.success) {
    throw new Error(res?.error || 'generation failed')
  }

  // image_data (data URL) works over local AND remote gateways; the raw
  // backend URL is the fallback when the gateway couldn't inline it.
  return res.image_data || res.image
}

/** Shape grid + color swatches, shared by Edit Profile and New Bot.
 *  Layout uses inline grid styles — arbitrary Tailwind classes like
 *  `grid-cols-7` are NOT in the app's precompiled CSS, which collapsed
 *  this into a single vertical column. */
function AvatarPicker({ shape, color, image, onShape, onColor, onImage, generateSeed }) {
  const pickerName = generateSeed?.name || 'agent'
  const imagen = useValue($imagenAvailable)
  const [tab, setTab] = useState('bot')
  const [describe, setDescribe] = useState('')
  const [genBusy, setGenBusy] = useState(false)

  if (imagen === null) {
    void probeImagen()
  }

  // Re-check a stale "unavailable" whenever the user lands on the Generate
  // tab — the gateway may have restarted with image.generate since.
  const goTab = id => {
    setTab(id)

    if (id === 'generate' && $imagenAvailable.get() === false) {
      $imagenAvailable.set(null)
      void probeImagen()
    }
  }

  const upload = async () => {
    const raw = await pickImageFromDevice()

    if (raw) {
      onImage(await normalizeAvatarImage(raw))
    }
  }

  const generate = async () => {
    if (genBusy) {
      return
    }

    setGenBusy(true)

    try {
      const custom = describe.trim()
      const img = custom
        ? await (async () => {
            const res = await host.request('image.generate', {
              prompt: `${custom}. Avatar for an AI agent: centered, bold flat vector style, solid color background, no text.`,
              aspect_ratio: 'square'
            })

            if (!res?.success) {
              throw new Error(res?.error || 'generation failed')
            }

            return res.image_data || res.image
          })()
        : await generateAvatarImage(generateSeed?.name || 'agent', generateSeed?.title, generateSeed?.description)

      if (img) {
        onImage(await normalizeAvatarImage(img))
      }
    } catch (err) {
      host.notifyError(err, 'Avatar generation failed')
    } finally {
      setGenBusy(false)
    }
  }

  const tabButton = (id, label) =>
    jsx(
      'button',
      {
        type: 'button',
        className: cn(
          'rounded-full px-3 py-1 text-xs font-medium transition-colors',
          tab === id
            ? 'bg-(--chrome-action-hover) text-foreground'
            : 'text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)'
        ),
        onClick: () => goTab(id),
        children: label
      },
      id
    )

  return jsxs('div', {
    className: 'grid justify-items-center gap-3',
    children: [
      // Tab pills: Bot | Generate | Upload | Pet
      jsxs('div', {
        className: 'flex items-center gap-1',
        children: [tabButton('bot', 'Bot'), tabButton('generate', 'Generate'), tabButton('upload', 'Upload'), tabButton('pet', 'Pet')]
      }),

      image && tab !== 'generate'
        ? jsx(Button, {
            type: 'button',
            variant: 'ghost',
            size: 'sm',
            onClick: () => onImage(null),
            children: 'Remove image — use shape'
          })
        : null,

      tab === 'bot'
        ? isBlobShape(shape) && blobatarSvg
          ? (() => {
              const { seedPart, kind } = parseBlobShape(shape, pickerName)
              const locked = Boolean(seedPart)
              return jsxs('div', {
                className: 'grid justify-items-center gap-3',
                children: [
                  // Silhouette pins: Auto (name decides) + the six blob kinds.
                  jsx('div', {
                    style: {
                      display: 'grid',
                      gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
                      gap: '6px',
                      justifyItems: 'center'
                    },
                    children: ['', ...BLOB_KINDS].map(k =>
                      jsx(
                        'button',
                        {
                          type: 'button',
                          title: k || 'Auto — the name decides',
                          className: cn(
                            'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                            k === kind && !image && 'ring-1 ring-(--ui-accent)'
                          ),
                          style: { width: 44, height: 44 },
                          onClick: () => {
                            onImage(null)
                            onShape(blobShapeString(seedPart, k))
                          },
                          children: k
                            ? jsx(BotFace, { shape: blobShapeString(seedPart, k), color, size: 32, name: pickerName })
                            : jsx('span', { className: 'text-[0.6rem] text-(--ui-text-tertiary)', children: 'Auto' })
                        },
                        k || 'auto'
                      )
                    )
                  }),
                  jsxs('div', {
                    className: 'flex items-center gap-1',
                    children: [
                      jsxs(Button, {
                        type: 'button',
                        variant: 'ghost',
                        size: 'sm',
                        onClick: () => {
                          onImage(null)
                          onShape(blobShapeString(Math.random().toString(36).slice(2, 10), kind))
                        },
                        children: [jsx(Codicon, { name: 'refresh', className: 'mr-1 text-[0.8rem]' }), 'Randomize']
                      }),
                      jsxs(Button, {
                        type: 'button',
                        variant: 'ghost',
                        size: 'sm',
                        title: locked
                          ? 'Unlock — the face follows the agent\u2019s name again'
                          : 'Keep this exact face even if the name changes',
                        onClick: () => onShape(blobShapeString(locked ? '' : pickerName, kind)),
                        children: [
                          jsx(Codicon, { name: locked ? 'unlock' : 'lock', className: 'mr-1 text-[0.8rem]' }),
                          locked ? 'Unlock' : 'Lock face'
                        ]
                      })
                    ]
                  }),
                  jsx('div', {
                    className: 'text-center text-[0.65rem] text-(--ui-text-quaternary)',
                    children: locked ? 'Face locked — renaming won\u2019t change it.' : 'Face follows the name.'
                  }),
                  jsx(Button, {
                    type: 'button',
                    variant: 'ghost',
                    size: 'sm',
                    className: 'text-(--ui-text-tertiary)',
                    onClick: () => onShape(defaultShapeFor(pickerName)),
                    children: 'Classic shapes'
                  })
                ]
              })
            })()
          : jsxs('div', {
            className: 'grid justify-items-center gap-3',
            children: [
              jsx('div', {
                style: {
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
                  gap: '6px',
                  justifyItems: 'center'
                },
                children: (blobatarSvg ? ['blobatar', ...AVATAR_PICKER_SHAPES] : AVATAR_PICKER_SHAPES).map(s =>
                  jsx(
                    'button',
                    {
                      type: 'button',
                      title: s === 'blobatar' ? 'Blob face — drawn from the agent\u2019s name' : undefined,
                      className: cn(
                        'flex items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover)',
                        s === shape && !image && 'ring-1 ring-(--ui-accent)'
                      ),
                      style: { width: 44, height: 44 },
                      onClick: () => {
                        onImage(null)
                        onShape(s)
                      },
                      children: jsx(BotFace, { shape: s, color, size: 32, name: pickerName })
                    },
                    s
                  )
                )
              }),
              jsx('div', {
                style: {
                  display: 'grid',
                  gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
                  gap: '8px',
                  justifyItems: 'center'
                },
                children: AVATAR_COLORS.map(c =>
                  jsx(
                    'button',
                    {
                      type: 'button',
                      className: cn(
                        'rounded-full transition-transform hover:scale-110',
                        c === color && 'ring-2 ring-(--ui-accent) ring-offset-1 ring-offset-(--ui-bg, transparent)'
                      ),
                      style: { width: 22, height: 22, backgroundColor: c },
                      onClick: () => onColor(c)
                    },
                    c
                  )
                )
              })
            ]
          })
        : null,

      tab === 'generate'
        ? imagen
          ? jsxs('div', {
              className: 'grid w-full gap-2',
              children: [
                jsx(Textarea, {
                  className: 'min-h-16 text-xs',
                  placeholder: 'Describe your avatar…',
                  value: describe,
                  onChange: event => setDescribe(event.target.value)
                }),
                jsxs(Button, {
                  type: 'button',
                  variant: 'secondary',
                  className: 'w-full justify-center',
                  disabled: genBusy,
                  onClick: generate,
                  children: [
                    genBusy
                      ? jsx(GlyphSpinner, { spinner: 'breathe', className: 'mr-1 text-[0.8rem]' })
                      : jsx(Codicon, { name: 'sparkle', className: 'mr-1 text-[0.8rem]' }),
                    genBusy ? 'Generating…' : 'Generate'
                  ]
                }),
                describe.trim()
                  ? null
                  : jsx('div', {
                      className: 'text-center text-[0.65rem] text-(--ui-text-quaternary)',
                      children: 'Leave blank to generate from the agent\u2019s name and description.'
                    })
              ]
            })
          : jsx('div', {
              className: 'px-2 py-3 text-center text-xs leading-5 text-(--ui-text-tertiary)',
              children:
                imagen === false
                  ? 'No image model available. If you just enabled one (or updated Hermes), restart the gateway: Ctrl+K → "Restart gateway".'
                  : 'Checking image backend…'
            })
        : null,

      tab === 'upload'
        ? jsxs(Button, {
            type: 'button',
            variant: 'secondary',
            className: 'w-full justify-center',
            onClick: upload,
            children: [jsx(Codicon, { name: 'device-camera', className: 'mr-1 text-[0.8rem]' }), 'Choose an image…']
          })
        : null,

      tab === 'pet' ? jsx(PetTab, { image, onImage }) : null
    ]
  })
}

// ── pet tab: attach a petdex companion that lives beside the avatar ─────────

// A petdex "spritesheet" is the FULL animation sheet (1536×1872 webp, ~2MB;
// 8×9 grid of 192×208 frames). Using it as an <img> both downloads megabytes
// per tile and shows the whole sheet squashed. Extract frame 0 once per slug
// via canvas, downscale to 96px, and cache the data URL. Concurrency-capped
// so opening the tab doesn't fire dozens of 2MB fetches at once.
const PET_FRAME_W = 192
const PET_FRAME_H = 208
const petFrameCache = new Map()
let petFetchActive = 0
const petFetchQueue = []

function pumpPetQueue() {
  while (petFetchActive < 4 && petFetchQueue.length) {
    const job = petFetchQueue.shift()
    petFetchActive++
    job().finally(() => {
      petFetchActive--
      pumpPetQueue()
    })
  }
}

function petFrameIcon(spriteUrl) {
  if (!spriteUrl) {
    return Promise.resolve(null)
  }

  if (!petFrameCache.has(spriteUrl)) {
    petFrameCache.set(
      spriteUrl,
      new Promise(resolve => {
        petFetchQueue.push(async () => {
          try {
            const resp = await fetch(spriteUrl, { signal: AbortSignal.timeout(15000) })
            const blob = await resp.blob()
            // Crop frame 0 during decode — never materialize the full sheet.
            const bitmap = await createImageBitmap(blob, 0, 0, PET_FRAME_W, PET_FRAME_H)
            const canvas = document.createElement('canvas')
            canvas.width = 96
            canvas.height = 104
            canvas.getContext('2d').drawImage(bitmap, 0, 0, 96, 104)
            bitmap.close()
            resolve(canvas.toDataURL('image/png'))
          } catch {
            petFrameCache.delete(spriteUrl)
            resolve(null)
          }
        })
        pumpPetQueue()
      })
    )
  }

  return petFrameCache.get(spriteUrl)
}

/** One pet tile image: frame 0 only, resolved lazily through the cache. */
function PetThumb({ spriteUrl, size = 40 }) {
  const [icon, setIcon] = useState(null)

  useEffect(() => {
    let alive = true
    petFrameIcon(spriteUrl).then(url => {
      if (alive) {
        setIcon(url)
      }
    })
    return () => {
      alive = false
    }
  }, [spriteUrl])

  if (!icon) {
    return jsx('div', {
      style: { width: size, height: size, borderRadius: 6, background: 'var(--chrome-action-hover, rgba(255,255,255,0.06))' }
    })
  }

  return jsx('img', {
    src: icon,
    alt: '',
    style: { width: size, height: size, objectFit: 'contain', imageRendering: 'pixelated', borderRadius: 6 }
  })
}

function PetTab({ image, onImage }) {
  // Selection is dialog-local: committed by the dialog's Save like any
  // uploaded/generated image (a direct meta write here gets clobbered by
  // Save's own image state).
  const [selectedSlug, setSelectedSlug] = useState(null)
  const { data, isLoading } = useQuery({
    queryKey: [ID, 'pet-gallery'],
    queryFn: () => host.request('pet.gallery', {}),
    staleTime: 300000
  })
  const [query, setQuery] = useState('')
  // Windowed rendering: the gallery is 4500+ pets — mounting an <img> per pet
  // froze the dialog. Render `limit` at a time and grow on scroll-to-bottom.
  const [limit, setLimit] = useState(24)
  const pets = data?.pets ?? []

  if (isLoading) {
    return jsx('div', {
      className: 'flex justify-center py-4',
      children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
    })
  }

  if (!pets.length) {
    return jsx('div', {
      className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
      children: 'No pets in the petdex gallery. Run `hermes pets` to explore.'
    })
  }

  const q = query.trim().toLowerCase()
  const filtered = q
    ? pets.filter(pet => (pet.displayName || '').toLowerCase().includes(q) || (pet.slug || '').includes(q))
    : pets
  // Installed and curated pets surface first — they're the likeliest picks.
  const ranked = filtered.slice().sort((a, b) => {
    const rank = pet => (pet.installed ? 0 : pet.curated ? 1 : 2)
    return rank(a) - rank(b)
  })
  const visible = ranked.slice(0, limit)

  const onScroll = event => {
    const el = event.currentTarget

    if (el.scrollTop + el.clientHeight >= el.scrollHeight - 120 && limit < ranked.length) {
      setLimit(prev => Math.min(prev + 24, ranked.length))
    }
  }

  return jsxs('div', {
    className: 'grid w-full gap-2',
    children: [
      jsx('div', {
        className: 'text-center text-[0.65rem] text-(--ui-text-quaternary)',
        children: 'Pick a pet as this agent’s profile picture.'
      }),
      jsx(Input, {
        className: 'h-7 text-xs',
        placeholder: `Search ${pets.length} pets…`,
        value: query,
        onChange: event => {
          setQuery(event.target.value)
          setLimit(24)
        }
      }),
      image && selectedSlug
        ? jsx(Button, {
            type: 'button',
            variant: 'ghost',
            size: 'sm',
            className: 'justify-center',
            onClick: () => {
              setSelectedSlug(null)
              onImage(null)
            },
            children: 'Remove — back to shape avatar'
          })
        : null,
      filtered.length === 0
        ? jsx('div', {
            className: 'py-3 text-center text-xs text-(--ui-text-quaternary)',
            children: 'No pets match.'
          })
        : jsxs('div', {
            onScroll,
            style: { maxHeight: 220, overflowY: 'auto' },
            children: [
              jsx('div', {
                style: {
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                  gap: '6px'
                },
                children: visible.map(pet =>
                  jsxs(
                    'button',
                    {
                      type: 'button',
                      className: cn(
                        'grid justify-items-center gap-1 rounded-md p-1.5 transition-colors hover:bg-(--chrome-action-hover)',
                        selectedSlug === pet.slug && 'ring-1 ring-(--ui-accent)'
                      ),
                      onClick: () => {
                        // The pet IS the profile picture: extract frame 0
                        // and hand it to the dialog as the avatar image.
                        // Persisted when the user hits Save.
                        setSelectedSlug(pet.slug)
                        void petFrameIcon(pet.spritesheetUrl).then(icon => {
                          if (icon) {
                            onImage(icon)
                          } else {
                            setSelectedSlug(null)
                            host.notify({ kind: 'error', message: 'Could not load that pet — try another.' })
                          }
                        })
                      },
                      children: [
                        jsx(PetThumb, { spriteUrl: pet.spritesheetUrl, size: 40 }),
                        jsx('span', {
                          className: 'w-full truncate text-center text-[0.6rem] text-(--ui-text-tertiary)',
                          children: pet.displayName
                        })
                      ]
                    },
                    pet.slug
                  )
                )
              }),
              limit < ranked.length
                ? jsx('div', {
                    className: 'py-2 text-center text-[0.65rem] text-(--ui-text-quaternary)',
                    children: `Scroll for more (${limit} of ${ranked.length})`
                  })
                : null
            ]
          })
    ]
  })
}

// ── data ─────────────────────────────────────────────────────────────────────

/** True once profiles.list reports the backend injects the bot-to-bot
 *  protocol into the system prompt itself (hermes-agent bot_mode_probe).
 *  Gates every SOUL.md protocol append below. */
let serverInjectsProtocol = false

function useRoster() {
  const activeConnectionId = useValue(host.state.connectionId)

  return useQuery({
    queryKey: [...ROSTER_KEY, activeConnectionId],
    queryFn: async () => {
      // Stamp the ISSUE time on the snapshot: mergeServerMeta compares it
      // against each bot's last local meta write, and a fetch issued before
      // a write can only carry pre-write ui_meta. (Issue time is the
      // conservative bound — the server answered no earlier than this.)
      const issuedAt = Date.now()
      // Rich rows (last_session, canonical_session, ui_meta, has_avatar)
      // come from the ACTIVE gateway's profiles.list — the canonical Bot
      // Chat is resolved server-side by NAME (the "Bot Chat" registry row),
      // so the roster never sends session pointers.
      // Refresh the alias identity index alongside the roster: alias routes
      // (Desktop profile → remote backend root) are what let a backend row
      // keep its configured friendly identity after activation (#89131).
      // Best-effort and feature-detected — a failed read keeps the last
      // good index rather than dropping identities mid-session.
      if (typeof host.profileRoutes === 'function') {
        try {
          indexAliasRoutes(await host.profileRoutes())
        } catch {
          /* keep the previous alias index */
        }
      }
      // Owner routing is ambient in the SDK now (post-#92731): requestForBot
      // resolves the active owner itself, no captured route needed here.
      const activeBot = { name: String(host.state.profile?.get?.() || 'default').trim() || 'default' }
      const local = await requestForBot(activeBot, 'profiles.list', {})
      // Newer backends inject the teammate-messaging protocol into every
      // session's system prompt (agent.bot_mode_protocol) — SOUL.md must not
      // carry a second copy. Older gateways lack the flag: keep appending.
      serverInjectsProtocol = Boolean(local?.bot_mode_protocol)

      // Multi-source desktops (hermes-agent #86875) also expose the union
      // agent roster across every registered connection. Merge agents from
      // OTHER sources in as additional rows. Feature-detected + best-effort:
      // an older Desktop build (no host.agents) or a roster hiccup leaves
      // the local list exactly as it was.
      if (typeof host.agents === 'function') {
        try {
          const union = await host.agents()
          const previous = $lastRoster.get().filter(row => !row?.ghost)
          const merged = mergeMultiSourceRoster(local, union, activeConnectionId, previous)
          const sources = Array.isArray(union?.sources) ? union.sources : []

          return {
            ...merged,
            profiles: (merged?.profiles || []).map(row => annotateBotSource(row, sources)),
            sources,
            fetchedAt: issuedAt
          }
        } catch {
          /* older build or roster failure — single-source list stands */
        }
      }

      return { ...(local && typeof local === 'object' ? local : {}), fetchedAt: issuedAt }
    },
    refetchInterval: 5000,
    staleTime: 5000,
    // Remote (SSH) gateways connect slowly and drop on sleep/wake; keep
    // retrying instead of latching a terminal error card.
    retry: true,
    retryDelay: attempt => Math.min(15000, 1000 * 2 ** attempt)
  })
}

/** Synchronous union-roster read for the composer surfaces (autocomplete
 *  provider + mention middleware). useRoster caches under
 *  [...ROSTER_KEY, activeConnectionId] — a 3-element key — so a bare
 *  getQueryData(ROSTER_KEY) exact-match lookup returns undefined forever
 *  (issue #89303: remote handles absent from @ autocomplete, mentions
 *  unrouted). Read the live connection's entry first, then fall back to a
 *  prefix scan keeping the freshest snapshot. Never throws: cold cache or
 *  legacy queryClient returns null and callers fall back to their own path. */
function cachedUnionRoster() {
  if (typeof queryClient === 'undefined' || !queryClient || typeof queryClient.getQueryData !== 'function') {
    return null
  }

  try {
    const connectionId = String(
      host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local'
    )
    const exact = queryClient.getQueryData([...ROSTER_KEY, connectionId])

    if (Array.isArray(exact?.profiles)) {
      return exact
    }

    if (typeof queryClient.getQueriesData === 'function') {
      let best = null

      // v5 takes a filters object; a legacy v3 queryClient treats the same
      // object as the key itself and simply matches nothing — harmless.
      for (const [, data] of queryClient.getQueriesData({ queryKey: ROSTER_KEY })) {
        if (
          Array.isArray(data?.profiles) &&
          (!best || Number(data.fetchedAt || 0) > Number(best.fetchedAt || 0))
        ) {
          best = data
        }
      }

      return best
    }
  } catch {
    /* cache hiccup — caller falls back (middleware refetches) */
  }

  return null
}

/** Merge the union agent roster (host.agents) over the active gateway's
 *  profiles.list. Active-source rows — matched by the LIVE connection id,
 *  falling back to the roster's primaryConnectionId, then the legacy
 *  kind==='local' rule on older desktops — are the agents profiles.list
 *  already returned: they only ANNOTATE the rich rows (handle, connection
 *  fields); rich fields stay authoritative and they are NOT duplicated.
 *  Rows from other sources become new roster entries tagged with their
 *  source label so BotRow can badge them, warm the captured agent, and route
 *  every open directly through that descriptor. Pure — exercised directly by
 *  the tests. */
function mergeMultiSourceRoster(local, union, activeConnectionId, previous = []) {
  const localProfiles = Array.isArray(local?.profiles) ? local.profiles : []
  const agents = Array.isArray(union?.agents) ? union.agents : []
  // A live id of null/'' means the window is on the unscoped local backend
  // (legacy hosts reported null for mode:'local'; the SDK now reports
  // 'local'). Do NOT fall back to registry primary when the third argument
  // was passed — primary can still say "spark" after the user clicked a
  // local bot, which skipped every Spark row as "active" and invented a
  // This-device shadow of default.
  const liveProvided = arguments.length >= 3
  const liveId = String(activeConnectionId || '').trim()
  let activeId = liveId || (liveProvided ? '' : String(union?.primaryConnectionId || '').trim())

  // Migrated remote-primary windows can still expose a legacy remote
  // descriptor without connectionId. That produces a null live id even
  // though profiles.list is answering from the registry primary. Infer the
  // primary only when its inventory matches the rich rows and the local
  // inventory does not. A genuinely local window has a matching local row,
  // so it keeps the null-is-local behavior used after clicking This device.
  if (!activeId && liveProvided) {
    const primaryId = String(union?.primaryConnectionId || '').trim()
    const richNames = new Set(localProfiles.map(profile => String(profile?.name || '').trim()).filter(Boolean))
    const localMatches = agents.some(
      agent => agent?.connectionKind === 'local' && richNames.has(String(agent?.profile || '').trim())
    )
    const primaryMatches = agents.some(
      agent => String(agent?.connectionId || '').trim() === primaryId && richNames.has(String(agent?.profile || '').trim())
    )

    if (!localMatches && primaryId && primaryMatches) {
      activeId = primaryId
    }
  }
  const activeByName = new Map()

  // Treat the rich list as one row per active-source profile. Clone every
  // row: some gateway clients reuse response objects, and annotating those in
  // place made each five-second refresh feed the previous union back into the
  // next merge, growing duplicate source rows indefinitely.
  for (const profile of localProfiles) {
    const name = String(profile?.name || '').trim()

    if (!name || profile?.remoteSource) {
      continue
    }

    if (profile?.sourceScoped && activeId && profile.connectionId !== activeId) {
      continue
    }

    if (!activeByName.has(name)) {
      activeByName.set(name, { ...profile, name })
    }
  }

  const profiles = [...activeByName.values()]

  // host.agents is an Electron/main-process capability. Defend the plugin
  // boundary too: older shells or reconnect races can still hand us repeated
  // identities even after the core roster deduplicates them.
  const seenSources = new Set()

  for (const agent of agents) {
    const profile = String(agent?.profile || '').trim()
    const connectionId = String(agent?.connectionId || '').trim()
    const sourceKey = `${connectionId}::${profile || 'default'}`

    if (!profile || seenSources.has(sourceKey)) {
      continue
    }

    seenSources.add(sourceKey)

    // The union enumerates EVERY registered connection, including the active
    // gateway that already answered profiles.list. Without this the active
    // gateway's own agents (connectionKind 'remote' on a remote-primary
    // desktop) would be appended as phantom duplicates — every bot listed
    // twice. Older Electron builds predate the connection ids; fall back to
    // the legacy local-source rule so single-source behavior stays intact.
    const isActiveSource = activeId ? connectionId === activeId : agent.connectionKind === 'local'
    const row = isActiveSource ? activeByName.get(profile) : null

    if (row) {
      // Annotate in place: the @name-device handle only differs from the
      // bare name when the profile exists on several sources.
      row.handle = agent.handle
      row.connectionId = agent.connectionId
      row.connectionKind = agent.connectionKind
      row.connectionLabel = agent.connectionLabel
      row.targetProfile = agent.targetProfile || profile
      row.route = {
        connectionId,
        mode: agent.connectionKind === 'local' ? 'local' : 'remote',
        profile,
        targetProfile: agent.targetProfile || profile
      }
      row.sourceScoped = true
      continue
    }

    if (isActiveSource) {
      // Union saw an active-source profile profiles.list didn't return (older
      // backend mid-refresh) — skip rather than invent a thin row.
      continue
    }

    profiles.push({
      name: profile,
      handle: agent.handle,
      connectionId,
      connectionKind: agent.connectionKind,
      connectionLabel: agent.connectionLabel,
      targetProfile: agent.targetProfile || profile,
      route: {
        connectionId,
        mode: agent.connectionKind === 'local' ? 'local' : 'remote',
        profile,
        targetProfile: agent.targetProfile || profile
      },
      remoteSource: true,
      sourceScoped: true
    })
  }

  // SSH sources drop to connect-on-demand the moment their tunnel is not
  // the live gateway. Keep previously painted remote rows so clicking the
  // local agent does not empty Bot Mode.
  if (Array.isArray(previous) && previous.length > 0) {
    const present = new Set(profiles.map(row => `${row.connectionId || ''}::${row.name}`))
    const unionSourceIds = new Set(agents.map(agent => String(agent?.connectionId || '').trim()).filter(Boolean))
    const omitted = new Set(
      (Array.isArray(union?.sources) ? union.sources : [])
        .filter(source => source?.error === 'connect-on-demand' || source?.reachable === false)
        .map(source => String(source.connectionId || '').trim())
        .filter(Boolean)
    )

    const registered = new Set(
      (Array.isArray(union?.sources) ? union.sources : [])
        .map(source => String(source?.connectionId || '').trim())
        .filter(Boolean)
    )

    for (const row of previous) {
      const connectionId = String(row?.connectionId || '').trim()
      const name = String(row?.name || '').trim()
      const key = `${connectionId}::${name || 'default'}`

      if (!row?.remoteSource || !connectionId || !name || present.has(key)) {
        continue
      }

      if (registered.size > 0 && !registered.has(connectionId)) {
        continue
      }

      if (omitted.has(connectionId) || !unionSourceIds.has(connectionId)) {
        profiles.push({ ...row, remoteSource: true, sourceScoped: true })
        present.add(key)
      }
    }
  }

  return { ...local, profiles }
}

/** The @handle users tag a bot with. Multi-source rosters precompute the
 *  handle (bare name, or name-device when the profile exists on several
 *  registered sources) — prefer it when present. The primary profile's
 *  callable alias is 'hermes' — the mention middleware resolves it back to
 *  'default' — so the word 'default' never surfaces in the UI. */
function botHandle(name, bot) {
  if (bot?.handle && bot.handle !== name) {
    return bot.handle
  }

  return (name || '').trim().toLowerCase() === 'default' ? 'hermes' : name
}

/** Taggable @-forms derived from a bot's friendly names — the core profile
 *  display name (`hermes profile rename`) and the Bot Mode title. Free text
 *  reduces to the mention charset two ways: slugified ("Research Buddy" →
 *  research-buddy, the form autocomplete inserts) and collapsed
 *  (researchbuddy). Reserved tokens are dropped so a bot renamed "Hermes"
 *  can never hijack the primary profile's @hermes alias. */
function mentionNameForms(value) {
  const name = String(value || '').trim().toLowerCase()

  if (!name) {
    return []
  }

  const slug = name.replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')
  const collapsed = name.replace(/[^a-z0-9_-]+/g, '')

  return [...new Set([slug, collapsed])].filter(
    form => /^[a-z0-9][a-z0-9_-]*$/.test(form) && !['all', 'everyone', 'user', 'default', 'hermes'].includes(form)
  )
}

/** Every friendly (renameable) name a roster row carries: the Bot Mode title
 *  (server-synced via ui_meta, locally stored, or persisted on a durable
 *  group descriptor) and the core profile display_name — in displayName's
 *  precedence order. Remote rows never borrow local meta (two `default`s
 *  must not share a title) — EXCEPT the connection-exact alias identity
 *  (#89131): a backend row claimed by a configured alias route carries the
 *  alias's friendly names, so @moxie keeps resolving after handoff. */
function botFriendlyNames(bot) {
  const metaByName = typeof $botMeta !== 'undefined' ? $botMeta.get() : null
  const localTitle = !bot?.remoteSource ? metaByName?.[bot?.name]?.title : null
  const alias = aliasIdentityFor(bot)
  const aliasTitle = alias
    ? alias.metaKeys.map(key => metaByName?.[key]?.title).find(title => typeof title === 'string' && title.trim()) ||
      alias.name
    : null

  return [bot?.ui_meta?.['hermes-bots']?.title, localTitle, aliasTitle, bot?.title, bot?.display_name]
}

/** The tag autocomplete inserts for a bot: the renamed (friendly) slug when
 *  the user gave the bot a real name, otherwise the profile @handle. The
 *  resolvers accept both, so older muscle memory keeps working. */
function botMentionTag(bot) {
  for (const friendly of botFriendlyNames(bot)) {
    const forms = mentionNameForms(friendly)

    if (forms.length) {
      return forms[0]
    }
  }

  return botHandle(bot?.name, bot)
}

function isActiveRosterBot(bot, active) {
  if (!active) {
    return false
  }

  const activeName = String(active.name || 'default').trim() || 'default'
  const activeId = String(active?.connectionId || '').trim()
  const botId = String(bot?.connectionId || '').trim()
  const botName = String(bot?.name || '').trim() || 'default'

  if (bot?.remoteSource) {
    return Boolean(activeId) && activeId === botId && botName === activeName
  }

  if (activeId && activeId !== 'local' && botId && activeId !== botId) {
    return false
  }

  return botName === activeName
}

function botSelectionKey(bot) {
  return bot?.sourceScoped || bot?.remoteSource ? botRosterKey(bot) : bot?.name
}

function isDefaultBot(bot) {
  const route = botConnectionRoute(bot)

  return String(route?.profile || bot?.name || '').trim().toLowerCase() === 'default'
}

function newBotChat(bot) {
  if (typeof host.newChat !== 'function') {
    host.notify?.({ kind: 'error', message: 'Update Hermes Desktop to open another Bot chat.' })

    return
  }

  const route = botConnectionRoute(bot)

  if (!route) {
    host.notify?.({ kind: 'error', message: 'Update Hermes Desktop to open another Bot chat.' })

    return
  }

  const ownerKey = botWorkspaceOwnerKey(bot)
  setBotsWorkspaceOwner(ownerKey, bot)
  host.newChat(route, { workspaceMode: 'bots', workspaceOwnerKey: ownerKey })
}

/** Resolve @handles in prose against the Bot Mode roster (local + Connections).
 *  Skips the bot already speaking in this chat. Unique bare names match;
 *  duplicate names require the @name-device handle. */
function resolveRosterMentions(text, roster, active = {}) {
  const members = Array.isArray(roster) ? roster : []
  const prose = String(text || '').replace(/```[\s\S]*?```/g, ' ').replace(/`[^`\n]*`/g, ' ')
  const byForm = new Map()

  for (const bot of members) {
    if (!bot?.name || isActiveRosterBot(bot, active)) {
      continue
    }

    const handle = String(botHandle(bot.name, bot) || '').toLowerCase()
    const name = String(bot.name || '').toLowerCase()
    const forms = new Set([handle, name])

    if (bot.handle) {
      forms.add(String(bot.handle).toLowerCase())
    }

    // Renamed bots are taggable by their friendly names too — the core
    // profile display_name and the Bot Mode title (issue: renaming a bot
    // didn't change what you @-tag it with).
    for (const friendly of botFriendlyNames(bot)) {
      for (const form of mentionNameForms(friendly)) {
        forms.add(form)
      }
    }

    for (const form of forms) {
      if (!form) {
        continue
      }

      const existing = byForm.get(form)

      if (existing && existing !== bot) {
        byForm.set(form, null)
        continue
      }

      if (!existing) {
        byForm.set(form, bot)
      }
    }
  }

  const mentioned = []
  const seen = new Set()

  for (const match of prose.matchAll(/(^|\s)@([a-z0-9][a-z0-9_-]*)/gi)) {
    let token = match[2].toLowerCase()

    if (token === 'hermes') {
      token = byForm.has('hermes') ? 'hermes' : token
    }

    const bot = byForm.get(token)

    if (!bot) {
      continue
    }

    const key = botRosterKey(bot)

    if (seen.has(key)) {
      continue
    }

    seen.add(key)
    mentioned.push(bot)
  }

  return mentioned
}

/** Source-qualified identity for a roster row — the React list key AND the
 *  cross-surface roster identity. Names alone are NOT unique in a
 *  multi-source roster (two connections can both expose 'default');
 *  duplicate keys make React reconciliation repeat whole blocks of the list
 *  on every poll repaint (the Aug 2026 dupe-bots smear). */
function botRosterKey(bot) {
  return `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`
}

function botRouteKey(route) {
  return `${route.connectionId}::${route.profile}`
}

function botMetaKey(bot) {
  const route = botConnectionRoute(bot)

  return route ? botRouteKey(route) : bot?.name
}

function persistBotMetaSnapshot(value, scoped = false) {
  try {
    const persisted = scoped
      ? commitBotMetaV2(pluginCtx?.storage, value)
      : Promise.resolve(pluginCtx?.storage?.set?.(BOT_META_V1_KEY, value))

    return persisted.catch(() => undefined)
  } catch {
    return Promise.resolve()
  }
}

function sourceByConnection(sources) {
  return new Map(
    (Array.isArray(sources) ? sources : [])
      .filter(source => source?.connectionId)
      .map(source => [String(source.connectionId), source])
  )
}

/** Copy current source health onto a row without changing its owner. */
function annotateBotSource(bot, sources) {
  const id = String(bot?.connectionId || '').trim()

  if (!id) {
    return bot
  }

  const list = Array.isArray(sources) ? sources : []
  const source = sourceByConnection(list).get(id)

  if (!source) {
    return list.length && bot?.sourceScoped ? { ...bot, sourceMissing: true, sourceReachable: false } : bot
  }

  return {
    ...bot,
    connectionKind: bot.connectionKind || source.kind,
    connectionLabel: bot.connectionLabel || source.label,
    sourceError: source.error || null,
    sourceMissing: false,
    sourceReachable: source.reachable
  }
}

function botSourceStatus(bot) {
  const error = String(bot?.sourceError || '').trim()

  if (bot?.sourceMissing) {
    return { available: false, key: 'missing', label: 'Gateway removed', tone: 'bad' }
  }

  if (error === 'connect-on-demand') {
    return { available: true, key: 'on-demand', label: 'On demand', tone: 'muted' }
  }

  if (error || bot?.sourceReachable === false) {
    return { available: false, key: 'unavailable', label: 'Unavailable', tone: 'warn' }
  }

  if (bot?.sourceReachable === true) {
    return { available: true, key: 'ready', label: 'Ready', tone: 'good' }
  }

  return { available: true, key: 'unknown', label: 'Status unknown', tone: 'muted' }
}
// ── cross-connection routing ─────────────────────────────────────────────────
// A bot from another registered connection (remoteSource rows) is reached
// through host.requestProfile with a route descriptor; local bots keep the
// active-gateway door. Feature-detected: older desktops without
// requestProfile simply have no remote routes (callers fall back / disable).

/** Non-throwing resolver behind botConnectionRoute(). Returns a typed status
 *  instead of throwing, so passive callers (display/meta lookups) can branch
 *  on `resolved | owner_removed | not_scoped` rather than catching whatever
 *  exception the strict wrapper below happens to throw. */
function resolveBotConnectionRoute(bot) {
  if (!bot?.sourceScoped && !bot?.remoteSource) {
    return { status: 'not_scoped', route: null }
  }

  const candidate = bot.route || {
    connectionId: bot.connectionId,
    mode: bot.connectionKind === 'local' ? 'local' : 'remote',
    profile: bot.name,
    targetProfile: bot.targetProfile || bot.name
  }
  const connectionId = String(candidate?.connectionId || '').trim()
  const profile = String(candidate?.profile || bot?.name || '').trim() || 'default'
  const targetProfile = String(candidate?.targetProfile || profile).trim() || profile

  if (!connectionId) {
    return { status: 'owner_removed', route: null, profile }
  }

  return {
    status: 'resolved',
    route: Object.freeze({
      connectionId,
      mode: candidate.mode === 'local' || connectionId === 'local' ? 'local' : 'remote',
      profile,
      targetProfile
    })
  }
}

/** Immutable owner descriptor for every source-scoped row. The active
 *  gateway is presentation state and is never consulted here. Strict: throws
 *  when the owning connection is gone -- correct for real dispatch
 *  (requestForBot, session creation, etc.), covered by
 *  remote-routing-races.test.mjs. Passive lookups (rendering, meta) must call
 *  resolveBotConnectionRoute() directly instead of catching this throw. */
function botConnectionRoute(bot) {
  const resolved = resolveBotConnectionRoute(bot)

  if (resolved.status === 'owner_removed') {
    throw new Error(`Bot ${resolved.profile} has no connection owner`)
  }

  return resolved.route
}

const BOTS_HOME_OWNER_KEY = 'bots:home'

function botWorkspaceOwnerKey(bot) {
  // Render-reachable (sidebar sync, Bots home open, context menus): an
  // orphaned row must yield a stable degraded key, never the dispatch throw.
  const route = resolveBotConnectionRoute(bot).route

  return `bot:${route ? botRouteKey(route) : String(bot?.name || 'default')}`
}

function groupWorkspaceOwnerKey(group) {
  return `group:${groupChatRoomKey(group, $groupChats.get()[group])}`
}

function setBotsWorkspaceOwner(ownerKey, bot = null, blockedMessage = 'Select a Bot or group first.') {
  // Render-reachable (sidebar listener fires on visibility flips). An
  // orphaned row degrades to the blocked target instead of throwing.
  const route = bot ? resolveBotConnectionRoute(bot).route : null
  const target = route ? { kind: 'route', route } : { kind: 'blocked', message: blockedMessage }

  host.setWorkspaceScope?.('bots', ownerKey || BOTS_HOME_OWNER_KEY, target)
}
function backendTargetProfile(route, fallbackProfile = 'default') {
  if (!route) {
    return fallbackProfile
  }

  return route.targetProfile || route.profile
}

function rewriteCliProfileOperands(argv, logical, target) {
  const next = [...argv]

  for (let index = 0; index < next.length; index += 1) {
    if (next[index] === '--profile' && next[index + 1] === logical) {
      next[index + 1] = target
      index += 1
    } else if (next[index] === `--profile=${logical}`) {
      next[index] = `--profile=${target}`
    }
  }

  const profileCommand = next.indexOf('profile')
  const operand = profileCommand >= 0 ? profileCommand + 2 : -1
  if (operand < next.length && next[operand] === logical) {
    next[operand] = target
  }

  return next
}

function scopedBotParams(route, method, params) {
  const logical = route.profile
  const target = backendTargetProfile(route, logical)
  let next = params

  if (Object.prototype.hasOwnProperty.call(params, 'profile')) {
    next = { ...next, profile: target }
  }

  if (method.startsWith('profiles.') && method !== 'profiles.create' && params.name === logical) {
    next = { ...next, name: target }
  }

  if (params.clone_from === logical) {
    next = { ...next, clone_from: target }
  }

  if (method === 'cli.exec' && Array.isArray(params.argv)) {
    next = { ...next, argv: rewriteCliProfileOperands(params.argv, logical, target) }
  }

  return next
}

function botBackendProfileScope(route, fallbackProfile = 'default') {
  if (!route) {
    return fallbackProfile
  }

  return { connectionId: route.connectionId, profile: backendTargetProfile(route, fallbackProfile) }
}

/** Gateway RPC on the bot's OWN source. Source-scoped rows always use the
 * explicit descriptor, including a registered local source. */
async function requestForBot(bot, method, params = {}) {
  const route = botConnectionRoute(bot)

  if (route) {
    if (typeof host.requestProfile !== 'function') {
      throw new Error(`Cannot route ${method} for ${route.connectionId}::${route.profile}`)
    }

    return host.requestProfile(route, method, scopedBotParams(route, method, params))
  }

  return host.request(method, params)
}

/** Stable per-member identity inside a group room. Local members keep their
 *  bare name (compat with rooms persisted before cross-connection groups);
 *  remote members get the source-qualified key so `dixie` on the Mini and a
 *  local `dixie` never share watermarks or sessions. */
function groupMemberKey(member) {
  return member?.sourceScoped || member?.remoteSource ? botRosterKey(member) : member?.name
}

/** Serializable immutable owner captured beside every group plumbing session. */
function groupSessionOwner(member) {
  const route = botConnectionRoute(member)
  const name = String(route?.profile || member?.name || '').trim() || 'default'

  if (!route) {
    return { name }
  }

  return {
    connectionId: route.connectionId,
    name,
    sourceScoped: true,
    remoteSource: route.mode !== 'local',
    route: { ...route }
  }
}

// ── alias identity for connection rows (#89131) ─────────────────────────────
// A Desktop per-profile alias (profile `moxie` with a Cloud/URL/SSH override)
// routes to a remote backend's root profile: its route reads
// { connectionId: C, profile: 'moxie', targetProfile: 'default' }. Once that
// backend answers the roster itself, the row's identity is (C, 'default') —
// a DIFFERENT key than the alias meta (C::moxie / 'moxie') — so the friendly
// name fell off after source/session activation: the row regressed to the
// raw Cloud hostname, or to generic 'Hermes' in Cloud-only mode.
//
// aliasRouteIndex bridges the backend row identity back to its configured
// alias. It is keyed by (connectionId, targetProfile), so two same-named
// `default` rows on different connections can never share a title, and it
// fails closed when two aliases claim the same backend row (mirroring the
// fail-closed route resolution). This is the one sanctioned exception to
// "remote rows never borrow local meta": the alias IS the local identity of
// exactly this connection row, proven by the configured route — never by a
// bare name match.
let aliasRouteIndex = new Map()

/** Rebuild the alias index from the credential-free route inventory. Only
 *  genuine aliases (route.profile !== route.targetProfile) participate. */
function indexAliasRoutes(routes) {
  const next = new Map()

  for (const route of Array.isArray(routes) ? routes : []) {
    const connectionId = String(route?.connectionId || '').trim()
    const profile = String(route?.profile || '').trim()
    const target = String(route?.targetProfile || '').trim()

    if (!connectionId || !profile || !target || profile === target) {
      continue
    }

    const key = `${connectionId}::${target}`

    // Two aliases pointing at the same backend row are ambiguous — neither
    // may claim the identity.
    next.set(key, next.has(key) ? null : {
      name: profile,
      // Alias meta can live under the source-qualified v2 key or the bare
      // v1 name key (aliases predate the v2 migration on mixed setups).
      metaKeys: [`${connectionId}::${profile}`, profile]
    })
  }

  aliasRouteIndex = next
}

/** The configured alias identity claiming this roster row, or null. Matches
 *  strictly by (connectionId, backend target profile); the alias row itself
 *  keeps resolving its own meta directly. */
function aliasIdentityFor(bot) {
  if (!aliasRouteIndex.size) {
    return null
  }

  const connectionId = String(
    bot?.connectionId ||
      bot?.route?.connectionId ||
      // Unannotated rich rows (no host.agents on this build) still belong to
      // the ACTIVE gateway — Cloud-only mode must resolve the alias too.
      (!bot?.remoteSource && !bot?.sourceScoped ? host.state.connectionId?.get?.() || '' : '')
  ).trim()

  if (!connectionId) {
    return null
  }

  const target = String(bot?.targetProfile || bot?.route?.targetProfile || bot?.name || '').trim() || 'default'
  const entry = aliasRouteIndex.get(`${connectionId}::${target}`) || null

  return entry && entry.name !== String(bot?.name || '').trim() ? entry : null
}

// Bot metadata is scoped to the active gateway until the server exposes a
// union of rich profile rows. Never paint that metadata onto a thin row from
// another source: two `default` agents must not borrow each other's title,
// pin, avatar, group, unread state, or canonical-chat pointer. The ONE
// exception is a configured alias route claiming the row — see
// aliasRouteIndex above — which is connection-exact, never name-based.
function botRosterMeta(bot, metaByName) {
  if (bot?.sourceScoped || bot?.remoteSource) {
    // Passive meta lookup: branch on the typed status instead of catching
    // botConnectionRoute's throw, so an owner_removed row (e.g. a stale
    // persisted group roster after its connection was deleted) reads as "no
    // route" without masking an unrelated failure under the same catch.
    const resolved = resolveBotConnectionRoute(bot)
    const route = resolved.status === 'resolved' ? resolved.route : null

    const direct = route ? metaByName?.[botRouteKey(route)] : null

    if (direct) {
      return direct
    }

    const alias = aliasIdentityFor(bot)

    if (alias) {
      for (const key of alias.metaKeys) {
        if (metaByName?.[key]) {
          return metaByName[key]
        }
      }
    }

    return direct
  }

  const own = metaByName?.[bot?.name]

  if (own) {
    return own
  }

  const alias = aliasIdentityFor(bot)

  if (alias) {
    for (const key of alias.metaKeys) {
      if (metaByName?.[key]) {
        return metaByName[key]
      }
    }
  }

  return own
}

function showsHandle(name, meta, bot) {
  const display = displayName({ name }, meta)
  return Boolean(name && display.toLowerCase() !== botHandle(name, bot).toLowerCase())
}

// ── canonical bot chat ───────────────────────────────────────────────────────
// Each bot has ONE forever chat, identified by NAME, never by pointer: the
// session titled exactly "Bot Chat" on that bot's profile. The core
// UNIQUE(title) index makes (profile, "Bot Chat") an exact registry, so every
// open consults that registry directly — there is nothing to verify, re-pin,
// grandfather, or recover. Stored-id pins (ui_meta['hermes-bots'].chat) were
// the previous identity and are REMOVED: every lost-chat incident traced to a
// dangled or stolen pointer that later guards then welded in. Legacy
// ui_meta.chat keys are simply ignored.

// In-flight creations, keyed by bot name — double-clicking a row must not
// mint two canonical chats.
const canonicalCreations = new Map()

/** Upper bound for per-profile session.list scans (hide sweep, canonical-chat
 *  adoption, stored-session lookups). */
const PROFILE_SESSION_LIST_LIMIT = 200
let botOpenGeneration = 0

/** The one canonical title. (profile, CANONICAL_CHAT_TITLE) IS the bot's
 *  forever-chat identity — see the header above. */
const CANONICAL_CHAT_TITLE = 'Bot Chat'

async function openStoredBotChat(owner, storedId, summary) {
  if (!storedId || typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }

  const { bot, name, route } = botOwner(owner)
  const ownerKey = botWorkspaceOwnerKey(bot)

  const hasAuthoritativeCount =
    typeof summary?.message_count === 'number' && Number.isFinite(summary.message_count)
  const expectHistory = hasAuthoritativeCount ? summary.message_count > 0 : true

  // A profile backend that just woke up can lose the hydration-timeout race
  // even though the session is fine (hermes-agent#89617) — clicking Retry
  // succeeds because the backend is warm by then. retryHydrationTimeoutOnce
  // asks the SDK layer to retry that same wait internally, BEFORE it arms the
  // core stranded-session overlay: a plugin-side retry can't do this because
  // only host.openSession sees the resume-exhausted latch that overlay reads.
  await host.openSession(storedId, {
    ...(route ? { route } : {}),
    profile: name,
    intent: 'tab',
    awaitHydration: true,
    expectHistory,
    keepAllProfilesScope: true,
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey,
    retryHydrationTimeoutOnce: true,
    tabTitle: CANONICAL_CHAT_TITLE
  })

  return storedId
}

/** True when a session summary IS the canonical registry row. root_title is
 *  the durable lineage-root title reported by exact-lookup gateways; plain
 *  title covers windowed listings. */
function isCanonicalBotChatHistory(history) {
  const rootTitle = String(history?.root_title || '').trim()
  const title = String(history?.title || '').trim()
  return rootTitle === CANONICAL_CHAT_TITLE || (!rootTitle && title === CANONICAL_CHAT_TITLE)
}

function botModeGatewayNeedsUpdate(error) {
  const message = String(error?.message || error || '')

  return /(?:method not found|no handler for|unknown method|unsupported rpc)/i.test(message)
}

function notifyBotOpenFailure(error, bot, fallbackMessage) {
  if (botModeGatewayNeedsUpdate(error)) {
    const gateway = bot.connectionLabel || bot.connectionId || 'this gateway'

    host.notify?.({
      kind: 'error',
      title: 'Update this gateway to use Bot Mode',
      message: `Update ${gateway}, then try again.`
    })

    return
  }

  host.notifyError?.(error, fallbackMessage)
}

/** THE identity lookup: the profile's session titled exactly "Bot Chat",
 *  consulted on the bot's OWN source. The core UNIQUE title index guarantees
 *  at most ONE such row per profile db — Profile → Named Session is an exact
 *  registry, so consult it exactly: `title` asks the gateway for an indexed
 *  WHERE title = ? lookup (window-free; a busy profile can push the
 *  forever-chat past any recency window). include_hidden is required
 *  (canonical chats are always hidden). Remote bots route via requestForBot
 *  on the immutable captured owner — activation is a UI concern and never
 *  authorizes this RPC. */
async function findExistingCanonicalChat(owner) {
  const { bot, name, route } = botOwner(owner)
  // FAIL CLOSED. A failed registry lookup MUST NOT read as "no Bot Chat
  // exists" — that is the one remaining way to fork a bot's forever chat.
  // The failure lives exactly in the post-update window: the desktop
  // restarts every profile backend, the first bot click races the warm-up,
  // the lookup RPC fails transiently, and a swallowed error here sent
  // createCanonicalChat() straight to session.create — minting a fresh
  // "Bot Chat" while the real one (data intact, hidden) still held the
  // canonical title. Users read that as "my bot lost all context after the
  // update". Cross-connection lookups fail MORE often (network), so this
  // matters doubly for remote bots. Both open paths catch and toast "try
  // again", which is the correct outcome: retry, never mint.
  let res
  try {
    res = await requestForBot(bot, 'session.list', {
      profile: backendTargetProfile(route, name),
      title: CANONICAL_CHAT_TITLE,
      limit: PROFILE_SESSION_LIST_LIMIT,
      include_hidden: true
    })
  } catch (error) {
    // Plugin tests and host bridges can return Error-like values from another
    // JS realm, where `instanceof Error` is false. Preserve the provider/RPC
    // message so update-required classification and diagnostics still work.
    const message = typeof error?.message === 'string' ? error.message : ''
    const detail = message ? ` (${message})` : ''
    throw new Error(`Could not check ${name}'s Bot Chat registry${detail} — not starting a new chat`)
  }

  const rows = res?.sessions ?? []
  return rows.find(row => isCanonicalBotChatHistory(row)) || null
}

/** Create the bot's ONE forever chat: a real session titled "Bot Chat",
 *  opened with a kickoff message (the gateway prunes zero-message sessions,
 *  so the chat is born with the bot introducing itself). Adopts the existing
 *  "Bot Chat" row instead of creating when the profile already has one —
 *  minting while a "Bot Chat" row exists is always wrong twice over: it
 *  forks the forever-chat AND the new row can never take the (already held)
 *  canonical title. Creates on the bot's own source via requestForBot. */
function createCanonicalChat(owner) {
  const { bot, name, key, route } = botOwner(owner)
  const inflight = canonicalCreations.get(key)

  if (inflight) {
    return inflight
  }

  const run = (async () => {
    const existing = await findExistingCanonicalChat(owner)

    if (existing?.id) {
      if (typeof host.openSession === 'function') {
        // The exact-lookup gateway reports the compression-lineage tip as
        // resolved_id; open the tip, the registry row stays the identity.
        await openStoredBotChat(owner, existing.resolved_id || existing.id, existing)
      }

      return existing.id
    }

    const res = await requestForBot(bot, 'session.create', {
      profile: backendTargetProfile(route, name),
      title: CANONICAL_CHAT_TITLE,
      // Always born hidden from the global sidebar — Bot Mode sessions are
      // plugin-owned. Core applies this via the generic `hidden` flag
      // (deferred as pending_hidden until the row exists); older gateways
      // ignore the unknown param and it stays visible.
      hidden: true
    })
    const sid = res?.stored_session_id
    const runtime = res?.session_id

    // session.create is intentionally lazy: its stored row does not exist until
    // the first prompt. Mounting `sid` immediately therefore emits a noisy REST
    // 404 ("Session not found"), and the turn-start auto-titler can win the race
    // against the deferred `title: 'Bot Chat'` — under name-identity that is an
    // identity outage: until the row is titled, the registry has no "Bot Chat"
    // entry, so a second click during the intro turn mints a duplicate.
    // session.title materializes the row now and records a user-authority title
    // before either the open or kickoff, closing both the 404 race and the
    // untitled window. Older gateways may not support the eager write; retain
    // the kickoff-and-retry fallback below.
    if (runtime) {
      try {
        await requestForBot(bot, 'session.title', { session_id: runtime, title: CANONICAL_CHAT_TITLE })
      } catch {
        /* compatibility fallback: prompt.submit will persist the lazy row */
      }
    }

    // Mount the session view FIRST, then send the kickoff — submitting into
    // an unmounted session left the intro reply invisible until reopen.
    let opened = false

    if (sid && typeof host.openSession === 'function') {
      try {
        await host.openSession(sid, {
          ...(route ? { route } : {}),
          profile: name,
          intent: 'main',
          keepAllProfilesScope: route ? true : false
        })
        opened = true
      } catch {
        // The stored row may not exist until the kickoff persists it. Retry
        // after prompt.submit below instead of leaving the chat off-screen.
      }
    }

    if (runtime) {
      await new Promise(resolve => window.setTimeout(resolve, 400))

      try {
        await requestForBot(bot, 'prompt.submit', { session_id: runtime, text: 'Hey, tell me about yourself!' })

        if (!opened && sid && typeof host.openSession === 'function') {
          await host.openSession(sid, {
            ...(route ? { route } : {}),
            profile: name,
            intent: 'main',
            keepAllProfilesScope: route ? true : false
          })
        }
      } catch {
        // The chat already exists under the canonical title — the next click
        // finds it by name instead of making a second Bot Chat.
      }
    }

    return sid || null
  })().finally(() => canonicalCreations.delete(key))

  canonicalCreations.set(key, run)

  return run
}

/** Open the bot's ONE forever chat and return the opened registry id.
 *
 *  The whole resolution is one registry consultation ON THE BOT'S OWN
 *  SOURCE: the profile's session titled "Bot Chat" exists → open it
 *  (lineage tip); it doesn't → create it. No id pointer is read or written
 *  anywhere in this path — remote bots included. The owner route rides
 *  every RPC (requestForBot) and the open (openStoredBotChat), so a remote
 *  bot's chat opens without re-homing Desktop's chrome. */
async function openBotCanonicalChat(owner) {
  const existing = await findExistingCanonicalChat(owner)

  if (existing?.id && typeof host.openSession === 'function') {
    const openedId = existing.resolved_id || existing.id
    await openStoredBotChat(owner, openedId, existing)
    // Both identities matter downstream: the durable registry row names the
    // chat; the resolved lineage tip is what actually takes session focus.
    // Callers matching focus against only the registry id mistook every
    // compressed Bot Chat for a stale open (first click bounced to the home).
    return { registryId: String(existing.id), openedId: String(openedId) }
  }

  const created = await createCanonicalChat(owner)
  return created ? { registryId: String(created), openedId: String(created) } : null
}

async function prepareBotSource(bot) {
  if (!bot.sourceScoped) {
    return
  }

  // Cross-connection RPCs ride the immutable captured route (requestForBot →
  // host.requestProfile) — Desktop's active connection does not move, and
  // activation is a UI concern that never authorizes the calls. All this
  // gate does is refuse when the desktop predates routed profile requests.
  const route = botConnectionRoute(bot)

  if (route && typeof host.requestProfile !== 'function') {
    throw new Error('Update Hermes Desktop to chat with agents on other connections.')
  }

  if (!route && typeof host.ensureAgent === 'function') {
    // Source-annotated row on the ACTIVE connection (no captured route):
    // legacy activation path, unchanged.
    await host.ensureAgent(bot.connectionId, bot.name)
  }
}

async function ensureBotMetadata(bot) {
  if (!bot?.sourceScoped) {
    return botRosterMeta(bot, $botMeta.get()) || {}
  }

  const route = botConnectionRoute(bot)
  const backendProfile = backendTargetProfile(route, bot.name)
  const result = await requestForBot(bot, 'profiles.list', {})
  const row = (result?.profiles || []).find(profile => profile?.name === backendProfile)
  const server = row?.ui_meta?.['hermes-bots']

  if (server && typeof server === 'object') {
    const key = botMetaKey(bot)
    $botMeta.set({ ...$botMeta.get(), [key]: { ...($botMeta.get()[key] || {}), ...server } })
    persistBotMetaSnapshot($botMeta.get(), true)
  }

  return botRosterMeta(bot, $botMeta.get()) || {}
}

/** Select one exact roster owner, then open its named canonical chat only when
 *  the current Desktop can route that owner without guessing. The workspace
 *  remembers only this transient opened-view observation; it never stores or
 *  resolves a canonical-chat id. */
async function openRosterBot(bot) {
  const generation = ++botOpenGeneration
  const key = botRosterKey(bot)
  const meta = botRosterMeta(bot, $botMeta.get())
  // Keep the currently visible group as a fallback until this explicit action
  // has actually fronted a new owner; a failed home open must not steal the
  // center from a group the user was reading.
  const previousGroup = $groupChatWorkspace.get()

  haptic('tap')
  saveSelectedRosterBot(bot)
  setBotsWorkspaceOwner(botWorkspaceOwnerKey(bot), bot)

  $groupChatWorkspace.set(null)

  if ($botUnread.get()[key]) {
    const next = { ...$botUnread.get() }
    delete next[key]
    $botUnread.set(next)
  }

  try {
    // Activation selects this row's source only. Canonical identity is resolved
    // after that by the owner profile's "Bot Chat" title registry.
    await prepareBotSource(bot)
  } catch (error) {
    if (generation === botOpenGeneration) {
      $openBotChat.set(null)
      if (previousGroup && !$groupChatWorkspace.get()) {
        $groupChatWorkspace.set(previousGroup)
      }
      syncBotsHomeWorkspace()

      notifyBotOpenFailure(error, bot, `Could not reach ${bot.connectionLabel || 'the gateway'}`)
    }

    return false
  }

  if (generation !== botOpenGeneration) {
    return false
  }

  try {
    const opened = await openBotCanonicalChat(bot)

    if (generation !== botOpenGeneration) {
      return false
    }

    if (opened) {
      // This is not an identity preference: opening already completed through
      // the name registry. Keep only enough ephemeral state to release the
      // home if another tab later claims the center. Track BOTH identities —
      // session focus reports the compression-lineage tip (openedId), not the
      // durable registry row, and matching focus against the registry id
      // alone released this claim on the first click of every compressed
      // Bot Chat (home bounced over the chat; a second click stuck only
      // because no new focus edge fired).
      $openBotChat.set({
        key,
        openedRegistryId: opened.registryId,
        openedSessionId: opened.openedId
      })
      closeBotsHomeWorkspace()
      return true
    }
  } catch (error) {
    if (generation === botOpenGeneration) {
      $openBotChat.set(null)
      if (previousGroup && !$groupChatWorkspace.get()) {
        $groupChatWorkspace.set(previousGroup)
      }
      syncBotsHomeWorkspace()

      notifyBotOpenFailure(error, bot, `Could not open ${displayName(bot, meta)}'s chat — try again`)
    }

    return false
  }

  // An older Desktop without the profile-scoped draft API has no safe fallback:
  // do not navigate the current workspace or create a draft on the wrong owner.
  if (typeof host.newChat !== 'function') {
    $openBotChat.set(null)
    if (previousGroup && !$groupChatWorkspace.get()) {
      $groupChatWorkspace.set(previousGroup)
    }
    syncBotsHomeWorkspace()
    return false
  }

  $openBotChat.set({ key, openedRegistryId: '' })
  closeBotsHomeWorkspace()
  newBotChat(bot)
  return true
}

function displayName(bot, meta) {
  // A configured alias route claiming this row overrides source-derived
  // identity: the friendly alias name must survive hosted-session
  // activation and Cloud-only rosters (#89131).
  const alias = aliasIdentityFor(bot)

  // Only THIN rows from another source trade the friendly name for their
  // connection label — the active gateway's own default must keep reading
  // "Hermes". Annotated active rows carry sourceScoped too, and keying this
  // off sourceScoped renamed the user's main agent to an IP-derived label
  // (community report, Aug 17 2026).
  if (bot?.remoteSource && (bot.name || '').trim().toLowerCase() === 'default' && bot.connectionLabel && !alias && !meta?.title?.trim()) {
    return bot.connectionLabel
  }

  if (meta?.title?.trim()) {
    return meta.title.trim()
  }

  // Core-profile display name (profile.yaml, set via `hermes profile rename
  // default <name>` or the dashboard) — the CLI-level equivalent of a Bot
  // Mode title. Rides the profiles.list row; presentation-only.
  if (typeof bot?.display_name === 'string' && bot.display_name.trim()) {
    return bot.display_name.trim()
  }

  // An untitled backend row claimed by an alias reads as the alias name —
  // never generic "Hermes" or a hostname-derived label.
  if (alias) {
    const raw = alias.name.replace(/[-_]+/g, ' ').trim()

    return raw.replace(/\b\w/g, ch => ch.toUpperCase())
  }

  // The primary profile is literally named "default" — as a bot identity
  // that reads like nobody bothered. Present it as Hermes (the agent it is)
  // unless the user gives it a real title.
  if ((bot.name || '').trim().toLowerCase() === 'default' && !bot.title) {
    return 'Hermes'
  }

  const raw = (bot.title || bot.name || '').replace(/[-_]+/g, ' ').trim()
  return raw.replace(/\b\w/g, ch => ch.toUpperCase())
}

/** Filter by the two stable identities rendered in every roster row: the
 * customizable display name and the profile's @handle. Keep the current
 * activity order — search narrows the roster, it never re-ranks it. */
function filterBots(roster, metaByName, query) {
  const needle = query.trim().toLowerCase().replace(/^@/, '')

  if (!needle) {
    return roster
  }

  return roster.filter(bot => {
    const meta = botRosterMeta(bot, metaByName)
    const display = displayName(bot, meta).toLowerCase()
    const profile = (bot.name || '').toLowerCase()
    const handle = botHandle(bot.name, bot).toLowerCase()
    // Multi-source rows also match on their device name ("homelab" finds
    // every bot living on the Homelab connection).
    const sourceLabel = (bot.connectionLabel || '').toLowerCase()
    const role = `${meta?.description || ''} ${bot.description || ''}`.toLowerCase()
    const preview = String(botActivitySession(bot)?.preview || '').toLowerCase()
    return (
      display.includes(needle) ||
      profile.includes(needle) ||
      handle.includes(needle) ||
      sourceLabel.includes(needle) ||
      role.includes(needle) ||
      preview.includes(needle)
    )
  })
}

function filterBotsByGateway(roster, connectionId) {
  if (!connectionId || connectionId === 'all') {
    return roster
  }

  return (roster || []).filter(bot => String(bot?.connectionId || '') === connectionId)
}

function botNeedsHandleLabel(bot, roster, metaByName) {
  const identity = displayName(bot, botRosterMeta(bot, metaByName)).trim().toLowerCase()
  const connectionId = String(bot?.connectionId || '')

  return (roster || []).some(
    candidate =>
      botRosterKey(candidate) !== botRosterKey(bot) &&
      String(candidate?.connectionId || '') === connectionId &&
      displayName(candidate, botRosterMeta(candidate, metaByName)).trim().toLowerCase() === identity &&
      botHandle(candidate.name, candidate) !== botHandle(bot.name, bot)
  )
}

function groupMatchesRosterFilters(name, members, metaByName, query, connectionId) {
  const inGateway = filterBotsByGateway(members, connectionId)

  if (connectionId && connectionId !== 'all' && inGateway.length === 0) {
    return false
  }

  const needle = String(query || '').trim().toLowerCase().replace(/^@/, '')

  return !needle || String(name || '').toLowerCase().includes(needle) || filterBots(inGateway, metaByName, needle).length > 0
}

function rosterGatewayOptions(sources, roster) {
  const byId = new Map()

  for (const source of Array.isArray(sources) ? sources : []) {
    const id = String(source?.connectionId || '').trim()

    if (id) {
      byId.set(id, { ...source, connectionId: id, count: 0 })
    }
  }

  for (const bot of roster || []) {
    const id = String(bot?.connectionId || '').trim()

    if (!id) {
      continue
    }

    const source = byId.get(id) || {
      connectionId: id,
      kind: bot.connectionKind,
      label: bot.connectionLabel || id,
      reachable: bot.sourceReachable,
      error: bot.sourceError,
      count: 0
    }
    source.count += 1
    byId.set(id, source)
  }

  return [...byId.values()].sort((a, b) =>
    String(a.label || a.connectionId).localeCompare(String(b.label || b.connectionId), undefined, {
      sensitivity: 'base'
    })
  )
}

function rosterGatewaySections(botRows, gatewayOptions, gatewayFilter = 'all') {
  const rows = Array.isArray(botRows) ? botRows : []
  const options = Array.isArray(gatewayOptions) ? gatewayOptions : []

  if (gatewayFilter !== 'all' || options.length <= 1) {
    return { sectioned: false, sections: [{ id: 'all', option: null, rows }] }
  }

  const byId = new Map()

  for (const row of rows) {
    const bot = row?.bot || row
    const id = String(bot?.connectionId || 'legacy').trim() || 'legacy'
    const bucket = byId.get(id) || []
    bucket.push(row)
    byId.set(id, bucket)
  }

  const known = new Set()
  const sections = []

  for (const option of options) {
    const id = String(option?.connectionId || '').trim()
    const sectionRows = byId.get(id)

    if (!id || !sectionRows?.length) {
      continue
    }

    known.add(id)
    sections.push({ id, option, rows: sectionRows })
  }

  for (const [id, sectionRows] of byId) {
    if (known.has(id)) {
      continue
    }

    const bot = sectionRows[0]?.bot || sectionRows[0]
    sections.push({
      id,
      option: {
        connectionId: id,
        kind: bot?.connectionKind || 'remote',
        label: bot?.connectionLabel || (id === 'legacy' ? 'Current gateway' : id),
        reachable: bot?.sourceReachable,
        error: bot?.sourceError
      },
      rows: sectionRows
    })
  }

  return { sectioned: true, sections }
}

function gatewayKindIcon(kind) {
  const icons = (typeof sdk === 'undefined' ? null : sdk.icons) || {}

  if (kind === 'local') return icons.Monitor
  if (kind === 'cloud') return icons.Cloud
  if (kind === 'ssh') return icons.Terminal
  return icons.Network
}

function gatewayKindCodicon(kind) {
  if (kind === 'local') return 'device-desktop'
  if (kind === 'cloud') return 'cloud'
  if (kind === 'ssh') return 'terminal'
  return 'remote-explorer'
}

/** Match the gateway switcher's Tabler glyphs while keeping older SDK shells
 * usable until they expose the shared icon namespace. */
function GatewayKindGlyph({ className, kind }) {
  const Icon = gatewayKindIcon(kind)

  return jsx('span', {
    'aria-hidden': true,
    className: cn('grid size-3.5 shrink-0 place-items-center', className),
    'data-connection-kind': kind || 'remote',
    'data-slot': 'connection-glyph',
    children: Icon
      ? jsx(Icon, { className: 'size-3' })
      : jsx(Codicon, { name: gatewayKindCodicon(kind), className: 'text-[0.75rem]' })
  })
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 64)
}

/** Flatten markdown syntax out of a one-line roster preview so rows read
 *  like Discord's — no raw **bold**, `code`, > quotes, or [link](url)
 *  characters in the preview line. */
function stripPreviewMarkdown(text) {
  return String(text || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`\n]*)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(^|\s)[*_](\S(?:.*?\S)?)[*_](?=\s|$|[.,;:!?])/g, '$1$2')
    .replace(/~~(.*?)~~/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Canonical multi-group read with legacy scalar compatibility. Profiles that
 *  predate `groups` still fall back to `group`; once the canonical array exists,
 *  it is authoritative. Writes keep `group` as a first-membership projection so
 *  older desktops can still display one room without corrupting the array. */
function botGroups(meta) {
  const groups = []
  const seen = new Set()
  const values = Array.isArray(meta?.groups) ? meta.groups : [meta?.group]

  for (const value of values) {
    if (typeof value !== 'string') {
      continue
    }

    const group = value.trim()

    if (group && !seen.has(group)) {
      seen.add(group)
      groups.push(group)
    }
  }

  return groups
}

function groupMembershipPatch(meta, group, enabled) {
  const name = String(group || '').trim()
  let groups = botGroups(meta)

  if (enabled) {
    if (name && !groups.includes(name)) {
      groups = [...groups, name]
    }
  } else {
    groups = groups.filter(existing => existing !== name)
  }

  return { groups, group: groups[0] || null }
}

/** Build the exact metadata cleanup for a room disband. The rendered member
 *  list is only a presentation snapshot and can already be empty on a remote
 *  connection while bot metadata still names the room. Durable room members
 *  and the full roster recover source-qualified owners; every remaining
 *  metadata record is still cleared locally, but an unresolved scoped key is
 *  never guessed into a server route. */
function groupDisbandMetadataPlan(group, members, room, roster, metaByName) {
  const owners = new Map()
  const patches = new Map()

  const rememberOwner = (owner, required = false) => {
    if (!owner?.name) {
      return
    }

    let key
    try {
      key = botMetaKey(owner)
    } catch {
      return
    }

    const meta = metaByName?.[key] || botRosterMeta(owner, metaByName) || {}
    if (!required && !botGroups(meta).includes(group)) {
      return
    }

    if (!owners.has(key)) {
      owners.set(key, owner)
    }
    patches.set(key, groupMembershipPatch(meta, group, false))
  }

  for (const owner of members || []) {
    rememberOwner(owner, true)
  }
  for (const owner of room?.members || []) {
    rememberOwner(owner, true)
  }
  for (const owner of roster || []) {
    rememberOwner(owner)
  }

  // Metadata itself is the final source of a metadata-only `0 bots` row.
  // Clear every record that names the room even when its exact server owner is
  // temporarily absent from the roster. Known owners still get a routed
  // profiles.configure write below; unknown scoped records remain local-only.
  for (const [key, meta] of Object.entries(metaByName || {})) {
    if (botGroups(meta).includes(group)) {
      patches.set(key, groupMembershipPatch(meta, group, false))
    }
  }

  return { owners, patches }
}

/** Group chats that should hold a roster row: every group named in bot meta
 *  (local members) plus every room record that still has stored members or
 *  log — cross-connection rooms whose members can't ride bot-meta. */
function groupChatNames(metaByName, rooms) {
  const names = new Set(knownGroups(metaByName))

  for (const [name, room] of Object.entries(rooms || {})) {
    if (room?.tombstone) {
      continue
    }

    if ((Array.isArray(room?.members) && room.members.length) || (Array.isArray(room?.log) && room.log.length)) {
      names.add(name)
    }
  }

  return [...names]
}

/** Names of REAL rooms in the atom — disband tombstones excluded. Feeds the
 *  create/rename collision sets so a just-disbanded name is immediately
 *  reusable even while an in-flight drive's tombstone still holds its key. */
function liveGroupChatNames() {
  return Object.entries($groupChats.get())
    .filter(([, room]) => !room?.tombstone)
    .map(([name]) => name)
}

/** Millisecond timestamp of a room's newest log entry (0 for a silent room) —
 *  the group's recency key, competing in the same ordering as bot rows. */
function groupLastActivity(room) {
  const log = Array.isArray(room?.log) ? room.log : []

  return log.length ? log[log.length - 1].at || 0 : 0
}

/** Seat a group's member roster: local bots whose meta names the group, plus
 *  the room record's stored descriptors (remote members can't ride bot-meta).
 *  Prefers the LIVE roster row for a stored descriptor when present. */
function groupChatMemberBots(group, roster, metaByName) {
  const local = (roster || []).filter(
    bot => botGroups(botRosterMeta(bot, metaByName)).includes(group)
  )
  const stored = ($groupChats.get()[group] || {}).members || []
  const seated = new Set(local.map(botRosterKey))
  const remote = []

  for (const descriptor of stored) {
    const key = botRosterKey(descriptor)

    if (seated.has(key)) {
      continue
    }

    seated.add(key)
    // A selected-but-offline ghost intentionally carries only enough identity
    // to paint the roster. Never let it replace the room's durable descriptor,
    // which owns the full handle/title used by mentions and remote sync.
    remote.push((roster || []).find(bot => !bot?.ghost && botRosterKey(bot) === key) || descriptor)
  }

  return [...local, ...remote]
}

/** Persist source-qualified identities for every selected member. The active
 *  source's row may become remote after a connection switch, so retaining it
 *  here is what keeps the same room intact across machines. */
function durableGroupChatMembers(bots) {
  return (bots || []).map(bot => {
    // Persistence pass over the whole seated roster: an orphaned member
    // (connection deleted) keeps its identity and simply persists with no
    // route — the same degraded shape the hydrate annotate produces. The
    // strict throw here would lose the entire room update over one row.
    const route = resolveBotConnectionRoute(bot).route
    // Keep the friendly identity on the stored descriptor: after a
    // connection switch the live roster row may be gone, and renamed-tag
    // mentions must still resolve against the persisted member.
    const title = String(botRosterMeta(bot, $botMeta.get())?.title || bot.ui_meta?.['hermes-bots']?.title || bot.title || '').trim()

    return {
      name: bot.name,
      handle: bot.handle || bot.name,
      ...(title ? { title } : {}),
      ...(bot.display_name ? { display_name: bot.display_name } : {}),
      connectionId: bot.connectionId,
      connectionKind: bot.connectionKind,
      connectionLabel: bot.connectionLabel,
      ...(route ? { route, targetProfile: route.targetProfile } : {}),
      // A swept/annotated member keeps its degraded mark across the rebuild —
      // otherwise the next room send would silently un-mark an orphaned row.
      ...(bot.sourceMissing ? { sourceMissing: true, sourceReachable: false } : {}),
      remoteSource: true,
      sourceScoped: Boolean(route)
    }
  })
}

/** Existing group names, alphabetical — feeds the Manage-groups dialog. */
function knownGroups(metaByName) {
  const names = new Set()

  for (const meta of Object.values(metaByName || {})) {
    for (const group of botGroups(meta)) {
      names.add(group)
    }
  }

  return [...names].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
}

// ── group chats: bounded round-robin coordination over a shared room log ─────
//
// Behavioral model (clean-room): a group conversation is ONE ordered room log
// owned by the plugin. A user send triggers at most GROUP_CHAT_MAX_ROUNDS
// serial round-robin rounds over the member roster — never parallel, no LLM
// router. Who speaks each round is a deterministic @mention parse since the
// last user message (mentioned members only, else everyone); whether a member
// actually speaks is its own turn's choice — replying with exactly "(pass)"
// (or nothing, or failing) is silence. Hard caps end every turn; a round in
// which everyone passed means the conversation settled. Each member runs its
// turn in its OWN persistent per-group Hermes session and is fed only the
// room messages that are NEW since it last saw the room.

const GROUP_CHAT_MAX_ROUNDS = 3
const GROUP_CHAT_MAX_MESSAGES = 10
const GROUP_CHAT_HISTORY_LIMIT = 24
const GROUP_CHAT_MAX_MEMBERS = 6

/** "(pass)" (loosely: pass / (pass) / pass.) or empty = the member stayed silent. */
function isGroupPassText(text) {
  const trimmed = String(text || '').trim()

  if (!trimmed) {
    return true
  }

  return /^\(?\s*pass\s*\)?\.?$/i.test(trimmed)
}

/** Deterministic @mention parse. Handles @name, @"two words" via display
 *  titles, and @everyone/@all. Names match case-insensitively against member
 *  profile names, display titles, and collapsed no-space forms. */
function parseGroupChatMentions(text, members) {
  const source = String(text || '')
  const mentioned = new Set()
  let everyone = false
  const handles = new Map()

  for (const member of members) {
    const title = String(member.title || '').trim()
    // Cross-connection members are also addressable by their @name-device
    // handle (the roster's disambiguated form) — same-named agents on two
    // machines resolve to the right one.
    const handle = String(member.handle || botHandle(member.name, member) || '').trim()
    const forms = new Set([
      member.name.toLowerCase(),
      member.name.toLowerCase().replace(/[\s_-]+/g, ''),
      ...(handle ? [handle.toLowerCase(), handle.toLowerCase().replace(/[\s_-]+/g, '')] : []),
      ...(title
        ? [title.toLowerCase(), title.toLowerCase().replace(/[\s_-]+/g, ''), title.split(/\s+/)[0].toLowerCase()]
        : [])
    ])

    // Renamed members answer to their friendly names too (profile
    // display_name and Bot Mode title), in slugged and collapsed forms —
    // the same tags the roster autocomplete inserts.
    for (const friendly of botFriendlyNames(member)) {
      for (const form of mentionNameForms(friendly)) {
        forms.add(form)
      }
    }

    for (const form of forms) {
      if (form) {
        handles.set(form, groupMemberKey(member))
      }
    }
  }

  for (const match of source.matchAll(/@([a-z0-9][a-z0-9._-]*)/gi)) {
    const handle = match[1].toLowerCase()

    if (handle === 'everyone' || handle === 'all') {
      everyone = true
      continue
    }

    if (handle === 'user') {
      continue
    }

    const resolved = handles.get(handle) || handles.get(handle.replace(/[._-]+/g, ''))

    if (resolved) {
      mentioned.add(resolved)
    }
  }

  return { everyone, mentioned }
}

/** Members that should take a turn this round: everyone when no member is
 *  @-mentioned in messages since the last user entry (or @everyone appears),
 *  otherwise only the mentioned members. Recomputed every round so a member
 *  pulled in mid-conversation joins the next round. */
function resolveGroupResponders(log, members) {
  let sinceLastUser = []

  for (let i = log.length - 1; i >= 0; i--) {
    if (log[i].from.kind === 'user') {
      sinceLastUser = log.slice(i)
      break
    }
  }

  const mentioned = new Set()
  let everyone = false

  for (const entry of sinceLastUser) {
    const parsed = parseGroupChatMentions(entry.text, members)

    if (parsed.everyone) {
      everyone = true
    }

    for (const name of parsed.mentioned) {
      mentioned.add(name)
    }
  }

  if (everyone || mentioned.size === 0) {
    return members
  }

  return members.filter(member => mentioned.has(groupMemberKey(member)))
}

/** Rotate the roster so a different member leads each round. */
function rotateGroupSpeakers(members, round) {
  if (members.length < 2) {
    return members
  }

  const shift = round % members.length

  return [...members.slice(shift), ...members.slice(0, shift)]
}

/** Transcript form of a room speaker's profile name. Friendly identity wins:
 *  a Bot Mode title or a core profile display_name (e.g. default renamed to
 *  "Lucy") labels the speaker everywhere this helper feeds — the "X is
 *  thinking…" working line, the activity feed, and transcript lines — so a
 *  renamed bot never shows up as its raw profile id or a stale "Hermes"
 *  (community report, Aug 21 2026: renamed default still read "Hermes is
 *  thinking…" in group rooms). The untitled primary profile is literally
 *  named "default" — render it as Hermes (matching displayName and the
 *  @hermes handle) so the main agent never loses its name in rooms. */
function groupSpeakerLabel(name) {
  const trimmed = (name || '').trim()

  if (!trimmed) {
    return trimmed
  }

  // Bot Mode title (edit dialog) — same first rung as displayName().
  const title = String($botMeta.get()?.[trimmed]?.title || '').trim()

  if (title) {
    return title
  }

  // Core profile display_name (`hermes profile rename …` / dashboard) from
  // the ACTIVE gateway's roster row. Source-scoped remote speakers carry
  // their device suffix separately and keep their raw name here.
  const roster = $lastRoster.get()
  const row = Array.isArray(roster)
    ? roster.find(bot => bot?.name === trimmed && !bot?.remoteSource && !bot?.sourceScoped)
    : null
  const renamed = typeof row?.display_name === 'string' ? row.display_name.trim() : ''

  if (renamed) {
    return renamed
  }

  return trimmed.toLowerCase() === 'default' ? 'Hermes' : trimmed
}

/** Room-log line as a member sees it: `Name (user): …` / `Name: …` /
 *  `Name (you): …`. */
function formatGroupChatLine(entry, viewerName) {
  // Attachments are staged into each member's session as real payloads; the
  // transcript line names them so the delta text and the bytes line up.
  const attached = Array.isArray(entry.images) && entry.images.length
    ? ` ${entry.images
        .map(img => {
          const label = img.kind === 'pdf' ? 'attached PDF' : img.kind === 'file' ? 'attached file' : 'attached image'
          return `[${label}: ${img.name || 'image'}]`
        })
        .join(' ')}`
    : ''

  if (entry.from.kind === 'user') {
    return `${entry.from.name || 'User'} (user): ${entry.text}${attached}`
  }

  const suffix = entry.from.name === viewerName ? ' (you)' : ''
  // Cross-connection speakers carry their device so same-named agents on
  // two machines stay tellable apart in every member's transcript.
  const source = entry.from.source ? ` [${entry.from.source}]` : ''

  return `${groupSpeakerLabel(entry.from.name)}${suffix}${source}: ${entry.text}${attached}`
}

/** The full per-turn payload for one member: participation rules + the room
 *  delta. Rules travel in the turn payload (not SOUL) so every existing bot
 *  can join a group chat without a profile migration. */
function buildGroupChatTurnPrompt({ groupName, members, viewer, deltaLines }) {
  const viewerKey = groupMemberKey(viewer)
  const peers = members.filter(m => groupMemberKey(m) !== viewerKey)
  const peerNames = peers
    .map(m => {
      const handle = m.title ? `${m.title} (@${botHandle(m.name, m)})` : `@${botHandle(m.name, m)}`
      return m.remoteSource ? `${handle} [on ${m.connectionLabel || m.connectionId}]` : handle
    })
    .join(', ')

  return [
    `[Group chat: "${groupName}"] You are @${botHandle(viewer.name, viewer)}, one participant in a group chat with ${peerNames || 'no one else yet'} and the user.`,
    '',
    'New messages in the room since your last turn (oldest first):',
    ...deltaLines.map(line => `  ${line}`),
    '',
    'Rules for this room:',
    '- Reply with ONE conversational message ONLY if you have something new worth adding: build on what was just said, claim or hand off work, answer a question aimed at you, or report a real result. Keep chatter short (1-3 sentences) — but when you are delivering a result, an answer the user asked for, or substantive work, give it at full quality and length; never thin out real content to fit the room.',
    '- If you have nothing new to add, reply with exactly "(pass)". Passing is good — it lets the conversation settle.',
    '- Mention a teammate as @name to pull them in; mention @user only for a judgment call or a result the user needs. Do not repeat points already made.',
    '- Never reveal content from your private 1:1 chats. Your reply text goes to the room verbatim — no preamble, no meta-commentary.'
  ].join('\n')
}

/** Trim a room log + its watermarks to the retained window, keeping
 *  watermark indices consistent with the trimmed array. */
function trimGroupChatLog(log, watermarks, limit = GROUP_CHAT_HISTORY_LIMIT * 4) {
  if (log.length <= limit) {
    return { log, watermarks }
  }

  const drop = log.length - limit
  const trimmed = {}

  for (const [name, index] of Object.entries(watermarks || {})) {
    trimmed[name] = Math.max(0, index - drop)
  }

  return { log: log.slice(drop), watermarks: trimmed }
}

/** Mutate one group's room state through the atom + persist the durable part. */
function updateGroupChat(group, mutate, { sync = true } = {}) {
  const all = { ...$groupChats.get() }
  const current = all[group] || { log: [], watermarks: {}, epoch: 0, running: false }
  const next = mutate({ ...current, log: [...current.log], watermarks: { ...current.watermarks } })
  const bounded = trimGroupChatLog(next.log, next.watermarks)

  next.log = bounded.log
  next.watermarks = bounded.watermarks
  all[group] = next
  $groupChats.set(all)

  try {
    const durable = {}

    for (const [name, room] of Object.entries(all)) {
      // Disband tombstones are runtime-only coordination state (they hold the
      // epoch bump for an in-flight drive). Persisting one would resurrect
      // the room as an empty record on the next load AND keep its name
      // "taken" for same-name recreates.
      if (room.tombstone) {
        continue
      }

      durable[name] = {
        log: room.log,
        watermarks: room.watermarks,
        sessions: room.sessions || {},
        sessionOwners: room.sessionOwners || {},
        // Timed-out turns awaiting a late reply — keyed by member, valued
        // with the pre-turn message baseline. Survives reloads so finished
        // work is still harvested after a window restart.
        stranded: room.stranded || {},
        // #93129: sticky per-member stop holds. Watermarks persist, so holds
        // must too — otherwise a window restart silently releases a bot the
        // user explicitly stopped.
        holds: room.holds || {},
        // Source-qualified member descriptors keep the room whole when the
        // active connection changes and today's local members become remote.
        members: Array.isArray(room.members) ? room.members : [],
        // Immutable room identity: the member-session title for new rooms.
        roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
        // Room picture (small data URL, same normalization as bot avatars).
        image: room.image || null,
        syncRevision: Math.max(0, Number(room.syncRevision || 0))
      }
    }

    Promise.resolve(pluginCtx?.storage?.set?.('group-chats', durable)).catch(() => undefined)
  } catch {
    /* storage unavailable — room survives for this window only */
  }
  if (sync) {
    scheduleGroupChatServerSync(all, { changedRooms: [group] })
  }

  return next
}

/** Soft-disband a group chat: remove only this group from every local member's
 *  membership list (the metadata syncs cross-machine via ui_meta), drop the
 *  room log from the atom + plugin storage, and close the room view if it's
 *  open. Other group memberships and the members' per-group gateway sessions
 *  ("Group: <roomId>", or legacy "Group: <name>") are intentionally KEPT. */
async function disbandGroupChat(group, members) {
  // Invalidate any in-flight round-robin FIRST: bump the epoch so a running
  // drive bails at its next member boundary instead of appending to a room
  // the user just discarded.
  const all = { ...$groupChats.get() }
  const prior = all[group] || {}
  const metaBefore = $botMeta.get()
  const cleanup = groupDisbandMetadataPlan(group, members, prior, $lastRoster.get(), metaBefore)
  let metadataPersistence = Promise.resolve()

  if (cleanup.patches.size) {
    const nextMeta = { ...metaBefore }

    for (const [key, patch] of cleanup.patches) {
      nextMeta[key] = { ...(nextMeta[key] || {}), ...patch }
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
    all[group] = { log: [], watermarks: {}, sessions: {}, epoch: (prior.epoch || 0) + 1, running: false, tombstone: true }
  }

  $groupChats.set(all)

  if ($groupChatWorkspace.get() === group) {
    $groupChatWorkspace.set(null)
  }

  // Retire the room's MAIN-window tab too (host.openWorkspace path).
  closeGroupChatMainTab(group)

  const needs = { ...$groupNeedsYou.get() }

  delete needs[group]
  $groupNeedsYou.set(needs)
  clearGroupClarify(group)

  // Persist the room map WITHOUT the disbanded room so it can't come back
  // on the next window load.
  try {
    const durable = {}

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

    await Promise.resolve(pluginCtx?.storage?.set?.('group-chats', durable))
  } catch {
    /* storage unavailable — the atom reset above still empties the room */
  }
  scheduleGroupChatServerSync($groupChats.get(), { allowEmpty: true, deletedRooms: [group] })
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
    queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
  }
}

/** Set or clear a group chat's room picture (small data URL, normalized by
 *  the same pipeline as bot avatars). Persists with the room record. */
function setGroupChatImage(group, image) {
  updateGroupChat(group, room => {
    room.image = image || null
    return room
  })
}

/** Rename a group chat. The group's NAME is its identity everywhere — the
 *  room-map key, each local member's ui_meta membership list, and derived
 *  state — so a rename re-keys all of them. Member gateway sessions are kept
 *  as-is: stored sids keep resuming, so no history is lost. The room's
 *  immutable roomId (the member-session title) is preserved across the
 *  rename, so even a member whose sid is later lost falls back to the same
 *  "Group: <roomId>" title lookup instead of a fresh "Group: <new name>".
 *  Returns the new name, or null when the target name is taken. */
async function renameGroupChat(oldName, newName, members) {
  const next = String(newName || '').trim().slice(0, 64)

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
    host.notify({ kind: 'error', message: `A group named “${next}” already exists.` })
    return null
  }

  // Move the room record wholesale — log, watermarks, sessions, members,
  // picture, and runtime flags all belong to the same room under its new name.
  const all = { ...$groupChats.get() }
  const room = all[oldName]

  if (room) {
    migrateGroupComposerDraft(groupComposerDraftKey(oldName, room), groupComposerDraftKey(next, room))
  }

  delete all[oldName]

  if (room) {
    all[next] = room
  }

  $groupChats.set(all)

  const needs = { ...$groupNeedsYou.get() }

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

    await saveBotMeta(member, { groups, group: groups[0] || null })
  }

  // Persist the re-keyed map (updateGroupChat writes the whole durable map).
  updateGroupChat(next, r => r, { sync: false })
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
    queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
  }

  return next
}

function groupChatEntryId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return globalThis.crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function appendGroupChatEntry(group, from, text, thread, images) {
  const entry = {
    id: groupChatEntryId(),
    at: Date.now(),
    from,
    text: String(text).trim(),
    thread: thread || 'legacy'
  }

  if (Array.isArray(images) && images.length) {
    // [{ name, data }] — data URLs. Persisted with the room log so reloads
    // keep showing what the members were shown.
    entry.images = images
  }

  // #93127 insurance: a residual double-append path (stale loop + fresh
  // loop both committing the same member reply) lands back-to-back and
  // byte-identical. Drop the echo instead of flooding the room. User
  // entries and non-adjacent repeats are never touched.
  const priorLog = ($groupChats.get()[group] || {}).log || []
  const lastEntry = priorLog[priorLog.length - 1]

  if (isDuplicateGroupAppend(lastEntry, from, entry.text, entry.thread)) {
    return lastEntry
  }

  updateGroupChat(group, room => {
    room.log.push(entry)
    return room
  })

  // Needs-you: a member addressing @user badges the group header.
  if (from.kind === 'member' && /@user\b/i.test(entry.text)) {
    $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: true })
  }

  return entry
}

/** Fresh room identity for a group. Independent of the editable display
 *  name: a disbanded-and-recreated group mints a new roomId even when the
 *  display name is identical, so member sessions never resume by title. */
function mintGroupRoomId() {
  return `r${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

/** Unique display name for a NEW group. Collisions get a " 2", " 3", …
 *  suffix; the BASE is truncated (never the joined string), so a 64-char
 *  base keeps its suffix instead of colliding with the original. */
function uniqueGroupChatName(base, taken) {
  if (!taken.has(base)) {
    return base
  }

  for (let n = 2; n < 100; n++) {
    const suffix = ` ${n}`
    const candidate = base.slice(0, 64 - suffix.length) + suffix

    if (!taken.has(candidate)) {
      return candidate
    }
  }

  throw new Error('No free name for the group.')
}

/** Ensure the member's per-group session exists and return a LIVE runtime
 *  session id for it. Gateway-native: session.create mints the session
 *  (lazy until its first message), session.resume by stored id — or by
 *  title, which also covers rehydrated rooms whose sid was lost — reopens
 *  it after restarts. Cross-connection members route to their OWN source
 *  via requestForBot; the window's gateway never switches. */
async function ensureGroupChatSession(group, member) {
  const room = $groupChats.get()[group] || {}
  // New rooms title member sessions by their immutable roomId so a
  // same-name recreate never resumes the old room's sessions by title;
  // legacy rooms without a roomId fall back to the display name.
  const title = `Group: ${room.roomId || group}`
  const key = groupMemberKey(member)
  const known = room.sessions && room.sessions[key]

  // Try resuming what we know (stored sid first, then title lookup).
  //
  // FAIL CLOSED on a transient lookup failure — mirrors the sibling fix in
  // findExistingCanonicalChat (87b645f52c). session.resume signals "this
  // target genuinely doesn't exist" with JSON-RPC code 4007; every other
  // failure (network blip, the backend still warming up after a restart,
  // an oversized-resume refusal) means the real session might still be
  // there and must not be read as "no session, mint a new one" — that
  // forks the member's real history, and the fork silently overwrites
  // room.sessions[key] so the old session becomes unreachable from the
  // room. Only a genuine 4007 on BOTH targets means there truly is nothing
  // to resume yet, so the loop falls through to session.create below.
  for (const target of [known, title]) {
    if (!target || target === true) {
      continue
    }

    try {
      const res = await requestForBot(member, 'session.resume', {
        session_id: target,
        profile: member.name,
        omit_messages: true
      })

      if (res?.session_id) {
        const stored = res.session_key || known

        if (stored) {
          updateGroupChat(group, current => {
            current.sessions = { ...(current.sessions || {}), [key]: stored }
            current.sessionOwners = { ...(current.sessionOwners || {}), [key]: groupSessionOwner(member) }
            return current
          })
        }

        return { runtime: res.session_id, stored }
      }
    } catch (error) {
      if (error?.code !== 4007) {
        const detail = error instanceof Error && error.message ? ` (${error.message})` : ''
        throw new Error(`Could not check ${member?.name || 'member'}'s group session${detail} — not starting a new one`)
      }
      /* genuinely doesn't exist (4007) — try the next target / fall through to create */
    }
  }

  const created = await requestForBot(member, 'session.create', {
    profile: member.name,
    title,
    // Room member sessions are plumbing — always hidden from the sidebar.
    hidden: true
  })
  const stored = created?.stored_session_id || null

  if (stored) {
    updateGroupChat(group, r => {
      r.sessions = { ...(r.sessions || {}), [key]: stored }
      r.sessionOwners = { ...(r.sessionOwners || {}), [key]: groupSessionOwner(member) }
      return r
    })
  }

  return { runtime: created?.session_id || null, stored }
}

const GROUP_TURN_TIMEOUT_MS = 180000
const GROUP_TURN_POLL_MS = 2000

// --- group-turn session-lease helpers (#93602) ------------------------------
// A member turn is a session-scoped RPC SEQUENCE (resume → attach → submit →
// poll) issued with the runtime id its first RPC minted. requestForBot routes
// each RPC through a per-request socket lease (retained:false secondaries in
// store/gateway), so between two RPCs the refcount can hit 0, the leased
// socket closes, the gateway detaches the runtime session on WS disconnect,
// and the orphan reaper frees it — the next RPC then fails 4001 "not in
// memory" and the bot goes silent in the room.

/** 4001-class "the runtime session was reaped" failure. Distinct from 4007
 *  ("genuinely never existed"), which must keep flowing to session.create. */
function isSessionGoneError(error) {
  if (!error || error.code === 4007) {
    return false
  }

  if (error.code === 4001) {
    return true
  }

  // Duck-typed (not instanceof): gateway errors can cross realm boundaries.
  const message = typeof error?.message === 'string' ? error.message : typeof error === 'string' ? error : ''

  return message.includes('not in memory') || /session not found/i.test(message)
}
// --- end group-turn session-lease helpers ---

/** Hold the member's pooled socket open for the WHOLE turn. Feature-detected:
 *  hosts without retainProfile (or members on the active gateway, which never
 *  closes mid-turn) get a no-op release. A failed acquire must not kill the
 *  turn — the catch-retry on submit still covers the race. */
async function retainGroupTurnRoute(member) {
  const noop = () => undefined

  let route = null

  try {
    route = botConnectionRoute(member)
  } catch {
    return noop
  }

  if (!route || typeof host.retainProfile !== 'function') {
    return noop
  }

  try {
    const release = await host.retainProfile(route)

    return typeof release === 'function' ? release : noop
  } catch {
    return noop
  }
}

/** prompt.submit with one belt-and-braces retry: when the runtime session was
 *  reaped between minting and submitting (4001 class), re-resume via the
 *  STORED id — the durable identity — to mint a fresh runtime id, and submit
 *  exactly once more. Returns the runtime id the submit actually landed on so
 *  the poll loop keeps a live fallback target. */
async function submitGroupTurnPrompt(member, runtime, stored, text) {
  try {
    await requestForBot(member, 'prompt.submit', { session_id: runtime, text })

    return runtime
  } catch (error) {
    if (!isSessionGoneError(error) || !stored) {
      throw error
    }

    const res = await requestForBot(member, 'session.resume', {
      session_id: stored,
      profile: member.name,
      omit_messages: true
    })
    const fresh = res?.session_id

    if (!fresh) {
      throw error
    }

    await requestForBot(member, 'prompt.submit', { session_id: fresh, text })

    return fresh
  }
}

// A member turn that is VISIBLY still working (session reports
// inflight/running) keeps its slot alive up to this hard cap. The base
// timeout alone silently dropped long real turns: a 7-minute research run
// timed out at 3 minutes, read as a pass, and its finished result never
// reached the room (db's Aug 2026 report).
const GROUP_TURN_HARD_CAP_MS = 20 * 60000

/** Mirror a member's pending prompt — clarify question OR command approval —
 *  from its resume snapshot into the room store, keyed
 *  `${group}::${memberKey}` (#90694). Returns true while a prompt is
 *  blocking, so the turn poll can extend its deadline — a waiting prompt
 *  must not be eaten by the group-turn timeout. Feature-detected: older
 *  backends without `pending_clarify`/`pending_approval` in the resume
 *  payload always sync to "no prompt". Clarify wins when both are somehow
 *  present (approvals resolve inside tool batches; clarify is the outer
 *  blocker). */
function syncGroupClarify(group, member, state) {
  const key = `${group}::${groupMemberKey(member)}`
  const clarify = state && typeof state.pending_clarify === 'object' ? state.pending_clarify : null
  const approval = state && typeof state.pending_approval === 'object' ? state.pending_approval : null
  const pending = clarify || approval
  const requestId = pending?.request_id || null
  const all = $groupClarify.get()
  const current = all[key]

  if (!requestId) {
    if (current) {
      const next = { ...all }
      delete next[key]
      $groupClarify.set(next)
    }

    return false
  }

  // Same request already mirrored — keep the object identity so the card
  // doesn't lose its draft to a re-render.
  if (current?.requestId === requestId) {
    return true
  }

  const base = {
    requestId,
    group,
    member: member.name,
    memberKey: groupMemberKey(member),
    // approval.respond keys on the session, not just the request — carry the
    // runtime id the snapshot came from.
    sessionId: state?.session_id || null,
    at: Date.now()
  }

  $groupClarify.set({
    ...all,
    [key]: clarify
      ? {
          ...base,
          kind: 'clarify',
          question: typeof clarify.question === 'string' ? clarify.question : '',
          choices: Array.isArray(clarify.choices) ? clarify.choices.filter(c => typeof c === 'string' && c) : [],
          multiSelect: Boolean(clarify.multi_select),
          // Batch clarifies carry `questions`; the room card answers them
          // one wire call per question, mirroring the 1:1 batch contract.
          questions: Array.isArray(clarify.questions) ? clarify.questions : null
        }
      : {
          ...base,
          kind: 'approval',
          question: typeof approval.description === 'string' ? approval.description : '',
          command: typeof approval.command === 'string' ? approval.command : '',
          // The server precomputes the choice set from allow_permanent
          // (once/session/always/deny); fall back to the minimal pair.
          choices: Array.isArray(approval.choices) && approval.choices.length
            ? approval.choices.filter(c => typeof c === 'string' && c)
            : ['once', 'deny'],
          multiSelect: false,
          questions: null
        }
  })
  // A blocked member is a question for the human — badge the room.
  $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: true })

  return true
}

/** Drop every mirrored clarify belonging to `group` (disband/rename). */
function clearGroupClarify(group) {
  const all = $groupClarify.get()
  const next = {}
  let changed = false

  for (const [key, value] of Object.entries(all)) {
    if (value?.group === group) {
      changed = true
    } else {
      next[key] = value
    }
  }

  if (changed) {
    $groupClarify.set(next)
  }
}

/** Answer a member's pending prompt from the room. Routes to the member's
 *  OWN source (requestForBot), so cross-connection members work.
 *  - clarify: `clarify.respond`; batch questions send one respond per
 *    question, sequentially — the LAST lock resolves the blocked tool
 *    server-side (same contract as the 1:1 batch card). allow_expired
 *    server-side makes racing the timeout harmless.
 *  - approval: `approval.respond` with the choice (once/session/always/deny),
 *    keyed by session + request_id — the same wire the 1:1 approval card
 *    and native notifications use. */
async function answerGroupClarify(entry, member, answers) {
  if (entry.kind === 'approval') {
    await requestForBot(member, 'approval.respond', {
      session_id: entry.sessionId || undefined,
      request_id: entry.requestId,
      choice: typeof answers === 'string' && answers ? answers : 'deny'
    })
  } else if (entry.questions && entry.questions.length) {
    for (const question of entry.questions) {
      const qid = question?.qid ?? question?.id
      await requestForBot(member, 'clarify.respond', {
        request_id: entry.requestId,
        question_id: qid,
        answer: answers?.[qid] ?? ''
      })
    }
  } else {
    await requestForBot(member, 'clarify.respond', {
      request_id: entry.requestId,
      answer: typeof answers === 'string' ? answers : ''
    })
  }

  const all = $groupClarify.get()
  const key = `${entry.group}::${entry.memberKey}`

  if (all[key]?.requestId === entry.requestId) {
    const next = { ...all }
    delete next[key]
    $groupClarify.set(next)
  }
}

/** One member turn, gateway-native: submit the room delta as a prompt into
 *  the member's per-group session, then poll the session until a NEW
 *  assistant message lands (or timeout → pass). While the session visibly
 *  reports work in flight the deadline extends (bounded by the hard cap),
 *  so slow models aren't cut off mid-run. A turn that still times out
 *  records a stranded marker so the finished reply can be harvested into
 *  the room at the member's next turn instead of being lost. */
async function runGroupChatMemberTurn(group, member, prompt, thread, images) {
  // #93602: hold the member's route socket for the whole turn. Without the
  // lease, every RPC below rides its own request-scoped socket lease; the
  // socket that minted `runtime` can close between RPCs, the gateway reaps
  // the runtime session, and prompt.submit dies 4001 — the bot goes silent.
  const releaseTurnLease = await retainGroupTurnRoute(member)

  try {
    return await runGroupChatMemberTurnLeased(group, member, prompt, thread, images)
  } finally {
    releaseTurnLease()
  }
}

async function runGroupChatMemberTurnLeased(group, member, prompt, thread, images) {
  const { runtime, stored } = await ensureGroupChatSession(group, member)

  if (!runtime) {
    return null
  }

  recordGroupActivity(group, { kind: 'working', member: member.name, thread })

  // Baseline: how many messages exist before our submit.
  let before = 0

  try {
    const pre = await requestForBot(member, 'session.resume', {
      session_id: stored || runtime,
      profile: member.name
    })
    before = Array.isArray(pre?.messages) ? pre.messages.length : pre?.message_count || 0
  } catch {
    /* lazy session — zero messages */
  }

  // Stage this delta's attachments into the member's session so the model
  // receives the actual payload with the prompt — the same attach RPCs the
  // 1:1 chat uses (they also work cross-connection, where the member's
  // gateway can't see this machine's files). Images queue as vision tiles,
  // PDFs render per-page via pdf.attach, and other files materialize in the
  // session workspace (their @file: refs are appended to the prompt so the
  // member's file tools can read them). A failed attach degrades that
  // member to text-only; the transcript line still names the attachment so
  // the member knows something was shared.
  const fileRefs = []

  for (const img of Array.isArray(images) ? images : []) {
    if (!img || typeof img.data !== 'string' || !img.data) {
      continue
    }

    try {
      if (img.kind === 'pdf') {
        await requestForBot(member, 'pdf.attach', {
          session_id: runtime,
          content_base64: img.data,
          filename: img.name || 'attachment.pdf'
        })
      } else if (img.kind === 'file') {
        const res = await requestForBot(member, 'file.attach', {
          session_id: runtime,
          data_url: img.data,
          name: img.name || 'attachment'
        })

        if (res?.ref_text) {
          fileRefs.push(`${img.name || 'attachment'} → ${res.ref_text}`)
        }
      } else {
        await requestForBot(member, 'image.attach_bytes', {
          session_id: runtime,
          content_base64: img.data,
          filename: img.name || 'attachment.png'
        })
      }
    } catch {
      /* text-only fallback for this member */
    }
  }

  const turnText = fileRefs.length
    ? `${prompt}\n\nAttached files staged in your session workspace:\n${fileRefs.join('\n')}`
    : prompt

  // #93602: one-shot recovery when the runtime session was reaped between
  // minting and submitting. Tracks the runtime id the submit landed on so
  // the poll fallback below targets a live session.
  const liveRuntime = await submitGroupTurnPrompt(member, runtime, stored, turnText)

  const started = Date.now()
  let deadline = started + GROUP_TURN_TIMEOUT_MS

  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, GROUP_TURN_POLL_MS))

    let state = null

    try {
      state = await requestForBot(member, 'session.resume', {
        session_id: stored || liveRuntime,
        profile: member.name
      })
    } catch {
      continue
    }

    const messages = Array.isArray(state?.messages) ? state.messages : []
    const busy = Boolean(state?.inflight || state?.running)
    // A clarify blocking inside the member's session is a question for the
    // HUMAN (#90694) — mirror it into the room store so a card renders, and
    // hold the turn open: the member isn't stalling, it's waiting on us.
    const awaitingUser = syncGroupClarify(group, member, state)
    const done = !busy && !awaitingUser

    if (messages.length > before && done) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]

        if (msg?.role === 'assistant') {
          const text = typeof msg.content === 'string'
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content.map(p => (typeof p === 'string' ? p : p?.text || '')).join('')
              : msg?.text || ''
          const replyText = String(text).trim()

          recordGroupActivity(group, {
            kind: isGroupPassText(replyText) ? 'passed' : 'replied',
            member: member.name,
            thread
          })

          return replyText
        }
      }

      recordGroupActivity(group, { kind: 'passed', member: member.name, thread })

      return null
    }

    // Still visibly working — or waiting on the user's answer to a clarify:
    // extend the deadline (never past the hard cap). A pending question must
    // outlive the base turn timeout or it dies unanswered at 3 minutes.
    if (busy || awaitingUser) {
      deadline = Math.min(started + GROUP_TURN_HARD_CAP_MS, Math.max(deadline, Date.now() + GROUP_TURN_TIMEOUT_MS))
    }
  }

  // Timeout — clear any still-mirrored question card (the server-side
  // clarify timeout runs its own course) and read as a pass, but remember the baseline + thread
  // (runtime-only) so the finished reply can be posted late into the RIGHT
  // thread instead of vanishing.
  recordGroupActivity(group, { kind: 'timed-out', member: member.name, thread })
  syncGroupClarify(group, member, null)
  updateGroupChat(group, r => {
    r.stranded = { ...(r.stranded || {}), [groupMemberKey(member)]: { before, thread } }
    return r
  })

  return null
}

/** Post a timed-out member's finished reply into the room, if it landed
 *  after we stopped waiting. Called at the member's next turn boundary and
 *  on user sends, so long-running work is delivered late rather than lost. */
async function harvestStrandedGroupReply(group, member) {
  const memberKey = groupMemberKey(member)
  const room = $groupChats.get()[group] || {}
  const marker = room.stranded?.[memberKey]
  // Markers were a bare number before threads; normalize both shapes.
  const strandedBefore = typeof marker === 'number' ? marker : marker?.before
  const strandedThread = (typeof marker === 'object' && marker?.thread) || 'legacy'

  if (typeof strandedBefore !== 'number') {
    return
  }

  let state = null

  try {
    const stored = room.sessions?.[memberKey]
    state = await requestForBot(member, 'session.resume', {
      session_id: stored || `Group: ${room.roomId || group}`,
      profile: member.name
    })
  } catch {
    return // source unreachable — leave the marker for the next boundary
  }

  if (state?.inflight || state?.running) {
    return // still grinding — keep waiting
  }

  // A stranded member blocked on a clarify is not "grinding" — surface the
  // question card (#90694) and keep the marker until it resolves.
  if (syncGroupClarify(group, member, state)) {
    return
  }

  // Done (or dead): the marker is consumed either way.
  updateGroupChat(group, r => {
    const next = { ...(r.stranded || {}) }
    delete next[memberKey]
    r.stranded = next
    return r
  })

  const messages = Array.isArray(state?.messages) ? state.messages : []

  if (messages.length <= strandedBefore) {
    return
  }

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]

    if (msg?.role === 'assistant') {
      const text = typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map(p => (typeof p === 'string' ? p : p?.text || '')).join('')
          : msg?.text || ''
      const reply = String(text).trim()

      if (reply && !isGroupPassText(reply)) {
        recordGroupActivity(group, { kind: 'delivered', member: member.name, thread: strandedThread })
        appendGroupChatEntry(
          group,
          { kind: 'member', name: member.name, ...(member.remoteSource ? { source: member.connectionLabel || member.connectionId } : {}) },
          reply,
          strandedThread
        )
        updateGroupChat(group, r => {
          r.watermarks[`${strandedThread}::${memberKey}`] = r.log.length
          return r
        })
      }

      return
    }
  }
}

// --- room-turn decision helpers (#93127) — pure, vm-sliced by tests ---

/** #93127: whether a finished member turn may still commit (append its reply
 *  and advance its watermark). A turn dispatched under an older epoch was
 *  superseded mid-flight by a newer user send — its late result must be
 *  dropped, because the new send's own loop re-drives this member with the
 *  full delta and committing both is exactly the double-delivery bug.
 *
 *  The re-drive premise is only true for a send in the SAME thread (delta
 *  filters are thread-scoped): a cross-thread epoch bump must NOT discard
 *  finished work no fresh loop will regenerate. Callers pass whether a newer
 *  USER entry landed in this thread since dispatch; the default (true)
 *  preserves the conservative drop when the caller can't tell. */
function shouldCommitMemberTurn(epochAtDispatch, currentEpoch, newerUserEntryInThread = true) {
  if (epochAtDispatch === currentEpoch) {
    return true
  }

  return !newerUserEntryInThread
}

/** #93127 insurance: byte-identical member echo detection. TRUE only when
 *  the immediately-preceding log entry has the same author (kind + name +
 *  source), same thread, and identical text, within a short recency window —
 *  a residual double-append fires back-to-back; two legitimately identical
 *  replies hours apart (or with anything in between) are never dropped. */
const GROUP_DUPLICATE_APPEND_WINDOW_MS = 10 * 60 * 1000

function isDuplicateGroupAppend(lastEntry, from, text, thread, now = Date.now()) {
  if (!lastEntry || !from || from.kind !== 'member' || lastEntry.from?.kind !== 'member') {
    return false
  }

  if (String(lastEntry.from?.name || '') !== String(from.name || '')) {
    return false
  }

  if (String(lastEntry.from?.source || '') !== String(from.source || '')) {
    return false
  }

  if (String(lastEntry.thread || 'legacy') !== String(thread || 'legacy')) {
    return false
  }

  if (now - (lastEntry.at || 0) > GROUP_DUPLICATE_APPEND_WINDOW_MS) {
    return false
  }

  return String(lastEntry.text || '') === String(text || '').trim()
}

// --- end room-turn decision helpers ---

// --- member-hold helpers (#93129) — pure, vm-sliced by tests ---

/** #93129: classify a USER room message's effect on member holds. Only user
 *  sends ever reach this (bot replies are appended by the round loop, never
 *  through sendToGroupChat), so a bot saying "stopped working on it" can
 *  never set a hold. Conservative on purpose: any standalone stop/halt/pause
 *  word next to a mention holds those members — "don't stop @x" therefore
 *  also holds, which errs toward the bot staying quiet until re-addressed
 *  (a wrongly-held bot is one mention away from release; a wrongly-running
 *  one keeps doing work it was told to stop). A non-stop direct mention
 *  releases the mentioned members — the user addressing a bot directly
 *  overrides its hold. */
function classifyGroupHoldDirective(text, mentionedKeys, everyone) {
  const value = String(text || '')
  const mentioned = [...(mentionedKeys || [])]
  const stop = /\b(stop|halt|pause)\b/i.test(value)
  const resume = /\b(resume|continue|go|proceed)\b/i.test(value)

  if (stop) {
    // "@all stop" holds every member — symmetric with "@all resume".
    return { hold: mentioned, holdAll: Boolean(everyone), release: [], releaseAll: false }
  }

  if (resume) {
    return { hold: [], holdAll: false, release: mentioned, releaseAll: Boolean(everyone) }
  }

  return { hold: [], holdAll: false, release: mentioned, releaseAll: false }
}

/** #93129: next holds map after one user message. Holds are keyed by
 *  memberKey at ROOM scope (not thread scope): every main-composer send
 *  mints a NEW thread, so a thread-scoped hold would never block the next
 *  send's turns and the stop would not stick. Returns the same object when
 *  nothing changed. */
function applyGroupHoldDirective(holds, mentions, text, stamp, allMemberKeys = []) {
  const prior = holds && typeof holds === 'object' ? holds : {}
  const action = classifyGroupHoldDirective(text, mentions?.mentioned || [], Boolean(mentions?.everyone))

  if (action.releaseAll) {
    return Object.keys(prior).length ? {} : prior
  }

  // "@all stop": expand to every member key the caller knows about.
  const toHold = action.holdAll ? [...allMemberKeys] : action.hold

  let next = prior

  for (const key of toHold) {
    if (next === prior) {
      next = { ...prior }
    }

    next[key] = { at: stamp?.at || Date.now(), byMessageId: stamp?.byMessageId || null, thread: stamp?.thread || null }
  }

  for (const key of action.release) {
    if (Object.prototype.hasOwnProperty.call(next, key)) {
      if (next === prior) {
        next = { ...prior }
      }

      delete next[key]
    }
  }

  return next
}

/** #93129: a held member's skip must consume its delta exactly once —
 *  advance the watermark past the current log so the same entries never
 *  re-trigger the skip. Null = nothing to consume (no write, no spin). */
function heldMemberWatermarkAdvance(seen, logLength) {
  return logLength > (seen || 0) ? logLength : null
}

// --- end member-hold helpers ---

/** Drive one bounded round-robin turn for ONE THREAD. Serial — one member at
 *  a time. A newer user send bumps the room epoch; this loop notices at the
 *  next member boundary, bails, and the newest send's own loop takes over.
 *  Watermarks are per thread+member (`${thread}::${memberKey}`), so parallel
 *  topics never eat each other's deltas. */
async function runGroupChatRounds(group, members, thread) {
  const startEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const isCurrent = () => (($groupChats.get()[group] || {}).epoch || 0) === startEpoch
  let posted = 0

  try {
    for (let round = 0; round < GROUP_CHAT_MAX_ROUNDS; round++) {
      // Deliver any replies that finished after their turn timed out —
      // every member, not just this round's responders, so long work is
      // late, never lost.
      for (const member of members) {
        if (!isCurrent()) {
          recordGroupActivity(group, { kind: 'cancelled', member: null, thread })
          return
        }

        await harvestStrandedGroupReply(group, member)
      }

      const roomLog = (($groupChats.get()[group] || {}).log || []).filter(e => groupThreadOf(e) === thread)
      // Exclude members the harvest pass just above confirmed are STILL
      // running (their stranded marker survived harvest because
      // state.inflight/running was true). Re-selecting one here would
      // prompt.submit into their live session — the gateway's default busy
      // policy redirects or hard-interrupts that turn (tui_gateway's
      // _handle_busy_submit), killing exactly the long-running work this
      // stranded/harvest mechanism exists to protect. Skip them; the next
      // harvest pass picks the reply up once it actually lands. A marker's
      // mere presence means "still stranded" (harvestStrandedGroupReply
      // deletes it once the member is confirmed done/dead) — presence, not
      // value shape, since markers are a bare number pre-thread or
      // {before, thread} post-thread.
      const strandedNow = ($groupChats.get()[group] || {}).stranded || {}
      const responders = rotateGroupSpeakers(resolveGroupResponders(roomLog, members), round)
        .filter(member => !Object.prototype.hasOwnProperty.call(strandedNow, groupMemberKey(member)))
      let spokeThisRound = 0

      for (const member of responders) {
        if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES) {
          if (!isCurrent()) {
            recordGroupActivity(group, { kind: 'cancelled', member: null, thread })
          }
          return
        }

        const room = $groupChats.get()[group] || { log: [], watermarks: {} }
        const memberKey = groupMemberKey(member)
        const markKey = `${thread}::${memberKey}`
        const seen = room.watermarks[markKey] || 0
        // Delta: NEW room entries, narrowed to this thread — the member's
        // turn sees only the conversation it's part of.
        const delta = room.log.slice(seen).filter(e => groupThreadOf(e) === thread)

        if (!delta.length) {
          continue
        }

        // #93129: a member the user told to stop is HELD — no turn until an
        // explicit release (resume / @all resume / a direct non-stop
        // mention). Consume the delta exactly once (watermark past the
        // current log) so the same entries never re-trigger this skip, and
        // surface WHY the bot is silent in the activity feed the first time.
        const heldEntry = (room.holds || {})[memberKey]

        if (heldEntry) {
          const advance = heldMemberWatermarkAdvance(seen, room.log.length)

          updateGroupChat(group, r => {
            if (advance !== null) {
              r.watermarks[markKey] = advance
            }

            if (r.holds?.[memberKey] && !r.holds[memberKey].noted) {
              r.holds = { ...r.holds, [memberKey]: { ...r.holds[memberKey], noted: true } }
            }

            return r
          })

          if (!heldEntry.noted) {
            recordGroupActivity(group, { kind: 'held', member: member.name, thread })
          }

          continue
        }

        const prompt = buildGroupChatTurnPrompt({
          groupName: group,
          members,
          viewer: member,
          deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map(e => formatGroupChatLine(e, member.name))
        })

        // Images riding this delta (user attachments — member entries don't
        // carry images today, but flatMap keeps this future-proof) get staged
        // into the member's session so the model sees the pixels, not just
        // the transcript's [attached image: …] marker.
        const deltaImages = delta.flatMap(e => (Array.isArray(e.images) ? e.images : []))

        // Surface WHO is on turn (runtime-only, like running/epoch) so the
        // room shows "Radar is thinking…" instead of a generic working line —
        // long model turns otherwise read as the room being stuck.
        updateGroupChat(group, r => {
          r.turn = member.name
          return r
        })

        let reply = null

        try {
          reply = await runGroupChatMemberTurn(group, member, prompt, thread, deltaImages)

          // Needs-attention hook (#93091 item 3): a turn that produced a real
          // reply (or an explicit pass) is a good turn — clear the badge.
          // A timed-out turn also returns null but never threw; leaving any
          // prior badge in place there is the conservative choice.
          if (reply !== null) {
            clearBotAttention(groupMemberKey(member))
          }
        } catch (error) {
          recordGroupActivity(group, { kind: 'failed', member: member.name, thread })
          noteBotAttention(groupMemberKey(member), error?.message || error)
          reply = null // a failed turn is a pass, never a room error
        }

        // #93127: the turn may have finished AFTER a newer user send bumped
        // the room epoch. That newer send's loop re-drives this member with
        // the full delta, so committing this stale result (watermark advance
        // + append) would double-deliver the same reply. Drop it here —
        // BEFORE the watermark advance and BEFORE the append. Only a newer
        // USER entry in THIS thread makes the re-drive premise true: a
        // cross-thread send bumps the epoch too, but its loop filters this
        // thread out and would never regenerate the finished reply. The
        // during-turn tail is anchored by entry id, not index — the history
        // trim drops entries from the FRONT, so an index slice could
        // overshoot after a mid-turn trim and silently commit a stale turn.
        const roomNow = $groupChats.get()[group] || { log: [] }
        const epochNow = roomNow.epoch || 0
        const anchorId = room.log.length ? room.log[room.log.length - 1].id : null
        const anchorIdx = anchorId === null ? -1 : roomNow.log.findIndex(e => e.id === anchorId)
        // Anchor trimmed away ⇒ every pre-turn entry was dropped, so every
        // surviving entry is newer — scanning the whole log stays exact.
        const turnTail = anchorIdx >= 0 ? roomNow.log.slice(anchorIdx + 1) : roomNow.log
        const newerUserEntryInThread = turnTail.some(
          e => e.from?.kind === 'user' && groupThreadOf(e) === thread
        )

        if (!shouldCommitMemberTurn(startEpoch, epochNow, newerUserEntryInThread)) {
          recordGroupActivity(group, { kind: 'cancelled', member: member.name, thread })
          return
        }

        // The member has now seen everything up to the pre-reply log length.
        updateGroupChat(group, r => {
          r.watermarks[markKey] = r.log.length
          return r
        })

        if (reply !== null && !isGroupPassText(reply)) {
          appendGroupChatEntry(
            group,
            { kind: 'member', name: member.name, ...(member.remoteSource ? { source: member.connectionLabel || member.connectionId } : {}) },
            reply,
            thread
          )
          // Its own message counts as seen too.
          updateGroupChat(group, r => {
            r.watermarks[markKey] = r.log.length
            return r
          })
          posted += 1
          spokeThisRound += 1
        }
      }

      if (spokeThisRound === 0) {
        return // everyone passed — the conversation settled
      }
    }
  } finally {
    if (isCurrent()) {
      recordGroupActivity(group, { kind: 'settled', member: null, thread })
      updateGroupChat(group, r => {
        r.running = false
        r.turn = null
        return r
      })

      // #89545: the loop's harvest pass only ran at the top of each round of
      // an ACTIVE loop — a member whose turn timed out after the final round
      // stayed stranded until the user's NEXT send. Poll for the late reply
      // in the background (bounded) so long work is late, never lost.
      // (window feature-detect: the engine also runs under node in tests.)
      const strandedLeft = Object.keys(($groupChats.get()[group] || {}).stranded || {})

      if (strandedLeft.length && typeof window !== 'undefined') {
        void harvestStrandedUntilSettled(group, members, thread)
      }
    }
  }
}

/** Bounded background harvest for members whose replies outlived the turn
 *  loop. Polls every 5s for up to 5 minutes; stops early when nothing is
 *  stranded, a new loop takes the room over (it harvests on its own), or the
 *  room record disappears (disband). */
async function harvestStrandedUntilSettled(group, members, thread) {
  const HARVEST_INTERVAL_MS = 5000
  const HARVEST_MAX_TRIES = 60

  for (let attempt = 0; attempt < HARVEST_MAX_TRIES; attempt++) {
    await new Promise(resolve => window.setTimeout(resolve, HARVEST_INTERVAL_MS))

    const room = $groupChats.get()[group]

    if (!room || room.running) {
      return
    }

    const stranded = room.stranded || {}

    if (!Object.keys(stranded).length) {
      return
    }

    for (const member of members) {
      if (Object.prototype.hasOwnProperty.call(stranded, groupMemberKey(member))) {
        try {
          await harvestStrandedGroupReply(group, member)
        } catch {
          // Best-effort: the next tick retries; the bound stops runaways.
        }
      }
    }
  }

  recordGroupActivity(group, { kind: 'failed', member: null, thread })
}

/** User send into a group room. `thread` continues that thread (its reply
 *  box); omitted/null mints a NEW thread — the main composer's Slack shape.
 *  Appends, bumps the room epoch (supersedes any running loop at its next
 *  member boundary), and starts the turn drive for the target thread.
 *  Returns the thread id the message landed in. */
function sendToGroupChat(group, members, text, thread, images) {
  const trimmed = String(text || '').trim()
  const attached = Array.isArray(images) ? images.filter(img => img && img.data) : []

  if ((!trimmed && !attached.length) || !members.length) {
    return null
  }

  const target = thread || mintGroupThreadId()

  $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: false })
  // Refresh the durable room roster on every send. This backfills rooms made
  // by older Desktop builds and keeps the gateway mirror complete even when
  // members overlap across multiple groups.
  updateGroupChat(group, room => {
    room.members = durableGroupChatMembers(members)
    return room
  })
  const sent = appendGroupChatEntry(group, { kind: 'user', name: 'You' }, trimmed, target, attached)

  const wasRunning = ($groupChats.get()[group] || {}).running === true

  updateGroupChat(group, room => {
    room.epoch = (room.epoch || 0) + 1
    room.running = true
    // #93129: user text is the ONLY input that changes member holds. An
    // explicit "stop @member" sets a sticky hold; "@member resume" (or
    // @all resume, or any direct non-stop mention of the held member)
    // releases it. Bot replies never flow through this function.
    room.holds = applyGroupHoldDirective(
      room.holds,
      parseGroupChatMentions(trimmed, members),
      trimmed,
      { at: sent?.at, byMessageId: sent?.id, thread: target },
      members.map(member => groupMemberKey(member))
    )
    return room
  })

  recordGroupActivity(group, { kind: 'queued', member: 'You', thread: target })

  if (!wasRunning) {
    void runGroupChatRounds(group, members, target).catch(() => {
      updateGroupChat(group, r => {
        r.running = false
        return r
      })
    })
  } else {
    // A loop is live; it bails at its next boundary. Chain the fresh loop
    // after a short settle so exactly one drive owns the room.
    setTimeout(() => {
      void runGroupChatRounds(group, members, target).catch(() => {
        updateGroupChat(group, r => {
          r.running = false
          return r
        })
      })
    }, 250)
  }

  return target
}

/** Share one in-flight async operation across concurrent callers. Failures
 * clear the slot so a later attempt can retry. */
function singleFlight(ref, start) {
  if (ref.current) {
    return ref.current
  }

  let flight
  try {
    flight = Promise.resolve(start())
  } catch (err) {
    flight = Promise.reject(err)
  }
  ref.current = flight
  flight.catch(() => {
    if (ref.current === flight) {
      ref.current = null
    }
  })
  return flight
}

/** The agent-to-agent messaging protocol, reusable so a CUSTOM SOUL keeps
 *  the handoff protocol too — a custom SOUL used to silently drop it,
 *  breaking @mentions for customized bots (@wesleysimplicio, #16). */
function messagingProtocolSection(name, roster) {
  const teammates = (roster || []).filter(b => b.name !== name)
  const handle = botHandle(name)

  return [
    '## Messaging other agents',
    '',
    'You work alongside other named agents. Every agent (including you) has',
    'ONE canonical conversation titled "Bot Chat" — created with the agent,',
    'so it always exists. Agent-to-agent messages are delivered straight',
    'into it, like a DM. To message a teammate, run:',
    '',
    '```',
    'hermes -p <agent-name> chat --in ~ -c "Bot Chat" --create-if-missing -Q -q "Message from \uD83E\uDD16 ' + handle + ' (@' + handle + '): your message"',
    '',
    'Run the send with background=true and notify_on_complete=true on the',
    'terminal tool, then finish your turn — the reply arrives later as a',
    'background process notification. Never block waiting for it.',
    '```',
    '',
    '(`--in ~ -c "Bot Chat" --create-if-missing` resumes their canonical',
    'conversation in the home workspace, creating it if the target has no',
    '"Bot Chat" yet. `-Q` keeps output clean. Always open with the',
    '"Message from \uD83E\uDD16 ' + handle + ' (@' + handle + '):" prefix so they know',
    'who is talking (the @handle lets the app show your avatar to them).',
    'Their reply prints to stdout — relay the relevant part back to the',
    'user, and say which agent it came from.)',
    '',
    'If a message in YOUR chat starts with "Message from \uD83E\uDD16 <name>", it is',
    'a teammate messaging you, not the user. Answer it directly — your reply',
    'reaches them via their own delivery — and use the same command if you',
    'need to start a conversation yourself.',
    '',
    'When the user writes @<agent-name> or says "ask <name> to ..." /',
    '"tell <name> ...", that is a handoff: message that agent, wait for the',
    'reply, and report back.',
    '',
    'The roster grows over time — run `hermes profile list` for the LIVE',
    'teammate list before a handoff. Teammates when you were created:',
    ...(teammates.length
      ? teammates.map(b => `- \`${b.name}\`${b.description ? ` — ${b.description}` : ''}`)
      : ['- (none yet)'])
  ].join('\n')
}

/** True when SOUL.md already carries the Bot Mode handoff section.
 *  #16 appends this at create-time; pre-existing profiles (especially
 *  `default`) never went through composeSoul and silently lack it. */
function hasMessagingProtocol(soul) {
  return /(^|\n)## Messaging other agents(\s|$)/.test(soul || '')
}

/** Idempotent: append the protocol once, never duplicate a custom SOUL
 *  that already has it (clone-from-default after a backfill, Edit save).
 *  No-op when the backend injects the protocol into the system prompt
 *  itself (bot_mode_protocol) — SOUL.md stays the user's identity text. */
function ensureMessagingProtocol(soul, name, roster) {
  const text = (soul || '').trim()
  if (serverInjectsProtocol || hasMessagingProtocol(text)) return text
  const section = messagingProtocolSection(name, roster)
  return text ? text + '\n\n' + section : section
}

const soulProtocolChecked = new Set()
const soulProtocolInflight = new Set()

/** One-shot per profile per session: if an existing SOUL has no protocol,
 *  append it. This is the install-time fix for default / pre-Bot-Mode
 *  personas that #16 never touched. Never overwrites identity text. */
function backfillMessagingProtocol(roster) {
  // Newer backends teach the protocol via the system prompt — never touch
  // user SOUL files when the server already covers every session.
  if (serverInjectsProtocol) {
    return
  }

  for (const bot of roster || []) {
    const name = bot && bot.name
    if (!name || soulProtocolChecked.has(name) || soulProtocolInflight.has(name)) {
      continue
    }

    soulProtocolInflight.add(name)
    host
      .request('profiles.describe', { name })
      .then(res => {
        const soul = (res && res.soul) || ''
        if (hasMessagingProtocol(soul)) {
          soulProtocolChecked.add(name)
          return null
        }
        return host
          .request('profiles.configure', { name, soul: ensureMessagingProtocol(soul, name, roster) })
          .then(() => {
            soulProtocolChecked.add(name)
          })
      })
      .catch(() => {
        // Older gateway or a one-off describe/configure miss — do not hammer.
        soulProtocolChecked.add(name)
      })
      .finally(() => {
        soulProtocolInflight.delete(name)
      })
  }
}

/** SOUL.md for a new bot: identity (or the user's custom SOUL) + the
 *  messaging protocol — which ships UNLESS the backend injects it into the
 *  system prompt itself (bot_mode_protocol capability). */
function composeSoul({ name, title, description, roster, customSoul }) {
  if (customSoul && customSoul.trim()) {
    return ensureMessagingProtocol(customSoul, name, roster)
  }

  const lines = [
    `# ${displayName({ name, title })}`,
    '',
    title ? `**Role:** ${title}` : null,
    description ? `**Mission:** ${description}` : null,
    '',
    `You are ${displayName({ name, title })}, a persistent named agent (profile \`${name}\`) on this machine.`,
    'You keep your own memory, skills, and conversation history across sessions.'
  ]

  const identity = lines.filter(line => line !== null).join('\n')

  return serverInjectsProtocol ? identity : identity + '\n\n' + messagingProtocolSection(name, roster)
}

// ── human-readable row helpers ───────────────────────────────────────────────

/** Bot-to-bot delivery prefix (see messagingProtocolSection): either the
 *  current "Message from 🤖 name (@handle):" form or the older
 *  "[Message from agent 'name']" shape. Captures the sender's handle. */
const A2A_RE = /^Message from (?:agent '([^']+)'|🤖\s*([^\s(@]+))/i

/** Strip the delivery prefix so a DM preview reads like a DM, not a log line. */
const A2A_PREFIX_RE = /^Message from (?:agent '[^']+'|🤖[^:]+):\s*/i

/** Classify a roster preview: `{ fromBot: handle|null }`. A preview that
 *  starts with the delivery prefix is a bot-to-bot message — the receiving
 *  bot's row should show WHO sent it, not present it as the human's chat. */
function previewKind(preview) {
  const text = (preview || '').trim()
  if (!text) {
    return { fromBot: null }
  }
  const match = text.match(A2A_RE)
  if (match) {
    // The captured name is whatever the delivery prefix carried — a raw
    // profile name. Map it the way every other surface does so the primary
    // profile reads @hermes, never @default (#89484).
    const sender = (match[1] || match[2] || '').trim().toLowerCase()
    return { fromBot: sender ? botHandle(sender) : null }
  }
  return { fromBot: null }
}

/** Session titles the gateway auto-assigns that carry no information. */
const GENERIC_TITLES = new Set(['', 'bot chat', 'new chat', 'new conversation', 'conversation', 'chat', 'untitled'])

function isGenericTitle(title) {
  return GENERIC_TITLES.has((title || '').trim().toLowerCase())
}

/** Title for the session chip: the real session title when it means
 *  something, otherwise a short label generated from the newest message
 *  (delivery prefixes stripped) so "Bot Chat" rows still say what the
 *  conversation is actually about. */
function generatedSessionTitle(session, preview) {
  const raw = (session?.title || '').trim()
  if (raw && !isGenericTitle(raw)) {
    return raw
  }
  const cleaned = (preview || '').trim().replace(A2A_PREFIX_RE, '').trim()
  if (!cleaned) {
    return raw || 'Conversation'
  }
  const words = cleaned.split(/\s+/).slice(0, 5).join(' ').replace(/[,;:.]+$/, '')
  if (!words) {
    return raw || 'Conversation'
  }
  return words.length > 34 ? `${words.slice(0, 33)}…` : words
}

/** Roster liveness window: a bot whose last message landed within this many
 *  seconds is treated as "active now" (pulsing dot in its row). */
const ACTIVE_WINDOW_S = 90
const RECENT_ACTIVITY_WINDOW_S = 7 * 24 * 60 * 60
const BOT_ROSTER_SEARCH_THRESHOLD = 8

/** The session whose activity best represents this bot — the FRESHER of the
 *  canonical Bot Chat (canonical_session, the profile's "Bot Chat" registry
 *  row resolved server-side by name) and the profile's newest visible
 *  conversation (last_session).
 *
 *  Canonical Bot Chats are hidden from the session list by design, so
 *  last_session alone never sees them: a bot you talk to all day through its
 *  Bot Chat reads "6d ago" because its newest VISIBLE session is a week old.
 *  Every activity signal (age label, pulse dot, unread watermark, recency
 *  sort) keys off this helper. Older gateways without the canonical_session
 *  field degrade to last_session unchanged. */
function botActivitySession(bot) {
  const preferred = bot?.canonical_session
  const last = bot?.last_session

  if (!preferred || !last) {
    return preferred || last || null
  }

  return (preferred.last_active || 0) >= (last.last_active || 0) ? preferred : last
}

/** Worker liveness window: kanban/tool workers heartbeat last_activity_at
 *  at least every 60s while running (agent/session_activity.py), so a
 *  worker whose stamp is older than this is finished or stalled. Wider
 *  than ACTIVE_WINDOW_S to bridge one missed heartbeat. */
const WORKER_ACTIVE_WINDOW_S = 150

/** True while this bot's freshest kanban/tool worker looks alive. Workers
 *  never surface in conversation lists, so without this a profile grinding
 *  through a 30-minute kanban task reads idle ("3 hr ago") the whole time
 *  (hermes-agent#90268). Older gateways omit worker_session — always false. */
function workerActiveAt(bot, now = Date.now()) {
  const ts = bot?.worker_session?.last_active || 0
  return Boolean(ts && now / 1000 - ts < WORKER_ACTIVE_WINDOW_S)
}

/** Bots that are working right now: the profile the gateway is running a
 *  turn for (busy), any bot whose last message landed inside the liveness
 *  window, plus any bot with a live kanban/tool worker. Pure — output
 *  follows the input roster's order, so presence never reorders or hides
 *  the normal list. */
function activeBots(roster, activeProfile, gatewayState, now = Date.now()) {
  return (roster || []).filter(bot => {
    const busyTurn = !bot.remoteSource && bot.name === activeProfile && gatewayState === 'busy'
    const last = botActivitySession(bot)?.last_active || 0
    const inWindow = Boolean(last && now / 1000 - last < ACTIVE_WINDOW_S)

    return busyTurn || inWindow || workerActiveAt(bot, now)
  })
}

function rosterActivityMatches(row, filter, now = Date.now()) {
  if (!filter || filter === 'all') {
    return true
  }

  if (filter === 'active') {
    return Boolean(row?.active)
  }

  const activity = Number(row?.activity || 0)
  const recent = Boolean(activity && now - activity <= RECENT_ACTIVITY_WINDOW_S * 1000)

  return filter === 'recent' ? recent : !recent
}

function botRowOwnsWorkspace(
  bot,
  activeGroup,
  botChatFocused,
  botsHomeFronted,
  focusedOwner,
  selectedRosterKey
) {
  if (activeGroup) {
    return false
  }

  if (botsHomeFronted || !botChatFocused) {
    return selectedRosterKey === botRosterKey(bot)
  }

  return isActiveRosterBot(bot, focusedOwner)
}

// ── bot row ──────────────────────────────────────────────────────────────────

function BotRow({ bot, onDelete, onEdit, onGroup, showHandle }) {
  const activeProfile = useValue(host.state.profile)
  const focusedOwner = focusedRosterOwner(useValue($focusedBotOwner))
  const selectedRosterKey = useValue($selectedRosterKey)
  const botChatFocused = useValue($botChatFocused)
  const botsHomeFronted = useValue($botsHomeFronted)
  const activeGroup = useValue($groupChatWorkspace)
  const allMeta = useValue($botMeta)
  const meta = botRosterMeta(bot, allMeta)
  const hidden = isBotHidden(bot, allMeta)
  const pinned = isBotPinned(bot, allMeta)
  const sourceStatus = botSourceStatus(bot)
  const groups = botGroups(meta)
  const last = bot.last_session
  // Highlight follows the chat on screen (focused session's owner), not the
  // gateway socket's home — a focused tab doesn't swap the socket, and on the
  // old keying the wrong bot stayed highlighted while you read another's chat.
  // A selected group chat suppresses every bot-row highlight: the group row
  // owns the selection then (#88979).
  const activeConnectionId = String(host.state.connectionId?.get?.() || 'local').trim()
  // The highlight follows whoever owns the MAIN workspace. While a chat owns
  // it, that chat's profile wins (a stale roster click must not key the
  // highlight to a bot you are not reading). While the Bots home owns it, the
  // source-qualified selection is the owner — and it is the only rule that
  // can highlight a remote row, which has no focusable local chat.
  const isActive = botRowOwnsWorkspace(
    bot,
    activeGroup,
    botChatFocused,
    botsHomeFronted,
    focusedOwner,
    selectedRosterKey
  )
  // Turn-busy is a SOCKET fact: only the gateway-home profile can be mid-turn.
  const isGatewayHome = !bot.remoteSource && bot.name === activeProfile &&
    isActiveRosterBot(bot, { name: activeProfile, connectionId: activeConnectionId })
  const { shape, color, image } = botAppearance(bot.name, meta)
  // Keep user photos/pets. Drop the 160px SVG backfill so the math face can move.
  const photo = Boolean(image && !isBackfilledFacePng(image))
  const gatewayState = useValue(host.state.gateway)
  // Preview identity must match click identity (#88200): when the backend
  // resolved the pinned canonical chat, preview THAT session — not the
  // profile's most recent (but unrelated) activity. Activity signals
  // (age label, pulse dot) follow the same rule via botActivitySession:
  // the canonical Bot Chat is hidden from last_session, so keying age off
  // last_session alone shows "6d ago" on a bot you just messaged.
  const previewSession = bot.canonical_session || last
  const activitySession = botActivitySession(bot)
  // A live kanban/tool worker counts as activity (#90268): fresh age while it
  // runs, falling back to chat activity when it ends.
  const workerActive = workerActiveAt(bot)
  const rowAgeTs = workerActive
    ? Math.max(activitySession?.last_active || 0, bot.worker_session?.last_active || 0)
    : activitySession?.last_active || 0
  const botMood = workerActive || (isGatewayHome && gatewayState === 'busy') ? 'work' : 'idle'
  // Subscribe on every render. A source switch turns the same keyed row from
  // thin to rich; conditionally calling useValue here breaks React hook order.
  const unreadByName = useValue($botUnread)
  const unread = Boolean(unreadByName[botSelectionKey(bot)])
  // Needs-attention badge (#93091 item 3): background failures record under
  // the selection key (group turns) or the route key (relay deliveries) —
  // check both. Local/unannotated rows carry no connectionId, so their relay
  // failures live under `<activeConnectionId>::<name>` — resolve that shape
  // too or active-gateway bots never badge. Hidden bots keep their entry;
  // hiding is display-only.
  const attentionByKey = useValue($botAttention)
  const attention =
    attentionByKey[botSelectionKey(bot)] ||
    attentionByKey[botRosterKey(bot)] ||
    attentionByKey[`${bot?.connectionId || activeConnectionId}::${bot?.name || 'default'}`] ||
    null
  // WHO sent the last message (bot-to-bot DM vs human) — the full stored
  // history lives in the canonical chat, not inline.
  // Preview identity must match click identity (#88200): when the backend
  // resolved the pinned canonical chat, preview THAT session — not the
  // profile's most recent (but unrelated) activity. Liveness checks above
  // keep last_session semantics: any recent activity means the bot is alive.
  const { fromBot } = previewKind(previewSession?.preview)
  // DM previews read like DMs: strip the delivery prefix, keep the message.
  const displayPreview = stripPreviewMarkdown(
    fromBot
      ? (previewSession?.preview || '').replace(A2A_PREFIX_RE, '').trim() || '…'
      : previewSession?.preview || ''
  )
  const handle = botHandle(bot.name, bot)
  const gatewayLabel = bot.connectionLabel || (bot.connectionId === 'local' ? 'This device' : '')
  const showDetailsRow = Boolean(showHandle || displayPreview || fromBot)
  const rowTooltip = [displayName(bot, meta), `@${handle}`, gatewayLabel, sourceStatus.label]
    .filter(Boolean)
    .join(' · ')

  const warm = () => {
    // Multi-source row: pre-dial the agent's OWN source (feature-detected).
    if (bot.sourceScoped && typeof host.warmAgent === 'function') {
      try {
        host.warmAgent(bot.connectionId, bot.name)
      } catch {
        /* warm is best-effort */
      }

      return
    }

    if (typeof host.warmProfile !== 'function') {
      return
    }

    try {
      host.warmProfile(bot.name)
    } catch {
      /* warm is best-effort */
    }
  }

  // Rows and Active Now share the exact-owner open path; only that path may
  // activate a source and resolve the canonical Bot Chat.
  const open = () => void openRosterBot(bot)

  const row = jsxs('button', {
    type: 'button',
    onPointerEnter: warm,
    onClick: open,
    className: cn(
      'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
      'hover:bg-(--chrome-action-hover)',
      isActive && 'bg-(--ui-row-active-background)'
    ),
    'aria-label': rowTooltip,
    children: [
      jsx('div', {
        className: cn('shrink-0', !sourceStatus.available && 'grayscale opacity-60'),
        children: jsx(BotFace, {
          shape,
          color,
          image: photo ? image : null,
          size: 34,
          name: bot.name,
          mood: botMood
        })
      }),
      jsxs('div', {
        className: 'min-w-0 flex-1',
        children: [
          jsxs('div', {
            className: 'flex items-baseline justify-between gap-2',
            children: [
              jsxs('div', {
                className: 'flex min-w-0 items-center gap-1.5',
                children: [
                  pinned
                    ? jsx(Tip, {
                        label: 'Pinned',
                        children: jsx(Codicon, {
                          name: 'pinned',
                          className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)'
                        })
                      })
                    : null,
                  hidden
                    ? jsx(Tip, {
                        label: 'Hidden from the roster',
                        children: jsx(Codicon, {
                          name: 'eye-closed',
                          className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)'
                        })
                      })
                    : null,
                  jsx(Tip, {
                    label: rowTooltip,
                    children: jsx('span', {
                      className: 'min-w-0 truncate text-[0.8125rem] font-medium',
                      children: displayName(bot, meta)
                    })
                  }),
                ]
              }),
              attention
                ? jsx(Tip, {
                    label: BOT_ATTENTION_HINTS[attention.reason] || 'Needs attention',
                    children: jsx(Codicon, {
                      name: 'warning',
                      className: 'shrink-0 text-[0.6875rem] text-(--ui-warning,#f59e0b)',
                      'aria-label': 'needs attention'
                    })
                  })
                : null,
              unread
                ? jsx('span', {
                    className: 'size-2 shrink-0 rounded-full bg-(--ui-accent)',
                    'aria-label': 'unread'
                  })
                : null,
              rowAgeTs
                ? jsx('span', {
                    className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                    children: relativeTime(rowAgeTs * 1000)
                  })
                : null
            ]
          }),
          showDetailsRow
            ? jsxs('div', {
                className: 'flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)',
                children: [
                  showHandle
                    ? jsx('span', {
                        className: 'shrink-0 font-mono text-[0.6875rem] text-(--ui-text-quaternary)',
                        children: `@${handle}`
                      })
                    : null,
                  showHandle && displayPreview
                    ? jsx('span', { className: 'shrink-0 text-(--ui-text-quaternary)', children: '·' })
                    : null,
                  displayPreview
                    ? jsx('span', {
                        className: cn('min-w-0 truncate', fromBot && 'italic'),
                        children: displayPreview
                      })
                    : null
                ]
              })
            : null
        ]
      })
    ]
  })

  return jsxs(ContextMenu, {
    children: [
      jsx(ContextMenuTrigger, { asChild: true, children: row }),
      jsxs(ContextMenuContent, {
        children: [
          jsx(ContextMenuItem, {
            onSelect: () => {
              void ensureBotMetadata(bot).then(current => {
                const pinned = Boolean(current.pinned)
                void saveBotMeta(bot, { pinned: !pinned })
                host.notify({
                  kind: 'info',
                  message: `${displayName(bot, current)} ${pinned ? 'unpinned' : 'pinned to top'}`
                })
              }).catch(error => host.notifyError?.(error, 'Could not load bot metadata'))
            },
            children: pinned ? 'Unpin' : 'Pin to top'
          }),
          jsx(ContextMenuItem, {
            onSelect: () => {
              void ensureBotMetadata(bot).then(current => {
                const hidden = Boolean(current.hidden)
                void saveBotMeta(bot, { hidden: !hidden })

                if (!hidden) {
                  fallbackSelectionAfterHide(botSelectionKey(bot))
                }

                host.notify({
                  kind: 'info',
                  message: hidden
                    ? `${displayName(bot, current)} is back in the roster`
                    : `${displayName(bot, current)} hidden — use the eye button in the Bots header to see hidden bots`
                })
              }).catch(error => host.notifyError?.(error, 'Could not load bot metadata'))
            },
            children: hidden ? 'Unhide' : 'Hide'
          }),
          jsx(ContextMenuSeparator, {}),
          jsx(ContextMenuItem, {
            onSelect: () => void ensureBotMetadata(bot).then(() => onEdit(bot)).catch(error => host.notifyError?.(error, 'Could not load bot')),
            children: 'Edit Profile'
          }),
          jsx(ContextMenuItem, {
            onSelect: () => void ensureBotMetadata(bot).then(() => onGroup(bot)).catch(error => host.notifyError?.(error, 'Could not load bot groups')),
            children: groups.length ? `Groups: ${groups.join(', ')}…` : 'Manage groups…'
          }),
          jsx(ContextMenuItem, {
            onSelect: () => {
              host.notify({ kind: 'info', message: `Duplicating ${displayName(bot, meta)}…` })
              duplicateBot(bot, $lastRoster.get())
                .then(name => {
                  queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
                  host.notify({ kind: 'success', message: `Created ${name} — full copy of ${bot.name}` })
                })
                .catch(err => host.notifyError(err, 'Duplicate failed'))
            },
            children: 'Duplicate'
          }),
          jsx(ContextMenuSeparator, {}),
          jsx(ContextMenuItem, {
            onSelect: () => {
              saveSelectedRosterBot(bot)
              setBotsWorkspaceOwner(botWorkspaceOwnerKey(bot), bot)
              newBotChat(bot)
            },
            children: 'New chat with this agent'
          }),
          isDefaultBot(bot) ? null : jsx(ContextMenuSeparator, {}),
          isDefaultBot(bot)
            ? null
            : jsx(ContextMenuItem, {
                onSelect: () => onDelete(bot),
                variant: 'destructive',
                children: 'Delete'
              })
        ]
      })
    ]
  })
}

// ── model picker (provider/model dropdowns via model.options) ───────────────

function useModelOptions(bot = null) {
  // Hook body runs during render: an orphaned row must paint the picker
  // disabled/erroring, not throw into the pane's error boundary.
  const resolved = bot ? resolveBotConnectionRoute(bot) : null
  const route = resolved?.status === 'resolved' ? resolved.route : null
  const orphaned = resolved?.status === 'owner_removed'

  return useQuery({
    queryKey: [ID, 'model-options', route ? botRouteKey(route) : 'active'],
    queryFn: () => requestForBot(bot, 'model.options', {
      include_unconfigured: true,
      explicit_only: false,
      refresh: true
    }),
    enabled: !orphaned,
    staleTime: 120000,
    retry: false
  })
}

/**
 * Provider + model dropdowns from the gateway's configured inventory — the
 * same data the core model picker shows. `value = {provider, model}`;
 * onChange receives the merged patch.
 */
function ModelPicker({ bot = null, value, onChange, placeholderModel = 'gateway default' }) {
  const { data, isLoading, error } = useModelOptions(bot)

  // Hooks are ALWAYS declared up front, before any conditional return.
  // Declaring them after a return trips React error #310.
  const NONE = '__default__'
  const CUSTOM = '__custom__'
  const providers = (data?.providers || []).filter(p => p && p.slug)
  const isKnown =
    !value.provider || value.provider === NONE || providers.some(p => p.slug === value.provider)
  const [useFreeText, setUseFreeText] = useState(!isKnown)

  if (isLoading) {
    return jsx('div', {
      className: 'flex justify-center py-2',
      children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
    })
  }

  if (error || !providers.length) {
    // Fallback: free text (older gateway or empty inventory).
    return jsxs('div', {
      style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' },
      children: [
        labeled(
          'Provider',
          jsx(Input, {
            placeholder: 'omnirouter / 9router / nous \u2026',
            value: value.provider,
            onChange: event => onChange({ provider: event.target.value })
          })
        ),
        labeled(
          'Model',
          jsx(Input, {
            placeholder: 'antigravity/gemini-3.6-flash-high',
            value: value.model,
            onChange: event => onChange({ model: event.target.value })
          })
        )
      ]
    })
  }

  if (useFreeText) {
    return jsxs('div', {
      style: { display: 'flex', flexDirection: 'column', gap: '8px' },
      children: [
        jsxs('div', {
          style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' },
          children: [
            labeled(
              'Provider (Custom)',
              jsx(Input, {
                placeholder: 'e.g. omnirouter, inferx, 9router',
                value: value.provider,
                onChange: event => onChange({ provider: event.target.value })
              })
            ),
            labeled(
              'Model (Custom)',
              jsx(Input, {
                placeholder: 'e.g. antigravity/gemini-3.6-flash-high',
                value: value.model,
                onChange: event => onChange({ model: event.target.value })
              })
            )
          ]
        }),
        jsx(Button, {
          variant: 'ghost',
          size: 'sm',
          className: 'h-6 self-start text-xs text-(--ui-text-tertiary)',
          onClick: () => setUseFreeText(false),
          children: '← Back to dropdowns'
        })
      ]
    })
  }

  const activeProvider = providers.find(p => p.slug === value.provider) || null
  const models = activeProvider
    ? (activeProvider.models || []).map(m => (typeof m === 'string' ? m : m.id || m.name || ''))
    : []

  return jsxs('div', {
    style: { display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: '10px' },
    children: [
      labeled(
        'Provider',
        jsxs(Select, {
          value: value.provider || NONE,
          onValueChange: v => {
            if (v === NONE) {
              onChange({ provider: '', model: '' })
            } else if (v === CUSTOM) {
              setUseFreeText(true)
            } else {
              const prov = providers.find(p => p.slug === v)
              const provModels = (prov?.models || []).map(m =>
                typeof m === 'string' ? m : m.id || m.name || ''
              )
              const first = provModels[0] || ''
              onChange({
                provider: v,
                model: prov && provModels.includes(value.model) ? value.model : first
              })
            }
          },
          children: [
            jsx(SelectTrigger, { className: 'h-8 rounded-md', children: jsx(SelectValue, {}) }),
            jsxs(SelectContent, {
              children: [
                jsx(SelectItem, { value: NONE, children: 'Inherit (launch profile)' }),
                ...providers.map(p =>
                  jsx(
                    SelectItem,
                    { value: p.slug, children: p.name ? `${p.name} (${p.slug})` : p.slug },
                    p.slug
                  )
                ),
                jsx(SelectItem, { value: CUSTOM, children: '✏️ Enter manually…' })
              ]
            })
          ]
        })
      ),
      labeled(
        'Model',
        activeProvider && models.length > 0
          ? jsxs(Select, {
              value: value.model || (models[0] ?? ''),
              onValueChange: v => onChange({ model: v }),
              children: [
                jsx(SelectTrigger, { className: 'h-8 rounded-md', children: jsx(SelectValue, {}) }),
                jsx(SelectContent, {
                  children: models.map(m => jsx(SelectItem, { value: m, children: m }, m))
                })
              ]
            })
          : jsx(Input, {
              placeholder: placeholderModel || 'e.g. model name',
              value: value.model,
              onChange: event => onChange({ model: event.target.value })
            })
      )
    ]
  })
}

// ── advanced profile config (skills / toolsets / model / SOUL) ──────────────
//
// Shared by Edit Profile and New Bot (edit mode only for skills/toolsets —
// a not-yet-created profile has nothing installed to toggle). Backed by
// profiles.describe / profiles.configure; feature-detects older gateways.

function CheckList({ items, onToggle, columns = 2 }) {
  return jsx('div', {
    style: {
      display: 'grid',
      gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      gap: '2px 12px'
    },
    children: items.map(item =>
      jsxs(
        'label',
        {
          className: 'flex min-w-0 cursor-pointer items-center gap-1.5 py-0.5 text-xs text-(--ui-text-secondary)',
          title: item.description || item.name,
          children: [
            jsx(Checkbox, {
              checked: item.enabled,
              onCheckedChange: value => onToggle(item.name, Boolean(value))
            }),
            jsx('span', { className: 'truncate', children: item.name }),
            item.tool_count
              ? jsx('span', {
                  className: 'shrink-0 text-[0.6rem] text-(--ui-text-quaternary)',
                  children: `${item.tool_count}`
                })
              : null
          ]
        },
        item.name
      )
    )
  })
}

function AdvancedProfileConfig({ bot, state, setState }) {
  const [loaded, setLoaded] = useState(false)
  const [unsupported, setUnsupported] = useState(false)
  const [skillFilter, setSkillFilter] = useState('')
  // Component body = render path: degrade an orphaned row to the bot's own
  // name scope instead of throwing into the dialog's error boundary.
  const botRoute = resolveBotConnectionRoute(bot).route
  const backendProfile = botRoute?.targetProfile || botRoute?.profile || bot.name
  const backendScope = botBackendProfileScope(botRoute, bot.name)

  if (!loaded) {
    setLoaded(true)
    Promise.all([
      requestForBot(bot, 'profiles.describe', { name: bot.name }),
      requestForBot(bot, 'mcp.catalog', { profile: bot.name }).catch(() => null)
    ])
      .then(([res, cat]) => {
        const configured = res.mcp_servers || []
        const have = new Set(configured.map(m => m.name))
        const catalog = ((cat && cat.servers) || []).filter(s => !have.has(s.name))
        setState(prev => ({
          ...prev,
          provider: res.model?.provider || '',
          model: res.model?.default || '',
          soul: res.soul || '',
          skills: res.skills || [],
          toolsets: res.toolsets || [],
          mcp: [
            ...configured.map(m => ({ ...m, enabled: m.enabled !== false })),
            ...catalog.map(s => ({
              name: s.name,
              enabled: false,
              fromCatalog: true,
              installed: s.installed,
              auth: s.auth,
              requires: s.requires || [],
              description: s.description || ''
            }))
          ],
          loaded: true
        }))
      })
      .catch(() => setUnsupported(true))
  }

  if (unsupported) {
    return jsx('div', {
      className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
      children: 'Full configuration needs a newer gateway (restart it after updating Hermes).'
    })
  }

  if (!state.loaded) {
    return jsx('div', {
      className: 'flex justify-center py-4',
      children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
    })
  }

  const visibleSkills = skillFilter.trim()
    ? state.skills.filter(s => s.name.toLowerCase().includes(skillFilter.trim().toLowerCase()))
    : state.skills

  const toggleSkill = (name, enabled) =>
    setState(prev => ({
      ...prev,
      dirtySkills: true,
      skills: prev.skills.map(s => (s.name === name ? { ...s, enabled } : s))
    }))

  const toggleToolset = (name, enabled) =>
    setState(prev => ({
      ...prev,
      dirtyToolsets: true,
      toolsets: prev.toolsets.map(t => (t.name === name ? { ...t, enabled } : t))
    }))

  const toggleMcp = (name, enabled) =>
    setState(prev => ({
      ...prev,
      dirtyMcp: true,
      mcp: (prev.mcp || []).map(m => (m.name === name ? { ...m, enabled } : m))
    }))

  const enabledSkills = state.skills.filter(s => s.enabled).length
  const enabledToolsets = state.toolsets.filter(t => t.enabled).length
  const mcpList = state.mcp || []
  const enabledMcp = mcpList.filter(m => m.enabled).length

  // Newer desktop builds export the WHOLE core Capabilities surface
  // (hermes-agent#87317): Skills (installed list + one-click hub installs +
  // full-skill detail), Tools (per-toolset config), and MCP — pinned to this
  // bot via fixedProfile, tab state kept out of the page router via embedded.
  // Render THAT instead of the checkbox stand-ins; writes go straight to the
  // bot's backend, so the dirty-section staging below only carries
  // model + SOUL on these builds. Older builds keep the full checklist UI.
  if (SkillsView && (!botRoute || skillsViewRoutesConnections)) {
    return jsxs('div', {
      className: 'grid gap-4',
      children: [
        jsx(ModelPicker, {
          bot,
          value: { provider: state.provider, model: state.model },
          onChange: patch => setState(prev => ({ ...prev, dirtyModel: true, ...patch }))
        }),
        labeled(
          'Capabilities (applies immediately — skills, tools, MCP)',
          jsx('div', {
            className: 'overflow-hidden rounded-md border border-(--ui-stroke-secondary)',
            style: { height: 460, minHeight: 300, resize: 'vertical', overflow: 'auto' },
            children: jsx(SkillsView, {
              embedded: true,
              fixedProfile: backendProfile,
              ...(botRoute ? { fixedConnection: botRoute.connectionId } : {})
            })
          })
        ),
        labeled(
          'SOUL.md (persona + agent-messaging protocol)',
          jsx(Textarea, {
            className: 'min-h-28 font-mono text-xs leading-5',
            value: state.soul,
            onChange: event => setState(prev => ({ ...prev, dirtySoul: true, soul: event.target.value }))
          })
        )
      ]
    })
  }

  if (bot?.sourceScoped && botRoute?.mode === 'remote' && !skillsViewRoutesConnections) {
    return jsxs('div', {
      className: 'grid gap-4',
      children: [
        jsx(ModelPicker, {
          bot,
          value: { provider: state.provider, model: state.model },
          onChange: patch => setState(prev => ({ ...prev, dirtyModel: true, ...patch }))
        }),
        jsx('div', {
          className: 'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-text-tertiary)',
          children: 'Remote capabilities require a newer desktop. Model and SOUL changes remain staged until you save.'
        }),
        labeled(
          'SOUL.md (persona + agent-messaging protocol)',
          jsx(Textarea, {
            className: 'min-h-28 font-mono text-xs leading-5',
            value: state.soul,
            onChange: event => setState(prev => ({ ...prev, dirtySoul: true, soul: event.target.value }))
          })
        )
      ]
    })
  }

  return jsxs('div', {
    className: 'grid gap-4',
    children: [
      jsx(ModelPicker, {
        bot,
        value: { provider: state.provider, model: state.model },
        onChange: patch => setState(prev => ({ ...prev, dirtyModel: true, ...patch }))
      }),
      labeled(
        `Skills (${enabledSkills}/${state.skills.length} enabled)`,
        jsxs('div', {
          className: 'grid gap-1.5 rounded-md border border-(--ui-stroke-secondary) p-2',
          children: [
            jsx(Input, {
              className: 'h-7 text-xs',
              placeholder: 'Filter skills…',
              value: skillFilter,
              onChange: event => setSkillFilter(event.target.value)
            }),
            jsx(ScrollArea, {
              className: 'hermes-scroll-cap',
              style: { maxHeight: 180 },
              children: jsx(CheckList, { items: visibleSkills, onToggle: toggleSkill, columns: 2 })
            }),
            jsx(HubSkillsSection, {
              forProfile: backendScope,
              onInstalled: name =>
                setState(prev =>
                  prev.skills.some(s => s.name === name)
                    ? prev
                    : { ...prev, skills: [...prev.skills, { name, enabled: true }] }
                )
            })
          ]
        })
      ),
      labeled(
        `Toolsets (${enabledToolsets}/${state.toolsets.length} enabled — unchecking all restores the default)`,
        jsx('div', {
          className: 'rounded-md border border-(--ui-stroke-secondary) p-2',
          children: jsx(ScrollArea, {
            className: 'hermes-scroll-cap',
            style: { maxHeight: 320 },
            children: jsx('div', {
              className: 'grid gap-1.5',
              children: state.toolsets.map(tset =>
                jsxs(
                  'div',
                  {
                    className: 'rounded-md border border-(--ui-stroke-secondary) p-2',
                    children: [
                      jsxs('label', {
                        className: 'flex items-center gap-2 text-xs font-medium text-(--ui-text-secondary)',
                        children: [
                          jsx(Checkbox, {
                            checked: !!tset.enabled,
                            onCheckedChange: value => toggleToolset(tset.name, Boolean(value))
                          }),
                          jsx('span', { children: tset.name })
                        ]
                      }),
                      // The REAL per-toolset config (env vars / API keys / model
                      // picker / post-setup), scoped to THIS bot's profile, when
                      // the desktop build exposes it. Older builds: just the toggle.
                      ToolsetConfigPanel
                        ? jsx('div', {
                            className: 'mt-1.5 border-t border-(--ui-stroke-secondary) pt-1.5',
                            children: jsx(ToolsetConfigPanel, { toolset: tset.name, profile: backendScope })
                          })
                        : null
                    ]
                  },
                  tset.name
                )
              )
            })
          })
        })
      ),
      labeled(
        'MCP servers',
        jsx('div', {
          className: 'overflow-hidden rounded-md border border-(--ui-stroke-secondary)',
          // The REAL MCP tab core Settings renders — per-server enable + OAuth
          // sign-in + API-key setup + live probes — scoped to this bot's profile.
          // Feature-detected: older desktop builds without the SDK export fall
          // back to the plugin's own checkbox list + inline setup buttons.
          children: McpTab && typeof host.getGateway === 'function'
            ? jsx('div', {
                style: { minHeight: 220, maxHeight: 360 },
                children: jsx(McpTab, { gateway: host.getGateway(), profile: backendScope })
              })
            : mcpList.length === 0
              ? jsx('div', {
                  className: 'px-1 py-2 text-center text-xs text-(--ui-text-tertiary)',
                  children: 'No MCP servers configured or in the catalog.'
                })
              : jsx(ScrollArea, {
                  className: 'hermes-scroll-cap',
                  style: { maxHeight: 180 },
                  children: jsx('div', {
                    className: 'grid gap-1 p-2',
                    children: mcpList.map(m => {
                      const needsSetup = m.fromCatalog && !m.installed && ((m.requires || []).length > 0 || (m.auth || '').toLowerCase() === 'oauth')
                      return jsxs(
                        'label',
                        {
                          className: 'flex items-start gap-2 text-xs text-(--ui-text-secondary)',
                          children: [
                            jsx(Checkbox, {
                              checked: !!m.enabled,
                              disabled: needsSetup,
                              onCheckedChange: value => toggleMcp(m.name, Boolean(value))
                            }),
                            jsxs('span', {
                              className: 'min-w-0',
                              children: [
                                jsx('span', { children: m.name }),
                                m.fromCatalog && !needsSetup
                                  ? jsx('span', {
                                      className: 'ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)',
                                      children: m.installed ? 'catalog · installed' : 'catalog'
                                    })
                                  : null,
                                needsSetup
                                  ? jsx(McpSetupButton, {
                                      profile: backendScope,
                                      entry: m,
                                      onDone: () => toggleMcp(m.name, true)
                                    })
                                  : null,
                                m.description
                                  ? jsx('div', {
                                      className: 'truncate text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                                      children: m.description
                                    })
                                  : null
                              ]
                            })
                          ]
                        },
                        m.name
                      )
                    })
                  })
                })
        })
      ),
      labeled(
        'SOUL.md (persona + agent-messaging protocol)',
        jsx(Textarea, {
          className: 'min-h-28 font-mono text-xs leading-5',
          value: state.soul,
          onChange: event => setState(prev => ({ ...prev, dirtySoul: true, soul: event.target.value }))
        })
      )
    ]
  })
}

// ── skills hub section: the REAL hub page (docs) embedded as a picker ──────
// https://hermes-agent.nousresearch.com/docs/skills?embed=picker hides the
// docs chrome and adds "+ Add to this Agent" per card, posting
// {type: 'hermes-skill-pick', ...} to us (hermes-agent#86243). We validate
// the origin, install via skills.manage, and bubble onInstalled so the
// checklist above gains the row. Search-box fallback kept for offline use.

const HUB_ORIGIN = 'https://hermes-agent.nousresearch.com'
const HUB_PICKER_URL = HUB_ORIGIN + '/docs/skills?embed=picker'

function HubSkillsSection({ forProfile, onInstalled }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [searching, setSearching] = useState(false)
  const [installing, setInstalling] = useState(null)
  const [installed, setInstalled] = useState({})
  const [browseHub, setBrowseHub] = useState(false)
  const installRef = useRef(null)
  const frameRef = useRef(null)

  // Picker messages from the embedded hub page. Origin- AND source-checked —
  // only OUR frame may ask for an install (the hub origin alone would let any
  // other window on it, e.g. an OAuth popup, trigger installs too); installs
  // route through the same install() the search fallback uses.
  useEffect(() => {
    if (!browseHub) {
      return undefined
    }

    const onMessage = event => {
      if (event.origin !== HUB_ORIGIN) {
        return
      }

      if (!frameRef.current || event.source !== frameRef.current.contentWindow) {
        return
      }

      const data = event.data

      if (!data || data.type !== 'hermes-skill-pick' || !data.name) {
        return
      }

      const target = String(data.identifier || data.name)

      // Skill identifiers are slugs / owner-name paths — keep anything
      // else out of skills.manage.
      if (!/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(target)) {
        return
      }

      if (installRef.current) {
        void installRef.current(target, String(data.name))
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [browseHub])

  const search = async () => {
    const q = query.trim()

    if (!q || searching) {
      return
    }

    setSearching(true)
    setResults(null)

    try {
      const res = await host.request('skills.manage', { action: 'search', query: q })
      setResults(res.results || [])
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }

  const install = async (name, displayName) => {
    const label = displayName || name

    if (installing) {
      return
    }

    setInstalling(label)

    try {
      // With forProfile the install lands in that bot's skills dir
      // (gateway skills.manage profile scoping); null = launch profile,
      // which is right at create time — the new bot clones/copies from it.
      await host.request('skills.manage', {
        action: 'install',
        query: name,
        ...(forProfile ? { profile: forProfile } : {})
      })
      setInstalled(prev => ({ ...prev, [label]: true }))
      host.notify({ kind: 'success', message: `Skill "${label}" installed` })

      if (typeof onInstalled === 'function') {
        onInstalled(label)
      }
    } catch (err) {
      host.notifyError(err, `Installing "${label}" failed`)
    } finally {
      setInstalling(null)
    }
  }

  installRef.current = install

  return jsxs('div', {
    className: 'grid gap-1.5 border-t border-(--ui-stroke-secondary) pt-2',
    children: [
      jsxs('div', {
        className: 'flex items-baseline justify-between gap-2',
        children: [
          jsx('div', {
            className: 'text-[0.7rem] font-medium text-(--ui-text-secondary)',
            children: 'Skills Hub'
          }),
          jsx('button', {
            type: 'button',
            className: 'text-[0.65rem] text-(--ui-text-quaternary) hover:text-(--ui-text-secondary)',
            onClick: () => setBrowseHub(v => !v),
            children: browseHub ? 'hide the hub browser' : 'browse the full hub ▾'
          })
        ]
      }),
      browseHub
        ? jsxs('div', {
            className: 'grid gap-1',
            children: [
              // Resizable viewport: native CSS resize handle (bottom-right
              // corner) lets the user drag it larger/smaller. The iframe
              // inside is rendered oversized and scaled DOWN (133% × 0.75)
              // so the hub page starts zoomed out — we can't style the
              // cross-origin page itself, but scaling the frame is ours.
              jsx('div', {
                style: {
                  width: '100%',
                  height: 560,
                  minHeight: 240,
                  minWidth: 320,
                  maxWidth: '100%',
                  resize: 'both',
                  overflow: 'hidden',
                  border: '1px solid var(--ui-stroke-secondary)',
                  borderRadius: 8,
                  position: 'relative'
                },
                children: jsx('iframe', {
                  src: HUB_PICKER_URL,
                  title: 'Hermes Skills Hub',
                  ref: frameRef,
                  style: {
                    width: '133.34%',
                    height: '133.34%',
                    border: 'none',
                    background: 'transparent',
                    transform: 'scale(0.75)',
                    transformOrigin: 'top left'
                  },
                  sandbox: 'allow-scripts allow-same-origin'
                })
              }),
              jsx('div', {
                className: 'px-1 text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                children:
                  installing
                    ? `Installing "${installing}"…`
                    : 'Hit "+ Add to this Agent" on any skill — it installs and appears in the list above. Drag the corner to resize.'
              })
            ]
          })
        : null,
      jsxs('div', {
        className: 'flex gap-1.5',
        children: [
          jsx(Input, {
            className: 'h-7 flex-1 text-xs',
            placeholder: 'Search the hub (community + well-known sources)…',
            value: query,
            onChange: event => setQuery(event.target.value),
            onKeyDown: event => {
              // IME guard: Enter confirming a composed word must not search.
              if (event.nativeEvent?.isComposing || event.keyCode === 229) {
                return
              }
              if (event.key === 'Enter') {
                event.preventDefault()
                void search()
              }
            }
          }),
          jsx(Button, {
            size: 'sm',
            variant: 'secondary',
            disabled: searching || !query.trim(),
            onClick: () => void search(),
            children: searching ? 'Searching…' : 'Search'
          })
        ]
      }),
      searching
        ? jsx('div', {
            className: 'px-1 text-[0.65rem] text-(--ui-text-quaternary)',
            children: 'Searching community + well-known sources — can take ~10s…'
          })
        : null,
      results === null
        ? null
        : results.length === 0
          ? jsx('div', {
              className: 'px-1 py-1.5 text-[0.7rem] text-(--ui-text-quaternary)',
              children: 'No hub skills matched.'
            })
          : jsx(ScrollArea, {
              className: 'hermes-scroll-cap',
              style: { maxHeight: 150 },
              children: jsx('div', {
                className: 'grid gap-1',
                children: results.map(r =>
                  jsxs(
                    'div',
                    {
                      className: 'flex items-center gap-2 text-xs',
                      children: [
                        jsxs('div', {
                          className: 'min-w-0 flex-1',
                          children: [
                            jsx('div', { className: 'truncate font-medium', children: r.name }),
                            r.description
                              ? jsx('div', {
                                  className: 'truncate text-[0.65rem] text-(--ui-text-quaternary)',
                                  children: r.description
                                })
                              : null
                          ]
                        }),
                        installed[r.name]
                          ? jsx('span', {
                              className: 'shrink-0 text-[0.65rem] text-(--ui-text-tertiary)',
                              children: '✓ added'
                            })
                          : jsx(Button, {
                              size: 'sm',
                              variant: 'ghost',
                              className: 'shrink-0 px-2 font-semibold',
                              disabled: installing !== null,
                              title: `Install "${r.name}" and add it to the list above`,
                              onClick: () => void install(r.name),
                              children: installing === r.name ? '…' : '+'
                            })
                      ]
                    },
                    r.name
                  )
                )
              })
            })
    ]
  })
}

function emptyAdvancedState() {
  return {
    loaded: false,
    provider: '',
    model: '',
    soul: '',
    skills: [],
    toolsets: [],
    mcp: [],
    dirtyModel: false,
    dirtySoul: false,
    dirtySkills: false,
    dirtyToolsets: false,
    dirtyMcp: false
  }
}

/** Persist only the dirty sections of the advanced editor. */
async function applyAdvancedConfig(bot, state) {
  const payload = { name: bot.name }
  const applied = {}

  if (state.dirtySoul) {
    payload.soul = ensureMessagingProtocol(state.soul, bot.name, $lastRoster.get())
  }

  if (state.dirtyModel) {
    const model = state.model.trim()
    const provider = state.provider.trim()

    if (model && provider) {
      payload.model = model
      payload.provider = provider
    } else if (!model && !provider) {
      try {
        const result = await requestForBot(bot, 'cli.exec', {
          argv: ['--profile', bot.name, 'config', 'unset', 'model']
        })
        applied.model = result?.blocked !== true && result?.code === 0
      } catch {
        applied.model = false
      }
    } else {
      applied.model = false
    }
  }

  if (state.dirtySkills) {
    payload.disabled_skills = state.skills.filter(s => !s.enabled).map(s => s.name)
  }

  if (state.dirtyToolsets) {
    const all = state.toolsets.length
    const enabled = state.toolsets.filter(t => t.enabled)
    // All enabled (or none) = clear the pin; otherwise pin the checked set.
    payload.enabled_toolsets = enabled.length === all || enabled.length === 0 ? [] : enabled.map(t => t.name)
  }

  if (state.dirtyMcp) {
    payload.enabled_mcp_servers = (state.mcp || []).filter(m => m.enabled).map(m => m.name)
  }

  if (Object.keys(payload).length === 1) {
    return { ok: Object.values(applied).every(Boolean), applied }
  }

  const result = await requestForBot(bot, 'profiles.configure', payload)
  const merged = { ...applied, ...(result?.applied || {}) }

  return { ...result, ok: Object.values(merged).every(Boolean), applied: merged }
}

// ── edit profile dialog ──────────────────────────────────────────────────────

function labeled(label, control) {
  return jsxs('div', {
    className: 'grid gap-1.5',
    children: [
      jsx('label', {
        className: 'text-xs font-medium text-(--ui-text-secondary)',
        children: label
      }),
      control
    ]
  })
}

function EditProfileDialog({ bot, open, onClose }) {
  const metaAll = useValue($botMeta)
  const meta = bot ? botRosterMeta(bot, metaAll) : null
  const appearance = bot ? botAppearance(bot.name, meta) : { shape: 'circle', color: AVATAR_COLORS[3] }
  const [shape, setShape] = useState(appearance.shape)
  const [color, setColor] = useState(appearance.color)
  const [image, setImage] = useState(appearance.image)
  const [title, setTitle] = useState(meta?.title || '')
  const [description, setDescription] = useState(bot?.description || '')
  const [busy, setBusy] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const [adv, setAdv] = useState(emptyAdvancedState())

  // Re-seed local state each time a different bot opens the dialog.
  const [seedKey, setSeedKey] = useState(null)
  const currentKey = bot ? `${botSelectionKey(bot)}:${open}` : null
  if (currentKey !== seedKey) {
    setSeedKey(currentKey)
    if (bot && open) {
      setShape(appearance.shape)
      setColor(appearance.color)
      setImage(appearance.image)
      setTitle(meta?.title || '')
      setDescription(bot.description || '')
      setBusy(false)
      setAdvanced(false)
      setAdv(emptyAdvancedState())
    }
  }

  if (!bot) {
    return null
  }

  const submit = async () => {
    if (busy) {
      return
    }

    setBusy(true)
    let advancedFailed = false
    const persistence = await saveBotMeta(bot, {
      shape,
      color,
      image,
      imageKind: image ? 'photo' : 'shape',
      title: title.trim(),
      custom: true
    })
    // Only an explicit remote failure is an error — 'unsupported' is the
    // documented older-gateway fallback (local wins, silently), and toasting
    // it would flag every save on every legacy setup forever.
    const lookFailed = persistence.serverOutcome === 'failed'

    if (lookFailed) {
      host.notify({ kind: 'error', message: 'Saved look locally; remote persistence failed' })
    }
    if (persistence.serverOutcome === 'persisted') {
      queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
    }

    const desc = description.trim()
    if (desc !== (bot.description || '').trim()) {
      try {
        await requestForBot(bot, 'cli.exec', {
          argv: ['profile', 'describe', bot.name, '--text', desc]
        })
        queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
      } catch (err) {
        host.notifyError(err, 'Saved look locally; description update failed')
      }
    }

    if (adv.loaded && (adv.dirtyModel || adv.dirtySoul || adv.dirtySkills || adv.dirtyToolsets || adv.dirtyMcp)) {
      try {
        const res = await applyAdvancedConfig(bot, adv)
        const failed = Object.entries(res?.applied || {}).filter(([, ok]) => !ok)

        if (failed.length) {
          advancedFailed = true
          host.notify({ kind: 'error', message: `Some sections failed: ${failed.map(([k]) => k).join(', ')}` })
        }
      } catch (err) {
        advancedFailed = true
        host.notifyError(err, 'Advanced configuration failed')
      }
    }

    if (!advancedFailed && !lookFailed) {
      host.notify({ kind: 'success', message: `${displayName(bot, { title })} updated` })
    }
    setBusy(false)
    onClose()
  }

  return jsx(Dialog, {
    open,
    onOpenChange: value => !value && !busy && onClose(),
    children: jsxs(DialogContent, {
      className: advanced ? 'max-w-3xl' : 'max-w-sm',
      // Same resizable-window treatment as the create dialog.
      style: advanced
        ? { resize: 'both', overflow: 'auto', minWidth: 420, minHeight: 360, maxWidth: '95vw', maxHeight: '90vh' }
        : undefined,
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'Edit Profile' }),
            jsx(DialogDescription, { children: `Appearance and role for ${displayName(bot, null)} (${bot.name}).` })
          ]
        }),
        jsxs('div', {
          className: 'grid gap-4',
          children: [
            jsx('div', {
              className: 'flex justify-center py-1',
              children: jsx(BotFace, { shape, color, image, size: 64, name: bot.name })
            }),
            jsx(AvatarPicker, {
              shape,
              color,
              image,
              onShape: setShape,
              onColor: setColor,
              onImage: setImage,
              generateSeed: { name: bot.name, title, description }
            }),
            labeled(
              'Title',
              jsx(Input, {
                placeholder: displayName(bot, null),
                value: title,
                onChange: event => setTitle(event.target.value)
              })
            ),
            labeled(
              'Description',
              jsx(Textarea, {
                className: 'min-h-16',
                placeholder: 'What should this agent help with?',
                value: description,
                onChange: event => setDescription(event.target.value)
              })
            ),
            jsxs('button', {
              type: 'button',
              className:
                'flex items-center gap-1 text-xs font-medium text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)',
              onClick: () => setAdvanced(v => !v),
              children: [
                jsx(Codicon, { name: advanced ? 'chevron-down' : 'chevron-right', className: 'text-[0.8rem]' }),
                'Advanced — model, skills, toolsets, SOUL.md'
              ]
            }),
            advanced
              ? jsx('div', {
                  className: 'rounded-md border border-(--ui-stroke-secondary) p-3',
                  children: jsx(AdvancedProfileConfig, { bot, state: adv, setState: setAdv })
                })
              : null
          ]
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, { variant: 'ghost', disabled: busy, onClick: onClose, children: 'Cancel' }),
            jsx(Button, { disabled: busy, onClick: submit, children: busy ? 'Saving…' : 'Save' })
          ]
        })
      ]
    })
  })
}

// ── create dialog ────────────────────────────────────────────────────────────

function CreateAgentDialog({ open, onClose, roster }) {
  const [name, setName] = useState('')
  // Create mode: the profile is created LAZILY. Capability toggles are staged in
  // component state; the profile is materialized either on Create (submit) or on
  // the first MCP credential setup (ensureAgentCreated), whichever comes first —
  // so OAuth / API-key setup works DURING creation, not only after in Edit.
  const createdRef = useRef(null)
  // In-flight profiles.create shared across concurrent triggers (Create
  // button + MCP setup buttons). Distinct from createdRef on purpose:
  // createdRef must stay a slug string for its sibling consumers.
  const flightRef = useRef(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  // Default shapes mode: deterministic blob face drawn from the agent's name
  // (falls back to the legacy shape vocabulary on older SDKs).
  const [shape, setShape] = useState(blobatarSvg ? 'blobatar' : 'circle')
  const [color, setColor] = useState(AVATAR_COLORS[3])
  const [image, setImage] = useState(null)
  const [advanced, setAdvanced] = useState(false)
  const [cloneFrom, setCloneFrom] = useState('default')
  const [model, setModel] = useState('')
  const [provider, setProvider] = useState('')
  const [soul, setSoul] = useState('')
  const [noSkills, setNoSkills] = useState(false)
  const [shareAuth, setShareAuth] = useState(true)
  const [advTab, setAdvTab] = useState('general')
  // Where the profile is created: '' = the active gateway (unchanged default),
  // else a registry connection id — the profiles.create lands on THAT
  // machine's backend via host.requestProfile, no gateway switch. Only
  // rendered when the desktop has a multi-connection registry.
  const [targetConnection, setTargetConnection] = useState('')
  const [connections, setConnections] = useState(null)

  useEffect(() => {
    if (!open || connections !== null || typeof host.connections !== 'function' || typeof host.requestProfile !== 'function') {
      return
    }

    host
      .connections()
      // host.connections() returns the registry ROWS on current SDKs, but the
      // envelope object ({version, primary, connections: [...]}) on desktops
      // that predate the SDK-side unwrap — accept both shapes.
      .then(value => setConnections(Array.isArray(value) ? value : Array.isArray(value?.connections) ? value.connections : []))
      .catch(() => setConnections([]))
  }, [open, connections])

  const activeConnectionId = String(host.state?.connectionId?.get?.() || '').trim()
  // Remote target = an explicitly picked registry connection that is not the
  // one this window is already on.
  const remoteTarget = Boolean(targetConnection) && targetConnection !== (activeConnectionId || 'local')
  const targetLabel = remoteTarget
    ? (connections || []).find(c => c.id === targetConnection)?.label || targetConnection
    : ''

  /** Gateway RPC on the create target: the picked connection's default
   *  backend for remote targets, the active gateway otherwise. */
  const requestForTarget = (method, params = {}) =>
    remoteTarget
      ? host.requestProfile(
          { connectionId: targetConnection, mode: 'remote', profile: 'default', targetProfile: 'default' },
          method,
          params
        )
      : host.request(method, params)

  // Set once ensureAgentCreated() materializes the profile for the live
  // Capabilities tab (SkillsView needs a real backend to point at). State —
  // not just createdRef — because the render must flip when it lands.
  const [createdForCaps, setCreatedForCaps] = useState(null)
  const [caps, setCaps] = useState(null)
  const [capsFailed, setCapsFailed] = useState(false)
  const [dirtyCaps, setDirtyCaps] = useState({ skills: false, toolsets: false, mcp: false })
  const [capFilter, setCapFilter] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const slug = slugify(name)
  const valid = slug.length > 0 && NAME_RE.test(slug)
  // Once the draft profile is materialized (Capabilities tab / MCP setup) it
  // shows up in the roster — its OWN slug must not read as "taken".
  // A remote-target create is gated by the TARGET machine's roster: a local
  // name clash is fine there, and the remote's own duplicate check rejects
  // real collisions at profiles.create time.
  const taken = remoteTarget
    ? roster.some(b => b.remoteSource && b.connectionId === targetConnection && b.name === slug && b.name !== createdRef.current)
    : roster.some(b => !b.remoteSource && b.name === slug && b.name !== createdRef.current)

  // Draft semantics for the lazily-created profile: opening the Capabilities
  // tab (or running MCP setup) materializes the profile so the LIVE config
  // surfaces have a real backend to write to — but until the user hits
  // Create Agent it is a DRAFT. Cancelling the dialog deletes it, so
  // preconfigure-then-back-out leaves zero residue. Best-effort and
  // fire-and-forget: a failed cleanup surfaces a toast, never blocks close.
  const discardDraft = () => {
    const draft = createdRef.current

    if (!draft) {
      return
    }

    createdRef.current = null
    flightRef.current = null
    const discard = remoteTarget
      ? requestForTarget('cli.exec', { argv: ['profile', 'delete', draft, '--yes'] })
      : deleteBot({ name: draft })
    void Promise.resolve(discard)
      .then(() => host.notify({ kind: 'success', message: `Draft agent "${draft}" discarded` }))
      .catch(err => host.notifyError(err, `Could not clean up draft profile "${draft}"`))
  }

  const reset = () => {
    setName('')
    setTitle('')
    setDescription('')
    setShape(blobatarSvg ? 'blobatar' : 'circle')
    setColor(AVATAR_COLORS[3])
    setImage(null)
    setAdvanced(false)
    // Same default as the initial useState — resetting to '__none__' made
    // the second agent you create silently start from a fresh profile
    // instead of cloning the main one like the first dialog open did.
    setCloneFrom('default')
    setModel('')
    setProvider('')
    setSoul('')
    setNoSkills(false)
    setShareAuth(true)
    setAdvTab('general')
    setCreatedForCaps(null)
    setCaps(null)
    setCapsFailed(false)
    setDirtyCaps({ skills: false, toolsets: false, mcp: false })
    setCapFilter('')
    setTargetConnection('')
    setBusy(false)
    setError(null)
    createdRef.current = null
    flightRef.current = null
  }

  // Capability catalog for the tabs: the profile doesn't exist yet, so show
  // what it WILL have — the clone source's catalog, else the main profile's.
  const capSource = cloneFrom === '__none__' ? 'default' : cloneFrom
  const ensureCaps = () => {
    if ((caps && caps.source === capSource) || capsFailed) {
      return
    }

    Promise.all([
      requestForTarget('profiles.describe', { name: remoteTarget ? 'default' : capSource }),
      requestForTarget('mcp.catalog', {}).catch(() => null)
    ])
      .then(([res, cat]) => {
        // Full MCP menu = the profile's configured servers + the bundled
        // catalog (installable). Configured entries win on name clash.
        const configured = res.mcp_servers || []
        const have = new Set(configured.map(m => m.name))
        const catalog = ((cat && cat.servers) || []).filter(s => !have.has(s.name))

        setCaps({
          source: capSource,
          skills: res.skills || [],
          toolsets: res.toolsets || [],
          mcp: [
            ...configured,
            ...catalog.map(s => ({
              name: s.name,
              enabled: false,
              fromCatalog: true,
              installed: s.installed,
              auth: s.auth,
              requires: s.requires || [],
              description: s.description || ''
            }))
          ]
        })
      })
      .catch(() => setCapsFailed(true))
  }

  const toggleCap = (kind, name, enabled) => {
    setDirtyCaps(prev => ({ ...prev, [kind === 'mcp' ? 'mcp' : kind]: true }))
    setCaps(prev =>
      prev
        ? { ...prev, [kind]: prev[kind].map(x => (x.name === name ? { ...x, enabled } : x)) }
        : prev
    )
  }

  // Materialize the profile exactly once. createdRef stores the finished slug
  // (its consumers — the taken check, draft discard on cancel, the MCP setup
  // button's profile param — all read a string); flightRef shares the
  // in-flight creation promise so simultaneous MCP setup / Create clicks fire
  // ONE profiles.create. A settled flight clears its slot: failures retry,
  // and a null result (form invalid at flight time) isn't sticky.
  const ensureAgentCreated = () => {
    // Renamed since the draft materialized? The old draft is orphaned —
    // discard it and create fresh under the new slug.
    if (createdRef.current && createdRef.current !== slug) {
      discardDraft()
      setCreatedForCaps(null)
    }

    if (createdRef.current) {
      return Promise.resolve(createdRef.current)
    }

    const flight = singleFlight(flightRef, async () => {
      if (!valid || taken) {
        return null
      }

      const descriptionText = [title, description].filter(Boolean).join(' — ')

      await requestForTarget('profiles.create', {
        name: slug,
        description: descriptionText,
        // Clone sources are profiles of the TARGET backend. The picker's
        // roster is the local one, so a remote create always starts from the
        // remote machine's default (or fresh) — never a local profile name
        // the remote box doesn't have.
        clone_from: cloneFrom === '__none__' ? null : remoteTarget ? 'default' : cloneFrom,
        no_skills: noSkills,
        // Shared (not copied) auth keeps ONE OAuth/token pool with the main
        // profile, so refreshes can't invalidate each other. Older gateways
        // ignore the param and copy — still functional, just forked.
        share_auth: shareAuth,
        soul: composeSoul({ name: slug, title, description, roster, customSoul: soul }),
        ...(model.trim() && provider.trim() ? { model: model.trim(), provider: provider.trim() } : {})
      })

      createdRef.current = slug

      // Apply capability picks from the Advanced tabs (best-effort; the
      // profile exists either way and Edit Profile can finish the job).
      try {
        const capPayload = {}

        if (dirtyCaps.skills && caps) {
          capPayload.disabled_skills = caps.skills.filter(s => !s.enabled).map(s => s.name)
        }
        if (dirtyCaps.toolsets && caps) {
          const en = caps.toolsets.filter(t => t.enabled)
          capPayload.enabled_toolsets =
            en.length === caps.toolsets.length || en.length === 0 ? [] : en.map(t => t.name)
        }
        if (dirtyCaps.mcp && caps) {
          capPayload.enabled_mcp_servers = caps.mcp.filter(m => m.enabled).map(m => m.name)
        }
        if (Object.keys(capPayload).length) {
          await requestForTarget('profiles.configure', { name: slug, ...capPayload })
        }
      } catch {
        /* capability application is best-effort */
      }

      if (remoteTarget) {
        // The bot lives on ANOTHER machine — local bot-meta is scoped to the
        // active gateway, so write appearance/title into the remote
        // profile's ui_meta (and asset store) directly. Best-effort: the
        // profile exists either way.
        const { image: avatarImage, ...look } = {
          shape,
          color,
          image,
          imageKind: image ? 'photo' : 'shape',
          title: title.trim(),
          created: Date.now()
        }

        try {
          void requestForTarget('profiles.configure', { name: slug, ui_meta: { 'hermes-bots': look } }).catch(() => undefined)

          if (avatarImage) {
            void requestForTarget('profiles.set_asset', { name: slug, asset: 'avatar', data: avatarImage }).catch(() => undefined)
          }
        } catch {
          /* older remote gateway */
        }
      } else {
        saveBotMeta(slug, { shape, color, image, imageKind: image ? 'photo' : 'shape', title: title.trim(), created: Date.now() })
      }

      queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
      return slug
    })

    return flight
  }

  const submit = async () => {
    if (!valid || taken || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const slugCreated = await ensureAgentCreated()
      if (!slugCreated) {
        setBusy(false)
        setError('Could not create the agent.')
        return
      }

      host.notify({
        kind: 'success',
        message: remoteTarget
          ? `Bot "${displayName({ name: slug, title })}" created on ${targetLabel}`
          : `Bot "${displayName({ name: slug, title })}" created`
      })
      const wasRemote = remoteTarget
      reset()
      onClose()

      if (wasRemote) {
        // The bot lives on another machine: it appears in the roster via the
        // union enumeration; chat routes through its own source. No local
        // canonical chat to birth here.
        queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
        return
      }

      $selectedBot.set(slug)

      // Birth the bot's forever chat right away: it introduces itself as
      // the first thing the user sees, and the pin exists from minute one.
      try {
        // Creates, pins, opens, and kicks off the intro in one flow.
        const sid = await createCanonicalChat(slug)

        if (!sid && typeof host.newChat === 'function') {
          host.newChat(slug)
        }
      } catch {
        if (typeof host.newChat === 'function') {
          host.newChat(slug)
        }
      }
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return jsx(Dialog, {
    open,
    onOpenChange: value => {
      if (!value && !busy) {
        // Cancel path (esc / overlay click): a materialized draft profile is
        // discarded — preconfigure-then-back-out leaves nothing behind.
        discardDraft()
        reset()
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: advanced ? 'max-w-3xl' : 'max-w-md',
      // Native resize handle (bottom-right corner): the dialog becomes a
      // window the user can grow/shrink. overflow:auto is required for CSS
      // resize to engage; caps keep it on screen.
      style: advanced
        ? { resize: 'both', overflow: 'auto', minWidth: 420, minHeight: 360, maxWidth: '95vw', maxHeight: '90vh' }
        : undefined,
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'New Bot' }),
            jsx(DialogDescription, {
              children: 'A named teammate with its own memory, skills, and chat. It can message your other agents.'
            })
          ]
        }),
        jsxs('div', {
          className: 'grid gap-3.5',
          children: [
            jsx('div', {
              className: 'flex justify-center py-1',
              children: jsx(BotFace, { shape, color, image, size: 56, name: slug || 'agent' })
            }),
            jsx(AvatarPicker, {
              shape,
              color,
              image,
              onShape: setShape,
              onColor: setColor,
              onImage: setImage,
              generateSeed: { name: slug || 'agent', title, description }
            }),
            labeled(
              'Name',
              jsx(Input, {
                autoFocus: true,
                placeholder: 'inbox-triage',
                value: name,
                onChange: event => setName(event.target.value)
              })
            ),
            taken
              ? jsx('div', {
                  className: 'text-xs text-(--ui-accent)',
                  children: remoteTarget
                    ? `An agent named "${slug}" already exists on ${targetLabel}.`
                    : `An agent named "${slug}" already exists.`
                })
              : null,
            // Multi-connection desktops choose WHERE the agent lives. Hidden
            // on single-connection setups — the active gateway is the only
            // possible home, exactly the old behavior.
            Array.isArray(connections) && connections.length > 1
              ? labeled(
                  'Create on',
                  jsxs(Select, {
                    value: targetConnection || activeConnectionId || 'local',
                    onValueChange: value => {
                      setTargetConnection(value === (activeConnectionId || 'local') ? '' : value)
                      // The capability catalog and clone list belong to the
                      // target backend — refetch for the new home. The live
                      // Capabilities tab re-pins to it via fixedConnection on
                      // builds that route it (staged checklists otherwise).
                      setCaps(null)
                      setCapsFailed(false)
                      setAdvTab('general')
                    },
                    children: [
                      jsx(SelectTrigger, {
                        className: 'h-8 rounded-md',
                        children: jsx(SelectValue, {})
                      }),
                      jsx(SelectContent, {
                        children: connections.map(connection =>
                          jsx(
                            SelectItem,
                            {
                              value: connection.id,
                              children:
                                connection.id === (activeConnectionId || 'local')
                                  ? `${connection.label || connection.id} (current)`
                                  : connection.label || connection.id
                            },
                            connection.id
                          )
                        )
                      })
                    ]
                  })
                )
              : null,
            remoteTarget
              ? jsx('div', {
                  className: 'text-[0.7rem] leading-5 text-(--ui-text-tertiary)',
                  children: `The agent is created on ${targetLabel} and appears in the roster as a Connections bot. Chat routes to that machine.`
                })
              : null,
            labeled(
              'Title',
              jsx(Input, {
                placeholder: 'Inbox Triage',
                value: title,
                onChange: event => setTitle(event.target.value)
              })
            ),
            labeled(
              'Description',
              jsx(Textarea, {
                className: 'min-h-16',
                placeholder: 'What should this Bot help with?',
                value: description,
                onChange: event => setDescription(event.target.value)
              })
            ),
            jsxs('button', {
              type: 'button',
              className:
                'flex items-center gap-1 text-xs font-medium text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)',
              onClick: () => {
                setAdvanced(v => {
                  if (!v) {
                    ensureCaps()
                  }
                  return !v
                })
              },
              children: [
                jsx(Codicon, { name: advanced ? 'chevron-down' : 'chevron-right', className: 'text-[0.8rem]' }),
                'Advanced'
              ]
            }),
            advanced
              ? jsxs('div', {
                  className: 'grid gap-3 rounded-md border border-(--ui-stroke-secondary) p-3',
                  children: [
                    jsx('div', {
                      className: 'flex gap-1',
                      // Newer desktops export the whole Capabilities surface —
                      // one live tab replaces the three staged checklists.
                      // The live Capabilities surface (SkillsView) binds to
                      // the ACTIVE gateway's backend unless this build routes
                      // fixedConnection (skillsViewRoutesConnections) — then a
                      // remote-target draft gets the live surface pinned to
                      // ITS machine. Builds without that routing keep the
                      // staged checklists for remote targets (their catalog
                      // reads already route to the target).
                      children: (SkillsView && (!remoteTarget || skillsViewRoutesConnections)
                        ? [
                            ['general', 'General'],
                            ['capabilities', 'Capabilities']
                          ]
                        : [
                            ['general', 'General'],
                            ['skills', 'Skills'],
                            ['toolsets', 'Tools'],
                            ['mcp', 'MCP']
                          ]
                      ).map(([id, label]) =>
                        jsx(
                          'button',
                          {
                            type: 'button',
                            className: cn(
                              'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
                              advTab === id
                                ? 'bg-(--chrome-action-hover) text-(--ui-text-primary)'
                                : 'text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)'
                            ),
                            onClick: () => {
                              setAdvTab(id)
                              setCapFilter('')
                              if (id === 'capabilities') {
                                // The live surface needs a real profile —
                                // materialize it now (same lazy-create door
                                // the MCP setup buttons use).
                                void ensureAgentCreated()
                                  .then(created => created && setCreatedForCaps(created))
                                  .catch(err => host.notifyError(err, 'Could not create the profile yet'))
                              } else if (id !== 'general') {
                                ensureCaps()
                              }
                            },
                            children: label
                          },
                          id
                        )
                      )
                    }),
                    advTab === 'general'
                      ? jsxs('div', {
                          className: 'grid gap-3.5',
                          children: [
                            labeled(
                              remoteTarget ? `Clone from profile (on ${targetLabel})` : 'Clone from profile',
                              jsxs(Select, {
                                disabled: remoteTarget,
                                value: remoteTarget ? 'default' : cloneFrom,
                                onValueChange: value => {
                                  setCloneFrom(value)
                                  setCaps(null)
                                  setCapsFailed(false)
                                },
                                children: [
                                  jsx(SelectTrigger, {
                                    className: 'h-8 rounded-md',
                                    children: jsx(SelectValue, {})
                                  }),
                                  jsxs(SelectContent, {
                                    children: [
                                      jsx(SelectItem, {
                                        value: '__none__',
                                        children: 'Fresh profile (bundled skills)'
                                      }),
                                      ...roster.map(b => jsx(SelectItem, { value: b.name, children: b.name }, b.name))
                                    ]
                                  })
                                ]
                              })
                            ),
                            jsx(ModelPicker, {
                              value: { provider, model },
                              onChange: patch => {
                                if ('provider' in patch) {
                                  setProvider(patch.provider)
                                }
                                if ('model' in patch) {
                                  setModel(patch.model)
                                }
                              },
                              placeholderModel: 'inherited from launch profile'
                            }),
                            labeled(
                              'SOUL.md (optional — replaces the generated persona)',
                              jsx(Textarea, {
                                className: 'min-h-24 font-mono text-xs leading-5',
                                placeholder:
                                  'Leave blank to auto-generate from name/title/description + agent-messaging roster.',
                                value: soul,
                                onChange: event => setSoul(event.target.value)
                              })
                            ),
                            jsxs('label', {
                              className: 'flex items-center gap-2 text-xs text-(--ui-text-secondary)',
                              children: [
                                jsx(Checkbox, {
                                  checked: shareAuth,
                                  onCheckedChange: value => setShareAuth(Boolean(value))
                                }),
                                'Share keys & accounts with the main profile'
                              ]
                            }),
                            jsx('div', {
                              className: 'pl-6 pt-0.5 text-[0.7rem] leading-5 text-(--ui-text-tertiary)',
                              children:
                                'Subscriptions, OAuth logins, and API keys stay shared (not copied), so token refreshes never invalidate each other. Uncheck for an isolated snapshot copy.'
                            }),
                            jsxs('label', {
                              className: 'flex items-center gap-2 text-xs text-(--ui-text-secondary)',
                              children: [
                                jsx(Checkbox, {
                                  checked: noSkills,
                                  onCheckedChange: value => setNoSkills(Boolean(value))
                                }),
                                'Create empty (skip bundled skills)'
                              ]
                            })
                          ]
                        })
                      : advTab === 'capabilities'
                        ? !valid || taken
                          ? jsx('div', {
                              className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
                              children: taken
                                ? 'That name is taken — pick another before configuring capabilities.'
                                : 'Name the bot first — a draft profile is created when you open this tab (discarded if you cancel).'
                            })
                          : !createdForCaps
                            ? jsx('div', {
                                className: 'flex justify-center py-4',
                                children: jsx(GlyphSpinner, {
                                  spinner: 'breathe',
                                  className: 'text-(--ui-text-tertiary)'
                                })
                              })
                            : jsx('div', {
                                className: 'overflow-hidden rounded-md border border-(--ui-stroke-secondary)',
                                style: { height: 440, minHeight: 280, resize: 'vertical', overflow: 'auto' },
                                // The REAL core Capabilities surface (skills +
                                // one-click hub installs + tools + MCP), pinned
                                // to the just-created profile — and, for a
                                // remote-target draft, to the target machine's
                                // backend via fixedConnection. Writes land
                                // immediately — no staging needed.
                                children: jsx(SkillsView, {
                                  embedded: true,
                                  fixedProfile: createdForCaps,
                                  ...(remoteTarget ? { fixedConnection: targetConnection } : {})
                                })
                              })
                      : capsFailed
                        ? jsx('div', {
                            className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
                            children:
                              'Capability catalog needs a newer gateway (restart it after updating Hermes).'
                          })
                        : !caps
                          ? jsx('div', {
                              className: 'flex justify-center py-4',
                              children: jsx(GlyphSpinner, {
                                spinner: 'breathe',
                                className: 'text-(--ui-text-tertiary)'
                              })
                            })
                          : advTab === 'skills'
                            ? noSkills
                              ? jsx('div', {
                                  className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
                                  children: '“Create empty” is checked — no bundled skills will be installed.'
                                })
                              : jsxs('div', {
                                  className: 'grid gap-1.5',
                                  children: [
                                    jsx(Input, {
                                      className: 'h-7 text-xs',
                                      placeholder: 'Filter skills…',
                                      value: capFilter,
                                      onChange: event => setCapFilter(event.target.value)
                                    }),
                                    jsx(ScrollArea, {
                                      className: 'hermes-scroll-cap',
                                      style: { maxHeight: 200 },
                                      children: jsx(CheckList, {
                                        items: capFilter.trim()
                                          ? caps.skills.filter(s =>
                                              s.name.toLowerCase().includes(capFilter.trim().toLowerCase())
                                            )
                                          : caps.skills,
                                        onToggle: (name, enabled) => toggleCap('skills', name, enabled),
                                        columns: 2
                                      })
                                    }),
                                    jsx('div', {
                                      className: 'text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                                      children: `Catalog from ${caps.source} — unchecked skills are disabled after creation.`
                                    }),
                                    jsx(HubSkillsSection, {
                                      forProfile: null,
                                      onInstalled: name =>
                                        setCaps(prev =>
                                          !prev || prev.skills.some(s => s.name === name)
                                            ? prev
                                            : { ...prev, skills: [...prev.skills, { name, enabled: true }] }
                                        )
                                    })
                                  ]
                                })
                            : advTab === 'toolsets'
                              ? jsxs('div', {
                                  className: 'grid gap-1.5',
                                  children: [
                                    jsx(ScrollArea, {
                                      className: 'hermes-scroll-cap',
                                      style: { maxHeight: 200 },
                                      children: jsx(CheckList, {
                                        items: caps.toolsets,
                                        onToggle: (name, enabled) => toggleCap('toolsets', name, enabled),
                                        columns: 2
                                      })
                                    }),
                                    jsx('div', {
                                      className: 'text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                                      children: 'Leaving all (or none) checked keeps the default toolset behavior.'
                                    })
                                  ]
                                })
                              : caps.mcp.length === 0
                                ? jsx('div', {
                                    className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
                                    children: 'No MCP servers configured or in the catalog.'
                                  })
                                : jsxs('div', {
                                    className: 'grid gap-1.5',
                                    children: [
                                      jsx(ScrollArea, {
                                        className: 'hermes-scroll-cap',
                                        style: { maxHeight: 200 },
                                        children: jsx('div', {
                                          className: 'grid gap-1',
                                          children: caps.mcp.map(m => {
                                            const needsSetup =
                                              m.fromCatalog && !m.installed && ((m.requires || []).length > 0 || (m.auth || '').toLowerCase() === 'oauth')

                                            return jsxs(
                                              'label',
                                              {
                                                className: 'flex items-start gap-2 text-xs text-(--ui-text-secondary)',
                                                children: [
                                                  jsx(Checkbox, {
                                                    checked: !!m.enabled,
                                                    disabled: needsSetup,
                                                    onCheckedChange: value => toggleCap('mcp', m.name, Boolean(value))
                                                  }),
                                                  jsxs('span', {
                                                    className: 'min-w-0',
                                                    children: [
                                                      jsx('span', { children: m.name }),
                                                      m.fromCatalog && !needsSetup
                                                        ? jsx('span', {
                                                            className: 'ml-1.5 text-[0.65rem] text-(--ui-text-quaternary)',
                                                            children: m.installed
                                                              ? 'catalog · installed'
                                                              : 'catalog'
                                                          })
                                                        : null,
                                                      needsSetup
                                                        ? jsx(McpSetupButton, {
                                                            profile: createdRef.current,
                                                            entry: m,
                                                            ensureProfile: ensureAgentCreated,
                                                            onDone: () => {
                                                              // Setup done: mark installed so the row's
                                                              // checkbox un-disables, and enable it.
                                                              setCaps(prev =>
                                                                prev
                                                                  ? {
                                                                      ...prev,
                                                                      mcp: prev.mcp.map(x =>
                                                                        x.name === m.name
                                                                          ? { ...x, installed: true, enabled: true }
                                                                          : x
                                                                      )
                                                                    }
                                                                  : prev
                                                              )
                                                              setDirtyCaps(prev => ({ ...prev, mcp: true }))
                                                            }
                                                          })
                                                        : null,
                                                      m.description
                                                        ? jsx('div', {
                                                            className:
                                                              'truncate text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                                                            children: m.description
                                                          })
                                                        : null
                                                    ]
                                                  })
                                                ]
                                              },
                                              m.name
                                            )
                                          })
                                        })
                                      }),
                                      jsx('div', {
                                        className: 'text-[0.65rem] leading-4 text-(--ui-text-quaternary)',
                                        children:
                                          'Configured servers copy from the main profile; catalog entries are the bundled MCP menu. Entries needing API keys route through setup first (credentials follow the shared keys setting).'
                                      })
                                    ]
                                  })
                  ]
                })
              : null,
            error
              ? jsx('div', {
                  className: 'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-accent)',
                  children: error
                })
              : null
          ]
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, {
              variant: 'ghost',
              disabled: busy,
              onClick: () => {
                discardDraft()
                reset()
                onClose()
              },
              children: 'Cancel'
            }),
            jsx(Button, {
              disabled: busy || !valid || taken,
              onClick: submit,
              children: busy ? 'Creating…' : 'Create Bot'
            })
          ]
        })
      ]
    })
  })
}

// ── routines (cron) ──────────────────────────────────────────────────────────
//
// Jobs are namespaced "[bot:<name>] <routine>". A job running in the active
// bot profile uses the plain instruction; a different profile keeps the
// hermes -p <bot> chat delegation wrapper so the run reaches that bot's
// history. The tile follows the bot you're chatting with (gateway profile).
const BOT_TAG_RE = /^\[bot:([a-z0-9][a-z0-9_-]*)\]\s*/i
const SAFE_ROUTINE_MARKER = '[bot-mode:routine:v2] '
const LEGACY_DELEGATED_ROUTINE_PREFIX = 'You are running the scheduled routine "'

function routineBot(job) {
  const match = BOT_TAG_RE.exec(job?.name || '')
  return match ? match[1].toLowerCase() : null
}

function routineTitle(job) {
  return (job?.name || '').replace(BOT_TAG_RE, '') || 'Untitled cronjob'
}

function isLegacyDelegatedRoutine(job) {
  const preview = typeof job?.prompt_preview === 'string' ? job.prompt_preview : job?.prompt
  return Boolean(routineBot(job) && typeof preview === 'string' && preview.startsWith(LEGACY_DELEGATED_ROUTINE_PREFIX))
}

async function loadRoutines(owner) {
  const bot = typeof owner === 'string' ? { name: owner } : owner
  const profile = String(bot?.name || '').trim()
  // profile scopes cron.manage to that bot's own cron store (core RPC gained an
  // optional `profile` param). Older gateways ignore the unknown param and
  // return the launch-profile store — the [bot:] tag filter in selectRoutineJobs
  // remains the graceful fallback there.
  const scope = profile ? { profile } : {}
  const data = await requestForBot(bot, 'cron.manage', { action: 'list', include_disabled: true, ...scope })
  const jobs = Array.isArray(data?.jobs) ? data.jobs : []
  const activeLegacyJobs = jobs.filter(
    job => isLegacyDelegatedRoutine(job) && job.enabled !== false && job.state !== 'paused'
  )

  // A pause failing must not fail the LIST — the pane would report "could
  // not load cronjobs" over data that loaded fine, and the 20s poll would
  // re-attempt the failing pause inside a failing query forever. Each pause
  // swallows its own error; the overlay only claims jobs the gateway
  // actually paused, and the next poll retries the rest.
  const pauses = await Promise.all(
    activeLegacyJobs.map(job =>
      requestForBot(bot, 'cron.manage', { action: 'pause', name: job.job_id, ...scope })
        .then(() => true)
        .catch(() => false)
    )
  )

  if (!activeLegacyJobs.length) {
    return data
  }

  const pausedIds = new Set(activeLegacyJobs.filter((job, index) => pauses[index]).map(job => job.job_id))
  return {
    ...data,
    jobs: jobs.map(job => (pausedIds.has(job.job_id) ? { ...job, enabled: false, state: 'paused' } : job))
  }
}

function useRoutines(owner) {
  const bot = typeof owner === 'string' ? { name: owner } : owner
  const route = botConnectionRoute(bot)
  const key = route ? botRosterKey(bot) : bot?.name || ''

  return useQuery({
    queryKey: [...ROUTINES_KEY, key],
    queryFn: () => loadRoutines(bot),
    enabled: Boolean(bot?.name),
    refetchInterval: 20000,
    staleTime: 8000
  })
}

function routineCreateTarget(owner, activeBot) {
  return owner || activeBot
}

async function invalidateRoutineOwner(owner) {
  const bot = typeof owner === 'string' ? { name: owner } : owner
  const route = botConnectionRoute(bot)
  const key = route ? botRosterKey(bot) : bot?.name || ''

  await queryClient.invalidateQueries({
    queryKey: [...ROUTINES_KEY, key],
    exact: true
  })
}

/** Pick which cron jobs to show. A failed refresh keeps the last good list. */
function selectRoutineJobs(data, error, lastJobs, bot) {
  const live = Array.isArray(data?.jobs) ? data.jobs : null
  const all = live ?? (error ? lastJobs : [])
  const scopedToBot = normalizedProfileName(data?.scoped) === normalizedProfileName(bot)
  return {
    live,
    all,
    jobs: scopedToBot ? all : all.filter(job => (routineBot(job) || 'default') === bot)
  }
}

/**
 * Why the Routines pane can be empty while the bot's cron store has jobs.
 *
 * On older gateways the pane only shows jobs namespaced `[bot:<name>]` for the
 * active bot (plus untagged legacy jobs on the default bot). When jobs exist in
 * the store but none surface for this bot, the user is left staring at the
 * generic empty state with no hint that cronjobs are present but hidden.
 * Return a short explanation string in that case, or null when the store is
 * genuinely empty (or the active bot's jobs are already shown).
 */
function routineFilterHint(all, jobs) {
  if (jobs.length !== 0 || !Array.isArray(all) || all.length === 0) {
    return null
  }
  return 'Cronjobs exist in this profile but none are tagged for this bot. ' +
    'Name a job "[bot:<name>] …" to show it here, or see them in Cron below.'
}

function normalizedProfileName(profile) {
  return typeof profile === 'string' ? profile.trim().toLowerCase() : ''
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", "'\"'\"'")}'`
}

function routineInputError(title, instruction) {
  if (String(title).includes('\0')) {
    return 'Cronjob name cannot contain NUL (U+0000).'
  }

  if (String(instruction).includes('\0')) {
    return 'Cronjob instruction cannot contain NUL (U+0000).'
  }

  return null
}

function routinePrompt(bot, title, instruction, activeProfile) {
  if (normalizedProfileName(bot) && normalizedProfileName(bot) === normalizedProfileName(activeProfile)) {
    return instruction
  }

  return (
    `${SAFE_ROUTINE_MARKER}You are running the scheduled routine "${title}" for agent '${bot}'. ` +
    `Execute it AS that agent so the run lands in its own history: run this in the terminal and relay the output:\n\n` +
    `hermes -p ${shellQuote(bot)} chat -c ${shellQuote(`Routine: ${title}`)} -q ${shellQuote(`[Scheduled routine] ${instruction}`)}\n\n` +
    `If the command fails, report the error instead.`
  )
}
function scheduleLabel(schedule) {
  const once = /^once in (.+)$/.exec(schedule || '')

  if (once) {
    return `Once (${once[1]})`
  }

  const bare = /^(\d+)([mhd])$/.exec(schedule || '')

  if (bare) {
    return `Once (${bare[1]}${bare[2]})`
  }

  const match = /^every (\d+)m$/.exec(schedule || '')

  if (match) {
    const minutes = Number(match[1])

    if (minutes % 1440 === 0) {
      const d = minutes / 1440
      return d === 1 ? 'Daily' : `Every ${d} days`
    }

    if (minutes % 60 === 0) {
      const h = minutes / 60
      return h === 1 ? 'Hourly' : `Every ${h}h`
    }

    return `Every ${minutes}m`
  }

  return schedule || ''
}

/** Absolute + relative rendering of a cron timestamp, or null when the job
 *  has never carried one (a job that has not run yet has no `last_run_at`). */
function routineTimestamp(value) {
  const ms = value ? new Date(value).getTime() : Number.NaN
  return Number.isFinite(ms) ? `${relativeTime(ms)} · ${new Date(ms).toLocaleString()}` : null
}

/** The facts `cron.manage list` already sends with every job, as label/value
 *  rows. Pure so the detail contract is testable without a renderer, and so
 *  the dialog cannot invent a field the gateway never sent: an absent value
 *  drops its row instead of rendering "undefined". */
function routineDetailRows(job) {
  const paused = job?.enabled === false || job?.state === 'paused'
  const label = scheduleLabel(job?.schedule)
  const raw = String(job?.schedule || '').trim()

  return [
    ['Status', paused ? 'Paused' : 'Active'],
    ['Schedule', label],
    // `scheduleLabel` humanizes "every 1440m" and cron expressions; keep the
    // raw string when it says something the label dropped.
    ['Schedule (raw)', raw && raw !== label ? raw : null],
    ['Repeat', job?.repeat],
    ['Next run', paused ? null : routineTimestamp(job?.next_run_at)],
    ['Last run', routineTimestamp(job?.last_run_at)],
    ['Last result', job?.last_status],
    ['Delivers to', job?.deliver],
    ['Model', job?.model],
    ['Working directory', job?.workdir]
  ]
    .filter(([, value]) => typeof value === 'string' && value.trim())
    .map(([name, value]) => ({ label: name, value: value.trim() }))
}

/** Why a job is not doing what the user expects. The row only ever showed
 *  "paused"; the scheduler's own reason and the last fire/delivery failures
 *  had no surface in Bot Mode at all. */
function routineDetailIssue(job) {
  const reasons = [job?.last_fire_error, job?.last_delivery_error, job?.paused_reason]
  const first = reasons.find(value => typeof value === 'string' && value.trim())

  return first ? first.trim() : null
}

/** Read-only inspector for one cronjob, rendered from the list payload the
 *  pane already holds — no extra RPC, and no second mutation path beside the
 *  row's own switch and delete. */
function RoutineDetailDialog({ job, onClose, open }) {
  const rows = job ? routineDetailRows(job) : []
  const issue = job ? routineDetailIssue(job) : null
  const instruction = String(job?.prompt_preview || '').trim()

  return jsx(Dialog, {
    open: Boolean(open && job),
    onOpenChange: value => {
      if (!value) {
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: 'max-w-md',
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { className: 'truncate', children: routineTitle(job) }),
            jsx(DialogDescription, { children: 'What this cronjob runs, and when it runs next.' })
          ]
        }),
        jsxs('div', {
          className: 'grid gap-3.5',
          children: [
            issue
              ? jsx('div', {
                  className:
                    'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs leading-5 text-(--ui-accent)',
                  children: issue
                })
              : null,
            jsx('div', {
              className: 'grid gap-1.5',
              children: rows.map(row =>
                jsxs('div', {
                  className: 'flex items-baseline justify-between gap-3 text-xs',
                  children: [
                    jsx('span', { className: 'shrink-0 text-(--ui-text-tertiary)', children: row.label }),
                    jsx('span', { className: 'min-w-0 truncate text-right', children: row.value })
                  ]
                }, row.label)
              )
            }),
            instruction
              ? labeled(
                  'Instruction',
                  jsx('div', {
                    className:
                      'max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs leading-5 text-(--ui-text-secondary)',
                    children: instruction
                  })
                )
              : null
          ]
        }),
        jsx(DialogFooter, {
          children: jsx(Button, { variant: 'secondary', onClick: onClose, children: 'Close' })
        })
      ]
    })
  })
}

function RoutineRow({ job, onOpen, owner }) {
  const profile = typeof owner === 'string' ? owner : owner?.name
  const [busy, setBusy] = useState(false)
  // Optimistic overlay: null = trust server state. Set immediately on
  // toggle so the switch responds even before the refetch lands.
  const [pendingActive, setPendingActive] = useState(null)
  const legacyUnsafe = isLegacyDelegatedRoutine(job)
  const serverActive = !legacyUnsafe && job.enabled !== false && job.state !== 'paused'
  const active = pendingActive === null ? serverActive : pendingActive

  if (pendingActive !== null && pendingActive === serverActive) {
    setPendingActive(null) // server caught up
  }

  const act = async action => {
    if (busy) {
      return
    }

    setBusy(true)

    if (action === 'pause' || action === 'resume') {
      setPendingActive(action === 'resume')
    }

    try {
      await requestForBot(owner, 'cron.manage', { action, name: job.job_id, ...(profile ? { profile } : {}) })
      await invalidateRoutineOwner(owner)
    } catch (err) {
      setPendingActive(null)
      host.notifyError(err, 'Cronjob update failed')
    } finally {
      setBusy(false)
    }
  }

  return jsxs('div', {
    className: cn(
      'group grid gap-1.5 rounded-lg border border-(--ui-stroke-secondary) p-2.5 transition-colors',
      'hover:border-(--ui-stroke-primary, var(--ui-stroke-secondary))'
    ),
    children: [
      jsxs('div', {
        className: 'flex items-center gap-2',
        children: [
          // The row's own button, not a click handler on the card: the switch
          // and delete control are siblings, so opening the details can never
          // swallow a toggle (and a nested button would be invalid markup).
          jsxs('button', {
            type: 'button',
            title: 'Cronjob details',
            className: 'flex min-w-0 flex-1 items-center gap-2 text-left transition-colors hover:text-foreground',
            onClick: () => onOpen?.(job),
            children: [
              jsx('span', {
                'aria-hidden': true,
                className: cn('size-1.5 shrink-0 rounded-full', active ? 'bg-emerald-500' : 'bg-(--ui-text-quaternary)')
              }),
              jsx('span', {
                className: cn('min-w-0 flex-1 truncate text-xs font-medium', !active && 'text-(--ui-text-tertiary)'),
                children: routineTitle(job)
              })
            ]
          }),
          jsx(Switch, {
            checked: active,
            disabled: busy || legacyUnsafe,
            onCheckedChange: value => act(value ? 'resume' : 'pause')
          }),
          jsx(Tip, {
            label: 'Delete cronjob',
            children: jsx('button', {
              type: 'button',
              disabled: busy,
              className:
                'flex size-5 items-center justify-center rounded text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover:opacity-100 hover:bg-(--chrome-action-hover) hover:text-foreground',
              onClick: () => act('remove'),
              children: jsx(Codicon, { name: 'trash', className: 'text-[0.75rem]' })
            })
          })
        ]
      }),
      jsxs('div', {
        className: 'flex items-center justify-between gap-2 pl-3.5',
        children: [
          jsxs('span', {
            className:
              'inline-flex items-center gap-1 rounded-full border border-(--ui-stroke-secondary) px-1.5 py-0.5 text-[0.65rem] text-(--ui-text-tertiary)',
            children: [jsx(Codicon, { name: 'calendar', className: 'text-[0.7rem]' }), scheduleLabel(job.schedule)]
          }),
          jsx('span', {
            className: 'truncate text-[0.65rem] text-(--ui-text-quaternary)',
            children: active && job.next_run_at ? `next ${relativeTime(new Date(job.next_run_at).getTime())}` : 'paused'
          })
        ]
      }),
      legacyUnsafe
        ? jsx('div', {
            className:
              'rounded-md border border-(--ui-stroke-secondary) px-2 py-1.5 text-[0.65rem] leading-4 text-(--ui-accent)',
            children: 'Paused for security: delete and recreate this legacy cronjob before running it again.'
          })
        : null
    ]
  })
}

// Structured schedule picker: frequency first, then only the detail that
// frequency needs (time of day, weekday, day of month, interval). Emits a
// Hermes-native schedule string; Advanced exposes it raw.
const FREQUENCIES = [
  { id: 'once', label: 'Once, in\u2026' },
  { id: 'hourly', label: 'Every hour' },
  { id: 'daily', label: 'Every day' },
  { id: 'weekdays', label: 'Weekdays' },
  { id: 'weekly', label: 'Every week' },
  { id: 'monthly', label: 'Every month' },
  { id: 'interval', label: 'Interval' },
  { id: 'advanced', label: 'Advanced\u2026' }
]

const WEEKDAYS = [
  { id: '1', label: 'Monday' },
  { id: '2', label: 'Tuesday' },
  { id: '3', label: 'Wednesday' },
  { id: '4', label: 'Thursday' },
  { id: '5', label: 'Friday' },
  { id: '6', label: 'Saturday' },
  { id: '0', label: 'Sunday' }
]

const TIMES = (() => {
  const out = []
  for (let h = 0; h < 24; h++) {
    for (const m of [0, 30]) {
      const ampm = h < 12 ? 'AM' : 'PM'
      const h12 = h % 12 === 0 ? 12 : h % 12
      out.push({ id: `${h}:${m}`, label: `${h12}:${String(m).padStart(2, '0')} ${ampm}`, h, m })
    }
  }
  return out
})()

/** Compose the Hermes schedule string from picker state. */
function composeSchedule(state) {
  const [h, m] = (state.time || '9:0').split(':').map(Number)

  switch (state.freq) {
    case 'once': {
      const n = Math.max(1, parseInt(state.onceN, 10) || 1)
      return `${n}${state.onceUnit || 'h'}`
    }
    case 'hourly':
      return 'every 1h'
    case 'daily':
      return `${m} ${h} * * *`
    case 'weekdays':
      return `${m} ${h} * * 1-5`
    case 'weekly':
      return `${m} ${h} * * ${state.weekday || '1'}`
    case 'monthly':
      return `${m} ${h} ${state.monthday || '1'} * *`
    case 'interval': {
      const n = Math.max(1, parseInt(state.intervalN, 10) || 1)
      return `every ${n}${state.intervalUnit || 'h'}`
    }
    default:
      return state.raw || ''
  }
}

function scheduleSummary(state) {
  const t = TIMES.find(x => x.id === state.time)
  const tl = t ? t.label : '9:00 AM'

  const unitWord = u => (u === 'm' ? 'minute(s)' : u === 'd' ? 'day(s)' : 'hour(s)')
  const cap =
    state.freq !== 'once' && String(state.repeatN || '').trim()
      ? `, ${Math.max(1, parseInt(state.repeatN, 10) || 1)} time(s) total`
      : ''

  switch (state.freq) {
    case 'once':
      return `Runs once, ${Math.max(1, parseInt(state.onceN, 10) || 1)} ${unitWord(state.onceUnit)} from now`
    case 'hourly':
      return 'Runs at the top of every hour' + cap
    case 'daily':
      return `Runs every day at ${tl}` + cap
    case 'weekdays':
      return `Runs Monday\u2013Friday at ${tl}` + cap
    case 'weekly':
      return `Runs every ${(WEEKDAYS.find(w => w.id === state.weekday) || WEEKDAYS[0]).label} at ${tl}` + cap
    case 'monthly':
      return `Runs on day ${state.monthday || '1'} of each month at ${tl}` + cap
    case 'interval':
      return `Runs every ${Math.max(1, parseInt(state.intervalN, 10) || 1)} ${unitWord(state.intervalUnit)}` + cap
    default:
      return 'Raw schedule \u2014 every Nm/Nh/Nd or 5-field cron'
  }
}

function pickerSelect(value, onChange, options) {
  return jsxs(Select, {
    value,
    onValueChange: onChange,
    children: [
      jsx(SelectTrigger, { className: 'h-8 rounded-md', children: jsx(SelectValue, {}) }),
      jsx(SelectContent, {
        children: options.map(o => jsx(SelectItem, { value: o.id, children: o.label }, o.id))
      })
    ]
  })
}

function SchedulePicker({ state, setState }) {
  const upd = patch => setState(prev => ({ ...prev, ...patch }))
  const needsTime = ['daily', 'weekdays', 'weekly', 'monthly'].includes(state.freq)

  return jsxs('div', {
    className: 'grid gap-2',
    children: [
      jsxs('div', {
        style: { display: 'grid', gridTemplateColumns: needsTime ? '1fr 1fr' : '1fr', gap: '8px' },
        children: [
          pickerSelect(state.freq, v => upd({ freq: v }), FREQUENCIES),
          needsTime ? pickerSelect(state.time, v => upd({ time: v }), TIMES) : null
        ]
      }),
      state.freq === 'once'
        ? jsxs('div', {
            style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' },
            children: [
              jsx(Input, {
                className: 'h-8',
                placeholder: '30',
                value: state.onceN,
                onChange: event => upd({ onceN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4) })
              }),
              pickerSelect(state.onceUnit, v => upd({ onceUnit: v }), [
                { id: 'm', label: 'minutes from now' },
                { id: 'h', label: 'hours from now' },
                { id: 'd', label: 'days from now' }
              ])
            ]
          })
        : null,
      state.freq === 'weekly'
        ? pickerSelect(state.weekday, v => upd({ weekday: v }), WEEKDAYS)
        : null,
      state.freq === 'monthly'
        ? labeled(
            'Day of month',
            jsx(Input, {
              className: 'h-8',
              placeholder: '1',
              value: state.monthday,
              onChange: event => upd({ monthday: event.target.value.replace(/[^0-9]/g, '').slice(0, 2) })
            })
          )
        : null,
      state.freq === 'interval'
        ? jsxs('div', {
            style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' },
            children: [
              jsx(Input, {
                className: 'h-8',
                placeholder: '2',
                value: state.intervalN,
                onChange: event => upd({ intervalN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4) })
              }),
              pickerSelect(state.intervalUnit, v => upd({ intervalUnit: v }), [
                { id: 'm', label: 'minutes' },
                { id: 'h', label: 'hours' },
                { id: 'd', label: 'days' }
              ])
            ]
          })
        : null,
      state.freq === 'advanced'
        ? jsx(Input, {
            className: 'h-8 font-mono text-xs',
            placeholder: 'every 1d \u00b7 every 2h \u00b7 0 9 * * * (cron)',
            value: state.raw,
            onChange: event => upd({ raw: event.target.value })
          })
        : null,
      state.freq !== 'once' && state.freq !== 'advanced'
        ? jsxs('div', {
            className: 'flex items-center gap-2',
            children: [
              jsx('span', { className: 'text-xs text-(--ui-text-tertiary)', children: 'Stop after' }),
              jsx(Input, {
                className: 'h-7 w-16 text-xs',
                placeholder: '\u221e',
                value: state.repeatN,
                onChange: event => upd({ repeatN: event.target.value.replace(/[^0-9]/g, '').slice(0, 4) })
              }),
              jsx('span', { className: 'text-xs text-(--ui-text-tertiary)', children: 'runs (blank = forever)' })
            ]
          })
        : null,
      jsx('div', {
        className: 'text-[0.65rem] text-(--ui-text-quaternary)',
        children: `${scheduleSummary(state)} \u00b7 ${composeSchedule(state) || '\u2014'}`
      })
    ]
  })
}

function defaultScheduleState() {
  return { freq: 'daily', time: '9:0', weekday: '1', monthday: '1', intervalN: '2', intervalUnit: 'h', onceN: '30', onceUnit: 'm', repeatN: '', raw: '' }
}

function CreateRoutineDialog({ bot, open, onClose }) {
  const [name, setName] = useState('')
  const [instruction, setInstruction] = useState('')
  const [sched, setSched] = useState(defaultScheduleState())
  const [continuity, setContinuity] = useState(false)
  // Where the run's output lands: 'history' = the run session only (Run
  // history / cron page, today's behavior); 'bot-chat' = inject into this
  // bot's canonical Bot Chat as a real message — the bot reads it, acts on
  // it, and responds there (costs the bot one agent turn per run).
  const [target, setTarget] = useState('history')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const activeProfile = useValue(host.state.profile)
  const profile = typeof bot === 'string' ? bot : bot?.name
  const schedule = composeSchedule(sched)

  const reset = () => {
    setName('')
    setInstruction('')
    setSched(defaultScheduleState())
    setContinuity(false)
    setTarget('history')
    setBusy(false)
    setError(null)
  }

  const submit = async () => {
    const title = name.trim()
    const task = instruction.trim()
    const inputError = routineInputError(title, task)

    if (inputError) {
      setError(inputError)
      return
    }

    if (!title || !task || !schedule.trim() || busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const repeatN =
        sched.freq !== 'once' && sched.freq !== 'advanced' && String(sched.repeatN || '').trim()
          ? Math.max(1, parseInt(sched.repeatN, 10) || 1)
          : null
      await requestForBot(bot, 'cron.manage', {
        action: 'add',
        name: `[bot:${profile}] ${title}`,
        schedule: schedule.trim(),
        prompt: routinePrompt(profile, title, task, activeProfile),
        ...(profile ? { profile } : {}),
        ...(repeatN ? { repeat: repeatN } : {}),
        ...(continuity ? { continuity: true } : {}),
        // 'bot-chat' (bare, no name): the job is created IN the bot's own
        // cron store (profile scoping above), so the scheduler resolves the
        // token to that profile — no cross-gateway name ambiguity possible.
        ...(target === 'bot-chat' ? { deliver: 'bot-chat' } : {})
      })
      await invalidateRoutineOwner(bot)
      host.notify({ kind: 'success', message: `Cronjob "${title}" scheduled` })
      reset()
      onClose()
    } catch (err) {
      setBusy(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return jsx(Dialog, {
    open,
    onOpenChange: value => {
      if (!value && !busy) {
        reset()
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: 'max-w-md',
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'New Cronjob' }),
            jsx(DialogDescription, {
              children: `A recurring task ${displayName(typeof bot === 'string' ? { name: bot } : bot, botRosterMeta(bot, $botMeta.get()))} runs on a schedule. Runs land in its own chat history.`
            })
          ]
        }),
        jsxs('div', {
          className: 'grid gap-3.5',
          children: [
            labeled(
              'Name',
              jsx(Input, {
                autoFocus: true,
                placeholder: 'Name this cronjob',
                value: name,
                onChange: event => setName(event.target.value)
              })
            ),
            labeled(
              'Instruction',
              jsx(Textarea, {
                className: 'min-h-20',
                placeholder: 'What should this cronjob do each time it runs?',
                value: instruction,
                onChange: event => setInstruction(event.target.value)
              })
            ),
            labeled('When to run', jsx(SchedulePicker, { state: sched, setState: setSched })),
            labeled(
              'Send results to',
              pickerSelect(target, setTarget, [
                { id: 'history', label: 'Run history only' },
                { id: 'bot-chat', label: `${displayName({ name: bot }, $botMeta.get()[bot])}\u2019s chat (bot responds)` }
              ])
            ),
            jsxs('label', {
              className: 'flex items-center gap-2 text-xs text-(--ui-text-tertiary) cursor-pointer select-none',
              children: [
                jsx('input', {
                  type: 'checkbox',
                  className: 'accent-(--ui-accent)',
                  checked: continuity,
                  onChange: event => setContinuity(event.target.checked)
                }),
                'Continuity: each run sees the previous run\u2019s output (dedupe, continue where it left off)'
              ]
            }),
            error
              ? jsx('div', {
                  className: 'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-xs text-(--ui-accent)',
                  children: error
                })
              : null
          ]
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, {
              variant: 'ghost',
              disabled: busy,
              onClick: () => {
                reset()
                onClose()
              },
              children: 'Cancel'
            }),
            jsx(Button, {
              disabled: busy || !name.trim() || !instruction.trim() || !schedule.trim(),
              onClick: submit,
              children: busy ? 'Scheduling…' : 'Create Cronjob'
            })
          ]
        })
      ]
    })
  })
}

/** Keeps $selectedBot in sync with the focused chat's owner profile.
 *  nanostores' `.listen()` never replays the current value the way
 *  `.subscribe()` does, so a disable → profile switch → re-enable sequence
 *  would otherwise leave $selectedBot pointed at whichever bot was active
 *  before the plugin was disabled — reseeding here on every register() call
 *  closes that gap. Connection qualification keeps same-named owners isolated.
 *  Returns the unbind function for ctx.onDispose. */
function bindProfileSync(ownerStore) {
  const sync = owner => {
    const profile = typeof owner === 'string' ? owner : owner?.profile

    if (!profile || typeof profile !== 'string') {
      return
    }

    const connectionId = String(
      typeof owner === 'object' ? owner?.connectionId || '' : ''
    ).trim()
    $selectedBot.set(connectionId ? `${connectionId}::${profile}` : profile)
  }

  sync(ownerStore.get?.())

  return ownerStore.listen(sync)
}

function resolveRoutineOwner(roster, focusedOwner, selected) {
  // A null focused owner is NOT a failure: the SDK fails closed to null
  // whenever the focused session has no unique bot owner (a normal chat,
  // ambiguous owner hints) — the common case while the user browses the
  // Bots pane. Fall through to the roster-clicked bot (the previously
  // working scope) instead of dead-ending the pane on the unavailable
  // placeholder for every agent (#94516).
  const selectedBot = roster.find(bot => botSelectionKey(bot) === selected)
  const focusedBot = focusedOwner
    ? roster.find(bot => isActiveRosterBot(bot, focusedOwner))
    : null

  if (focusedOwner?.authoritative) {
    // An authoritative focused owner wins, but only through its exact roster
    // row. If that row is absent, fail closed instead of routing cron
    // reads/mutations through a stale selection or an unscoped profile name.
    return focusedBot || null
  }

  return focusedBot || selectedBot || (focusedOwner ? { name: focusedOwner.name } : null)
}

function RoutinesPane() {
  const selected = useValue($selectedBot)
  const focusedOwner = focusedRosterOwner(useValue($focusedBotOwner))
  // A complete focused owner is authoritative. If its exact roster row is
  // absent, fail closed instead of routing cron reads/mutations through a
  // stale selection or an unscoped profile name.
  const owner = resolveRoutineOwner($lastRoster.get(), focusedOwner, selected)
  const bot = String(owner?.name || focusedOwner?.name || 'default').trim() || 'default'
  const allMeta = useValue($botMeta)
  const meta = owner ? botRosterMeta(owner, allMeta) : null
  const { shape, color, image } = botAppearance(bot, meta)
  const { data, error, isLoading, refetch } = useRoutines(owner)
  const [createOpen, setCreateOpen] = useState(false)
  const [createOwner, setCreateOwner] = useState(null)
  // Hold the id, not the record: the 20s poll replaces every job object, and
  // an open inspector must follow the live row (next run, pause, last error)
  // instead of freezing the snapshot that was on screen when it opened.
  const [detailJobId, setDetailJobId] = useState(null)
  const createTarget = owner ? routineCreateTarget(createOwner, bot) : null

  const openCreate = () => {
    if (!owner) {
      return
    }

    setCreateOwner(owner)
    setCreateOpen(true)
  }

  if (!owner) {
    return jsx('div', {
      className: 'flex h-full items-center justify-center px-4 text-center text-xs text-(--ui-text-tertiary)',
      children: 'Cronjobs are unavailable until this agent appears in the roster.'
    })
  }

  const view = selectRoutineJobs(data, error, $lastJobs.get(), bot)
  if (view.live) {
    $lastJobs.set(view.live)
  }
  const jobs = view.jobs
  const detailJob = detailJobId ? jobs.find(job => job.job_id === detailJobId) || null : null
  const staleNotice = error && !view.live && view.all.length
    ? 'Could not refresh cronjobs. Showing the last list we had.'
    : null
  const filterHint = routineFilterHint(view.all, jobs)

  return jsxs('div', {
    className: 'flex h-full flex-col',
    children: [
      jsxs('div', {
        className: 'flex items-center gap-2 px-3 pt-3 pb-2',
        children: [
          jsx(BotFace, { shape, color, image, size: 22, name: bot }),
          jsxs('div', {
            className: 'min-w-0 flex-1',
            children: [
              jsxs('div', {
                className: 'flex min-w-0 items-baseline gap-1.5 truncate',
                children: [
                  jsx('div', {
                    className: 'truncate text-xs font-semibold',
                    children: displayName({ name: bot }, meta)
                  }),
                  showsHandle(bot, meta)
                    ? jsx('span', {
                        className: 'shrink-0 font-mono text-[0.65rem] text-(--ui-text-quaternary)',
                        children: `@${botHandle(bot)}`
                      })
                    : null
                ]
              }),
              jsx('div', {
                className: 'text-[0.65rem] uppercase tracking-wider text-(--ui-text-quaternary)',
                children: 'Cronjobs'
              })
            ]
          }),
          jsx(Tip, {
            label: 'New Cronjob',
            children: jsx('button', {
              type: 'button',
              className:
                'flex size-6 shrink-0 items-center justify-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
              onClick: openCreate,
              children: jsx(Codicon, { name: 'add' })
            })
          })
        ]
      }),
      jsx('div', { className: 'mx-3 border-t border-(--ui-stroke-secondary)' }),
      staleNotice
        ? jsx('div', {
            className: 'mx-3 mt-2 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
            children: staleNotice
          })
        : null,
      isLoading && !view.all.length
        ? jsx('div', {
            className: 'flex flex-1 items-center justify-center',
            children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
          })
        : error && !view.all.length
          ? jsxs('div', {
              className: 'flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center',
              children: [
                jsx(Codicon, { name: 'warning', className: 'text-[1.6rem] text-(--ui-text-quaternary)' }),
                jsx('div', {
                  className: 'text-xs leading-5 text-(--ui-text-tertiary)',
                  children: 'Could not load cronjobs. The list may still be there.'
                }),
                jsx(Button, {
                  variant: 'secondary',
                  size: 'sm',
                  onClick: () => void refetch(),
                  children: 'Retry'
                })
              ]
            })
        : jobs.length === 0
          ? jsxs('div', {
              className: 'flex flex-1 flex-col items-center justify-center gap-3 px-4 text-center',
              children: [
                // No generic placeholder here: an icon + "cronjobs are…" blurb and the
                // create button both just said "empty" (Teknium, Aug 2026). The hint
                // text stays only when jobs exist but are hidden by the bot filter —
                // that carries real information, not an empty-state marker.
                filterHint
                  ? jsx('div', {
                      className: 'text-xs leading-5 text-(--ui-text-tertiary)',
                      children: filterHint
                    })
                  : null,
                jsx(Button, {
                  variant: 'secondary',
                  size: 'sm',
                  onClick: openCreate,
                  children: filterHint ? 'Create a cronjob for this bot' : 'Create Cronjob'
                })
              ]
            })
          : jsx(ScrollArea, {
              className: 'min-h-0 flex-1',
              children: jsx('div', {
                className: 'grid gap-1.5 px-2.5 py-2',
                children: jobs.map(job =>
                  jsx(RoutineRow, { job, onOpen: opened => setDetailJobId(opened.job_id), owner }, job.job_id)
                )
              })
            }),
      jsx(RoutineDetailDialog, {
        job: detailJob,
        open: Boolean(detailJob),
        onClose: () => setDetailJobId(null)
      }),
      jsx(CreateRoutineDialog, {
        bot: createTarget,
        open: createOpen,
        onClose: () => {
          setCreateOpen(false)
          setCreateOwner(null)
        }
        // key is the jsx() THIRD argument — as a prop it is silently ignored
        // and the dialog kept stale per-bot form state when the target changed.
      }, createTarget)
    ]
  })
}

// ── roster pane ──────────────────────────────────────────────────────────────

/** "Active now" presence strip above the roster: chips for every bot that is
 *  working right now (the gateway-busy selected profile + bots whose last
 *  message landed inside the liveness window). Reuses the row avatar; each
 *  chip opens that bot's canonical Bot Chat. Omitted entirely when nothing
 *  is active, and never reorders the roster below it. */
function ActiveNowStrip({ roster, activeProfile, gatewayState, metaByName, onOpen }) {
  const active = activeBots(roster, activeProfile, gatewayState)

  if (!active.length) {
    return null
  }

  return jsxs('div', {
    role: 'status',
    'aria-live': 'polite',
    'aria-label': 'Active now',
    className: 'flex flex-wrap items-center gap-1.5 px-2.5 pb-1.5',
    children: [
      jsx('span', {
        className: 'text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)',
        children: 'Active now'
      }),
      ...active.map(bot => {
        const meta = botRosterMeta(bot, metaByName)
        const { shape, color, image } = botAppearance(bot.name, meta)
        const photo = Boolean(image && !isBackfilledFacePng(image))
        const label = displayName(bot, meta)

        return jsx(
          Tip,
          {
            label: `Open ${label}'s chat`,
            children: jsx('button', {
              type: 'button',
              'aria-label': `Open ${label}'s chat`,
              className: cn(
                'flex items-center gap-1.5 rounded-md bg-(--chrome-action-hover) px-1.5 py-1 text-left transition-colors',
                'hover:bg-(--chrome-action-hover) hover:text-foreground'
              ),
              onClick: () => onOpen(bot),
              children: [
                jsx(BotFace, {
                  shape,
                  color,
                  image: photo ? image : null,
                  size: 24,
                  name: bot.name,
                  mood: 'work'
                }),
                jsx('span', {
                  className: 'max-w-28 truncate text-xs font-medium',
                  children: label
                })
              ]
            })
          },
          botRosterKey(bot)
        )
      })
    ]
  })
}

/** Assign a bot to a group-chat membership without replacing its others.
 *  Existing groups are independent toggles; the input creates and joins a new
 *  one. Canonical groups + the legacy scalar projection ride ui_meta. */
function GroupDialog({ bot, onClose }) {
  const meta = useValue($botMeta)
  const [name, setName] = useState('')
  const current = botGroups(botRosterMeta(bot, meta))
  const groups = knownGroups(meta)

  const setMembership = (group, enabled) => {
    void saveBotMeta(bot, groupMembershipPatch(botRosterMeta(bot, meta), group, enabled))
    host.notify({
      kind: 'info',
      message: enabled
        ? `${displayName(bot, botRosterMeta(bot, meta))} added to “${group}”`
        : `${displayName(bot, botRosterMeta(bot, meta))} removed from “${group}”`
    })
  }

  return jsx(Dialog, {
    open: Boolean(bot),
    onOpenChange: value => {
      if (!value) {
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: 'max-w-sm',
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'Manage groups' }),
            jsx(DialogDescription, {
              children: 'A bot can join multiple group chats. Memberships sync to every machine.'
            })
          ]
        }),
        groups.length
          ? jsx('div', {
              className: 'grid gap-1.5',
              children: groups.map(group => {
                const enabled = current.includes(group)

                return jsxs(
                  'label',
                  {
                    className:
                      'flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-(--chrome-action-hover)',
                    children: [
                      jsx(Checkbox, {
                        checked: enabled,
                        onCheckedChange: checked => setMembership(group, checked === true)
                      }),
                      jsx('span', { children: group })
                    ]
                  },
                  group
                )
              })
            })
          : null,
        jsxs('form', {
          className: 'flex items-center gap-1.5',
          onSubmit: event => {
            event.preventDefault()
            const trimmed = name.trim()

            if (trimmed) {
              setMembership(trimmed, true)
              setName('')
            }
          },
          children: [
            jsx(Input, {
              autoFocus: true,
              placeholder: groups.length ? 'New group…' : 'Group name (e.g. Research)',
              value: name,
              onChange: event => setName(event.target.value)
            }),
            jsx(Button, { type: 'submit', size: 'sm', disabled: !name.trim(), children: 'Create & join' })
          ]
        }),
        current.length
          ? jsx(Button, {
              variant: 'ghost',
              size: 'sm',
              className: 'justify-self-start',
              onClick: () => void saveBotMeta(bot, { groups: [], group: null }),
              children: 'Remove from all groups'
            })
          : null
      ]
    })
  })
}

/** Compact picture controls shared by group-chat creation and settings:
 *  a live preview (image, else the organization glyph), Upload / Generate /
 *  Remove. Reuses the bot-avatar pipeline (device picker, 256px normalize,
 *  image.generate probe) so room pictures cost the same as bot avatars. */
function GroupImageControls({ image, onImage, seedName, seedMembers }) {
  const imagen = useValue($imagenAvailable)
  const [busy, setBusy] = useState(false)

  if (imagen === null) {
    void probeImagen()
  }

  const upload = async () => {
    const raw = await pickImageFromDevice()

    if (raw) {
      onImage(await normalizeAvatarImage(raw))
    }
  }

  const generate = async () => {
    if (busy) {
      return
    }

    setBusy(true)

    try {
      const who = [seedName, seedMembers?.length ? `a team of ${seedMembers.join(', ')}` : '']
        .filter(Boolean)
        .join(' — ')
      const res = await host.request('image.generate', {
        prompt:
          `Group chat icon for an AI agent team called "${who || 'a bot team'}". ` +
          'Friendly minimal emblem, bold flat vector style, solid color background, centered, no text.',
        aspect_ratio: 'square'
      })

      if (!res?.success) {
        throw new Error(res?.error || 'generation failed')
      }

      const img = res.image_data || res.image

      if (img) {
        onImage(await normalizeAvatarImage(img))
      }
    } catch (err) {
      host.notifyError(err, 'Group picture generation failed')
    } finally {
      setBusy(false)
    }
  }

  return jsxs('div', {
    className: 'flex items-center gap-2',
    children: [
      jsx('div', {
        className:
          'flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-full bg-(--chrome-action-hover)',
        children: image
          ? jsx('img', { src: image, alt: '', className: 'size-full object-cover' })
          : jsx(Codicon, { name: 'organization', className: 'text-(--ui-text-tertiary)' })
      }),
      jsx(Button, { type: 'button', variant: 'secondary', size: 'sm', onClick: upload, children: 'Upload' }),
      imagen
        ? jsx(Button, {
            type: 'button',
            variant: 'secondary',
            size: 'sm',
            disabled: busy,
            onClick: generate,
            children: busy ? 'Generating…' : 'Generate'
          })
        : null,
      image
        ? jsx(Button, { type: 'button', variant: 'ghost', size: 'sm', onClick: () => onImage(null), children: 'Remove' })
        : null
    ]
  })
}

/** Edit an existing group chat's name and picture. Renames re-key the room
 *  and every local member's membership (renameGroupChat); the picture rides
 *  the room record. Both apply on Save so a cancelled dialog changes nothing. */
function GroupChatSettingsDialog({ group, members, open, onClose, onRenamed }) {
  const rooms = useValue($groupChats)
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

  return jsx(Dialog, {
    open,
    onOpenChange: value => {
      if (!value) {
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: 'max-w-sm',
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'Group settings' }),
            jsx(DialogDescription, {
              children: 'Rename the group or set a room picture. Members and history are kept.'
            })
          ]
        }),
        jsx(GroupImageControls, {
          image,
          onImage: setImage,
          seedName: name.trim() || group,
          seedMembers: (members || []).map(b => b.name)
        }),
        jsx('form', {
          onSubmit: event => {
            event.preventDefault()
            void save()
          },
          children: jsx(Input, {
            'aria-label': 'Group name',
            autoFocus: true,
            maxLength: 64,
            value: name,
            onChange: event => setName(event.target.value)
          })
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, { variant: 'secondary', onClick: onClose, children: 'Cancel' }),
            jsx(Button, { disabled: !name.trim(), onClick: () => void save(), children: 'Save' })
          ]
        })
      ]
    })
  })
}

/** Discord-style group chat creation: pick 2+ bots via checkboxes (with
 *  search), name the group, create. Assignment appends to each local bot's
 *  group membership list, so the room appears in the roster and syncs
 *  cross-machine via ui_meta without replacing its other groups. */
function CreateGroupChatDialog({ open, roster, onClose, onCreated }) {
  const allMeta = useValue($botMeta)
  const [query, setQuery] = useState('')
  const [checked, setChecked] = useState({})
  const [name, setName] = useState('')
  const [image, setImage] = useState(null)

  // Reset per open so a cancelled draft doesn't leak into the next one.
  useEffect(() => {
    if (open) {
      setQuery('')
      setChecked({})
      setName('')
      setImage(null)
    }
  }, [open])

  // An outage placeholder preserves one selected owner's identity in the
  // sidebar, but it is not a routable room member. Never offer it here.
  const selectableRoster = roster.filter(bot => !bot?.ghost)
  const selected = selectableRoster.filter(bot => checked[botRosterKey(bot)])
  const visible = filterBots(selectableRoster, allMeta, query)
  const atCap = selected.length >= GROUP_CHAT_MAX_MEMBERS
  const placeholder = selected.length
    ? selected.map(bot => displayName(bot, botRosterMeta(bot, allMeta))).join(', ')
    : 'Group name'
  const canCreate = selected.length >= 2 && Boolean(name.trim() || selected.length)

  const create = () => {
    const base = (name.trim() || placeholder).slice(0, 64)

    if (selected.length < 2 || !base) {
      return
    }

    // Creating a group is always a FRESH room. Without this, re-creating a
    // group under an existing name (easy — the default name is just the
    // member names) silently reopens the old room with its full log, which
    // reads as "not a fresh group" (db's Aug 2026 report). Uniquify against
    // both live rooms and any bot's current grouping, then mint a fresh
    // roomId: member sessions are titled by that roomId, so a
    // disbanded-and-recreated group with the SAME display name still gets
    // new sessions instead of resuming the old room's by title.
    const taken = new Set(liveGroupChatNames())

    for (const meta of Object.values($botMeta.get() || {})) {
      for (const existing of botGroups(meta)) {
        taken.add(existing)
      }
    }

    const groupName = uniqueGroupChatName(base, taken)
    const roomId = mintGroupRoomId()

    for (const bot of selected) {
      void saveBotMeta(bot, groupMembershipPatch(botRosterMeta(bot, allMeta), groupName, true))
    }

    // Persist every machine identity, including today's active source. That
    // member becomes remote after a source switch and cannot rely on the new
    // gateway's name-keyed bot metadata to remain seated in this room.
    const roomMembers = durableGroupChatMembers(selected)

    updateGroupChat(groupName, room => {
      room.members = roomMembers
      room.roomId = roomId

      if (image) {
        room.image = image
      }

      return room
    })

    host.notify({ kind: 'info', message: `“${groupName}” created with ${selected.length} bots` })
    onClose()
    onCreated?.(groupName)
  }

  return jsx(Dialog, {
    open,
    onOpenChange: value => {
      if (!value) {
        onClose()
      }
    },
    children: jsxs(DialogContent, {
      className: 'max-w-md',
      children: [
        jsxs(DialogHeader, {
          children: [
            jsx(DialogTitle, { children: 'New Group Chat' }),
            jsx(DialogDescription, {
              children: `Pick 2–${GROUP_CHAT_MAX_MEMBERS} bots. Local memberships sync through each Bot profile; cross-machine members stay scoped to this room.`
            })
          ]
        }),
        jsx(SearchField, {
          'aria-label': 'Search bots to add',
          autoFocus: true,
          containerClassName: 'w-full',
          inputClassName: 'w-full',
          placeholder: 'Search bots to add…',
          value: query,
          onChange: setQuery
        }),
        selected.length
          ? jsx('div', {
              className: 'flex flex-wrap gap-1',
              children: selected.map(bot =>
                jsxs('button', {
                  type: 'button',
                  className:
                    'flex items-center gap-1 rounded-full bg-(--chrome-action-hover) py-0.5 pl-2 pr-1.5 text-[0.6875rem] text-(--ui-text-secondary) transition-colors hover:text-foreground',
                  title: 'Remove from selection',
                  onClick: () => setChecked(prev => ({ ...prev, [botRosterKey(bot)]: false })),
                  children: [displayName(bot, botRosterMeta(bot, allMeta)), jsx(Codicon, { name: 'close', className: 'text-[0.6rem]' })]
                }, botRosterKey(bot))
              )
            })
          : null,
        jsx(ScrollArea, {
          className: 'max-h-64 min-h-0',
          children: jsx('div', {
            className: 'grid gap-0.5 pr-2',
            children: visible.length
              ? visible.map(bot => {
                  const meta = botRosterMeta(bot, allMeta)
                  const { shape, color, image } = botAppearance(bot.name, meta)
                  const isChecked = Boolean(checked[botRosterKey(bot)])
                  const disabled = !isChecked && atCap
                  const currentGroups = botGroups(meta)

                  return jsxs('label', {
                    className: cn(
                      'flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 transition-colors hover:bg-(--chrome-action-hover)',
                      disabled && 'cursor-not-allowed opacity-50'
                    ),
                    children: [
                      jsx(BotFace, {
                        shape,
                        color,
                        image: image && !isBackfilledFacePng(image) ? image : null,
                        size: 24,
                        name: bot.name
                      }),
                      jsxs('div', {
                        className: 'min-w-0 flex-1',
                        children: [
                          jsx('div', { className: 'truncate text-xs text-foreground', children: displayName(bot, meta) }),
                          jsx('div', {
                            className: 'truncate text-[0.625rem] text-(--ui-text-quaternary)',
                            children: [
                              currentGroups.length
                                ? `@${botHandle(bot.name, bot)} · in ${currentGroups.map(group => `“${group}”`).join(', ')}`
                                : `@${botHandle(bot.name, bot)}`,
                              bot.remoteSource && bot.connectionLabel ? ` · ${bot.connectionLabel}` : ''
                            ].join('')
                          })
                        ]
                      }),
                      jsx(Checkbox, {
                        checked: isChecked,
                        disabled,
                        onCheckedChange: value => setChecked(prev => ({ ...prev, [botRosterKey(bot)]: Boolean(value) }))
                      })
                    ]
                  }, botRosterKey(bot))
                })
              : jsx('div', {
                  className: 'px-1.5 py-3 text-center text-xs text-(--ui-text-tertiary)',
                  children: query.trim() ? `No bots match “${query.trim()}”` : 'No bots yet — create one first.'
                })
          })
        }),
        jsxs('div', {
          className: 'grid gap-2',
          children: [
            jsx(GroupImageControls, {
              image,
              onImage: setImage,
              seedName: name.trim() || (selected.length ? placeholder : ''),
              seedMembers: selected.map(bot => displayName(bot, botRosterMeta(bot, allMeta)))
            }),
            jsx('form', {
              onSubmit: event => {
                event.preventDefault()
                create()
              },
              children: jsx(Input, {
                'aria-label': 'Group name',
                maxLength: 64,
                placeholder,
                value: name,
                onChange: event => setName(event.target.value)
              })
            })
          ]
        }),
        jsxs(DialogFooter, {
          children: [
            jsx(Button, { variant: 'secondary', onClick: onClose, children: 'Cancel' }),
            jsx(Button, {
              disabled: !canCreate,
              title: selected.length < 2 ? 'Pick at least 2 bots' : undefined,
              onClick: create,
              children: `Create Group${selected.length ? ` (${selected.length})` : ''}`
            })
          ]
        })
      ]
    })
  })
}

// ── threads: the Slack/Discord shape ─────────────────────────────────────────
// Every room entry belongs to a THREAD. Messaging the room composer starts a
// new thread with the whole group; replying inside a thread continues that
// work. Member turns are scoped to the thread that triggered them — deltas,
// watermarks, and responder resolution all key on the thread id.

function groupThreadOf(entry) {
  return entry?.thread || 'legacy'
}

function mintGroupThreadId() {
  return `t${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

// Pre-thread logs (hydrated from storage) get synthetic thread ids: a user
// entry after a real lull starts one, so multi-turn tasks stay whole instead
// of splitting on every follow-up.
const GROUP_THREAD_GAP_MS = 15 * 60000

function assignLegacyThreads(log) {
  let current = null
  let n = 0

  return (log || []).map((entry, i) => {
    if (entry?.thread) {
      current = null

      return entry
    }

    const prev = log[i - 1]
    const lull = !prev || (entry.at || 0) - (prev.at || 0) > GROUP_THREAD_GAP_MS

    if (!current || (entry.from?.kind === 'user' && lull)) {
      current = `legacy-${n++}`
    }

    return { ...entry, thread: current }
  })
}

/** Merged room view for one group: shared timeline with per-member
 *  attribution, a composer that drives the round-robin, and a working
 *  indicator while member turns run. Renders identically in the MAIN chat
 *  window (host.openWorkspace tile) and in the bots panel (older-desktop
 *  fallback); `onBack` is where the Back button routes — the main tile's
 *  closer, or clearing the in-panel workspace atom. */
/** The active @-token at the caret: text from the nearest '@' (that begins a
 *  word) up to the caret, or null when the caret isn't inside a mention. */
function mentionTokenAt(text, caret) {
  const upto = String(text || '').slice(0, caret)
  const match = /(^|\s)@([a-z0-9._-]*)$/i.exec(upto)

  if (!match) {
    return null
  }

  return { query: match[2].toLowerCase(), start: caret - match[2].length - 1 }
}

/** Mention-aware composer input for group rooms. The core composer's
 *  @-completion area doesn't mount inside workspace tiles (#89049), so this
 *  wraps the plain SDK Input with a member-scoped popover: @everyone/@all
 *  quick picks plus each seated member's handle. Insertion produces exactly
 *  the strings parseGroupChatMentions resolves. Keyboard: Up/Down navigate,
 *  Enter/Tab insert (Enter falls through to submit when the popover is
 *  closed), Escape dismisses. */
function GroupMentionInput({ members, onChange, onSubmitDraft, value, ...inputProps }) {
  const allMeta = useValue($botMeta)
  const inputRef = useRef(null)
  const [token, setToken] = useState(null)
  const [selected, setSelected] = useState(0)

  const options = []

  if (token) {
    for (const pick of ['everyone', 'all']) {
      if (pick.startsWith(token.query)) {
        options.push({ handle: pick, meta: 'Every bot in the room' })
      }
    }

    for (const member of members) {
      const handle = String(member.handle || botHandle(member.name, member) || '').trim()
      const display = displayName(member, botRosterMeta(member, allMeta))
      // Renamed members complete on their friendly tag; parser resolves both.
      const tag = String(botMentionTag(member) || handle).trim()

      if (!tag) {
        continue
      }

      if (
        token.query &&
        !tag.toLowerCase().startsWith(token.query) &&
        !(handle && handle.toLowerCase().startsWith(token.query)) &&
        !display.toLowerCase().startsWith(token.query)
      ) {
        continue
      }

      options.push({
        handle: tag,
        meta: display
      })
    }
  }

  const open = Boolean(token) && options.length > 0
  const active = open ? Math.min(selected, options.length - 1) : 0

  const refreshToken = target => {
    setToken(mentionTokenAt(target.value, target.selectionStart ?? target.value.length))
    setSelected(0)
  }

  const insert = handle => {
    if (!token) {
      return
    }

    const caret = inputRef.current?.selectionStart ?? value.length
    const next = `${value.slice(0, token.start)}@${handle} ${value.slice(caret)}`
    onChange(next)
    setToken(null)

    // Restore focus with the caret after the inserted mention.
    const pos = token.start + handle.length + 2
    requestAnimationFrame(() => {
      const el = inputRef.current

      if (el) {
        el.focus()
        try {
          el.setSelectionRange(pos, pos)
        } catch {
          /* input type without selection support */
        }
      }
    })
  }

  return jsxs('div', {
    className: 'relative min-w-0 flex-1',
    children: [
      open
        ? jsx('div', {
            className:
              'absolute bottom-full left-0 z-50 mb-1 max-h-48 w-64 overflow-y-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-primary,#111) py-1 shadow-lg',
            children: options.map((option, index) =>
              jsxs('button', {
                type: 'button',
                className: cn(
                  'flex w-full items-baseline gap-2 px-2 py-1 text-left text-xs',
                  index === active ? 'bg-(--ui-control-hover-background) text-foreground' : 'text-(--ui-text-secondary)'
                ),
                // preventDefault on mousedown so the input keeps focus.
                onMouseDown: event => {
                  event.preventDefault()
                  insert(option.handle)
                },
                onMouseEnter: () => setSelected(index),
                children: [
                  jsx('span', { className: 'font-medium', children: `@${option.handle}` }),
                  jsx('span', { className: 'truncate text-[0.65rem] text-(--ui-text-quaternary)', children: option.meta })
                ]
              }, option.handle)
            )
          })
        : null,
      jsx(Textarea, {
        ...inputProps,
        ref: inputRef,
        value,
        rows: 1,
        // Multi-line room prompts (#89884): the composer was a single-line
        // Input whose form submitted on every Enter — newlines were
        // impossible. Enter (no Shift) still submits via onSubmitDraft;
        // Shift+Enter falls through to the textarea's native newline.
        className: cn('max-h-40 min-h-9 resize-none', inputProps.className),
        onChange: event => {
          onChange(event.target.value)
          refreshToken(event.target)
        },
        onClick: event => refreshToken(event.target),
        onKeyDown: event => {
          // IME composition guard (same as the core composer): Enter here
          // confirms the composed Chinese/Japanese/Korean text — it must not
          // insert a mention nor submit the draft. nativeEvent.isComposing
          // covers Chromium; keyCode 229 covers macOS Chinese IMEs that fire
          // Enter after compositionend with isComposing already false.
          if (event.nativeEvent?.isComposing || event.keyCode === 229) {
            return
          }
          if (open) {
            if (event.key === 'ArrowDown') {
              event.preventDefault()
              setSelected((active + 1) % options.length)

              return
            }

            if (event.key === 'ArrowUp') {
              event.preventDefault()
              setSelected((active - 1 + options.length) % options.length)

              return
            }

            if (event.key === 'Enter' || event.key === 'Tab') {
              event.preventDefault()
              insert(options[active].handle)

              return
            }

            if (event.key === 'Escape') {
              event.preventDefault()
              setToken(null)

              return
            }
          }

          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            onSubmitDraft?.()
          }
        },
        onBlur: () => setToken(null)
      })
    ]
  })
}

/** One member's pending prompt, rendered in the room (#90694).
 *  - clarify: choices as tap buttons (multi-select stages; single-select
 *    stages one), free text always available, batch sub-questions each get
 *    their own input.
 *  - approval: the command in a code row plus the server's choice set
 *    (once/session/always/deny) as buttons — no free text; approvals are a
 *    closed choice. Answer sends via the member's own source. */
function GroupClarifyCard({ entry, members }) {
  const { group } = entry
  const isApproval = entry.kind === 'approval'
  const member = members.find(m => groupMemberKey(m) === entry.memberKey) || members.find(m => m.name === entry.member)
  const [drafts, setDrafts] = useState({})
  const [picked, setPicked] = useState({})
  const [sending, setSending] = useState(false)
  const questions = entry.questions && entry.questions.length
    ? entry.questions.map((q, i) => ({
        qid: q?.qid ?? q?.id ?? `q${i}`,
        question: typeof q?.question === 'string' ? q.question : '',
        choices: Array.isArray(q?.choices) ? q.choices.filter(c => typeof c === 'string' && c) : [],
        multiSelect: Boolean(q?.multi_select ?? q?.multiSelect)
      }))
    : [{ qid: '__single__', question: entry.question, choices: entry.choices, multiSelect: entry.multiSelect }]

  const answerFor = q => {
    const chosen = picked[q.qid] || []

    if (chosen.length) {
      return q.multiSelect ? JSON.stringify(chosen) : chosen[0]
    }

    return isApproval ? '' : (drafts[q.qid] || '').trim()
  }

  const allAnswered = questions.every(q => answerFor(q))

  const submit = async () => {
    if (!member || sending || !allAnswered) {
      return
    }

    setSending(true)

    try {
      if (isApproval || !(entry.questions && entry.questions.length)) {
        await answerGroupClarify(entry, member, answerFor(questions[0]))
      } else {
        const answers = {}

        for (const q of questions) {
          answers[q.qid] = answerFor(q)
        }

        await answerGroupClarify(entry, member, answers)
      }

      // Echo the exchange into the room log so the thread reads complete.
      const summary = isApproval
        ? `${answerFor(questions[0])} — ${entry.command || entry.question || 'command approval'}`
        : questions
            .map(q => (questions.length > 1 ? `${q.question}: ${answerFor(q)}` : answerFor(q)))
            .join('\n')
      appendGroupChatEntry(group, { kind: 'user', name: 'You' }, summary, entry.thread || 'legacy')
    } catch (err) {
      host.notify({ kind: 'error', message: `Could not send the answer to @${botHandle(entry.member, member)}: ${err?.message || err}` })
    } finally {
      setSending(false)
    }
  }

  return jsxs('div', {
    className:
      'grid gap-1.5 rounded-md border border-(--ui-accent,#4f9cf9)/50 bg-(--ui-accent,#4f9cf9)/5 px-2.5 py-2',
    children: [
      jsxs('div', {
        className: 'flex items-center gap-1.5 text-xs font-medium',
        children: [
          jsx(Codicon, {
            name: isApproval ? 'shield' : 'question',
            className: 'shrink-0 text-(--ui-accent,#4f9cf9)'
          }),
          isApproval
            ? `@${botHandle(entry.member, member)} wants to run a command:`
            : `@${botHandle(entry.member, member)} asks:`
        ]
      }),
      isApproval && entry.command
        ? jsx('code', {
            className:
              'block overflow-x-auto rounded bg-(--ui-bg-secondary,rgba(0,0,0,0.25)) px-2 py-1 font-mono text-[0.7rem] whitespace-pre-wrap break-all',
            children: entry.command
          })
        : null,
      ...questions.map(q =>
        jsxs('div', {
          className: 'grid gap-1',
          children: [
            q.question
              ? jsx('div', { className: 'text-xs whitespace-pre-wrap', children: q.question })
              : null,
            q.choices.length
              ? jsx('div', {
                  className: 'flex flex-wrap gap-1',
                  children: q.choices.map(choice => {
                    const chosen = (picked[q.qid] || []).includes(choice)

                    return jsx(Button, {
                      size: 'sm',
                      variant: chosen ? 'default' : 'secondary',
                      className: cn(
                        'h-6 px-2 text-[0.7rem]',
                        isApproval && choice === 'deny' && !chosen && 'text-destructive'
                      ),
                      onClick: () => {
                        setDrafts(prev => ({ ...prev, [q.qid]: '' }))
                        setPicked(prev => {
                          const current = prev[q.qid] || []
                          const next = q.multiSelect
                            ? chosen
                              ? current.filter(c => c !== choice)
                              : [...current, choice]
                            : chosen
                              ? []
                              : [choice]

                          return { ...prev, [q.qid]: next }
                        })
                      },
                      children: choice
                    }, `choice:${q.qid}:${choice}`)
                  })
                })
              : null,
            // Approvals are a closed choice set — no free-text input.
            isApproval
              ? null
              : jsx(Input, {
                  'aria-label': `Answer @${entry.member}`,
                  placeholder: q.choices.length ? 'Or type your own answer…' : 'Type your answer…',
                  value: drafts[q.qid] || '',
                  onChange: event => {
                    const value = event.target.value
                    setPicked(prev => ({ ...prev, [q.qid]: [] }))
                    setDrafts(prev => ({ ...prev, [q.qid]: value }))
                  },
                  onKeyDown: event => {
                    // IME guard: Enter confirming a composed word must not submit.
                    if (event.nativeEvent?.isComposing || event.keyCode === 229) {
                      return
                    }
                    if (event.key === 'Enter' && questions.length === 1) {
                      event.preventDefault()
                      void submit()
                    }
                  },
                  className: 'h-7 text-xs'
                }, `input:${q.qid}`)
          ]
        }, `q:${q.qid}`)
      ),
      jsx('div', {
        className: 'flex justify-end',
        children: jsx(Button, {
          size: 'sm',
          disabled: sending || !allAnswered || !member,
          onClick: () => void submit(),
          children: sending ? 'Sending…' : isApproval ? 'Respond' : 'Answer'
        })
      })
    ]
  })
}

// Group composer drafts are window-local UI state. They must survive pane
// parking/re-registration and owner switches, but must never enter shared room
// metadata (where another Desktop would see half-typed text or attachment
// bytes). Current rooms key by immutable roomId; legacy rooms fall back to the
// display name until they are upgraded.
const groupComposerDrafts = new Map()

function emptyGroupComposerDraft() {
  return { activeReplyThread: null, main: '', pendingAttachments: {}, replies: {}, revision: 0 }
}

function groupComposerDraftKey(group, room) {
  return groupChatRoomKey(group, room)
}

function groupComposerDraftSnapshot(key) {
  return groupComposerDrafts.get(key) || emptyGroupComposerDraft()
}

function updateGroupComposerDraft(key, mutate) {
  const current = groupComposerDraftSnapshot(key)
  const next = mutate({
    ...current,
    pendingAttachments: Object.fromEntries(
      Object.entries(current.pendingAttachments || {}).map(([thread, attachments]) => [
        thread,
        [...(attachments || [])]
      ])
    ),
    replies: { ...(current.replies || {}) }
  })

  next.revision = current.revision + 1
  groupComposerDrafts.delete(key)
  groupComposerDrafts.set(key, next)

  return next
}

function restoreGroupComposerDraft(key, expectedRevision, snapshot) {
  const current = groupComposerDraftSnapshot(key)

  if (current.revision !== expectedRevision) {
    return null
  }

  const restored = {
    ...snapshot,
    pendingAttachments: Object.fromEntries(
      Object.entries(snapshot.pendingAttachments || {}).map(([thread, attachments]) => [
        thread,
        [...(attachments || [])]
      ])
    ),
    replies: { ...(snapshot.replies || {}) },
    revision: current.revision + 1
  }

  groupComposerDrafts.set(key, restored)

  return restored
}

function clearGroupComposerDraft(key) {
  groupComposerDrafts.delete(key)
}

function migrateGroupComposerDraft(oldKey, newKey) {
  if (oldKey === newKey || !groupComposerDrafts.has(oldKey)) {
    return
  }

  if (!groupComposerDrafts.has(newKey)) {
    groupComposerDrafts.set(newKey, groupComposerDrafts.get(oldKey))
  }

  groupComposerDrafts.delete(oldKey)
}

function GroupChatWorkspace({ group, members, onBack, visible = true }) {
  const rooms = useValue($groupChats)
  const allMeta = useValue($botMeta)
  const room = rooms[group] || { log: [], running: false }
  const composerKey = groupComposerDraftKey(group, room)
  const composerKeyRef = useRef(composerKey)
  const [composerDraft, setComposerDraft] = useState(() => groupComposerDraftSnapshot(composerKey))

  if (composerKeyRef.current !== composerKey) {
    migrateGroupComposerDraft(composerKeyRef.current, composerKey)
    composerKeyRef.current = composerKey
  }

  const updateComposerDraft = mutate => {
    const next = updateGroupComposerDraft(composerKeyRef.current, mutate)
    setComposerDraft(next)

    return next
  }

  const draft = composerDraft.main || ''
  const replyDrafts = composerDraft.replies || {}
  const replyThread = composerDraft.activeReplyThread || null
  const pendingImages = composerDraft.pendingAttachments || {}
  const setDraft = value =>
    updateComposerDraft(current => ({
      ...current,
      main: typeof value === 'function' ? value(current.main || '') : value
    }))
  const setReplyDrafts = value =>
    updateComposerDraft(current => ({
      ...current,
      replies: typeof value === 'function' ? value(current.replies || {}) : value
    }))
  const setReplyThread = value =>
    updateComposerDraft(current => ({
      ...current,
      activeReplyThread:
        typeof value === 'function' ? value(current.activeReplyThread || null) : value
    }))
  const setPendingImages = value =>
    updateComposerDraft(current => ({
      ...current,
      pendingAttachments:
        typeof value === 'function' ? value(current.pendingAttachments || {}) : value
    }))
  const [confirmDisband, setConfirmDisband] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  // Click-to-disambiguate: which log entry is showing its speaker's full
  // @handle (the roster's name-device form when names collide across
  // connections). Naturally every speaker just shows its display name.
  const [revealedSpeaker, setRevealedSpeaker] = useState(null)
  // Threads, the Slack/Discord shape: entries carry a thread id. The most
  // recently active thread renders open; older ones collapse to summary rows.
  // `openThreads` is the user's explicit expand/collapse overrides, and
  // `replyThread` is the thread whose reply box currently owns the composer
  // (null = the main composer, which STARTS a new thread).
  const [openThreads, setOpenThreads] = useState({})
  // Pending image attachments per composer: `null` thread key = the main
  // composer, otherwise the reply box of that thread. Data URLs, already
  // downscaled — they ride the send into every responding member's session.

  // Scroll anchoring (#89835): rooms used to open at scroll position 0 and
  // stay there while replies streamed in. Scroll the bottom sentinel into
  // view on mount and whenever the log grows — but only when the user is
  // already near the bottom, so reading history is never yanked away.
  const bottomSentinelRef = useRef(null)
  const stickToBottomRef = useRef(true)

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

      viewport.addEventListener('scroll', onScroll, { passive: true })

      return () => viewport.removeEventListener('scroll', onScroll)
    }
  }, [])

  useEffect(() => {
    if (stickToBottomRef.current) {
      bottomSentinelRef.current?.scrollIntoView({ block: 'end' })
    }
  }, [room.log.length, room.running])

  // Retained-pane reopen (#89835 follow-up): a hot-mounted room pane stays
  // mounted while another workspace tab is active, so returning to it never
  // remounts and the mount-time anchor doesn't rerun. Re-anchor on the
  // hidden → visible edge — an explicit reopen, so it overrides a stale
  // read-position and mirrors what a fresh open does.
  const wasVisibleRef = useRef(visible)

  useEffect(() => {
    if (visible && !wasVisibleRef.current) {
      stickToBottomRef.current = true
      bottomSentinelRef.current?.scrollIntoView({ block: 'end' })
    }

    wasVisibleRef.current = visible
  }, [visible])

  const imagesFor = thread => pendingImages[thread ?? 'main'] || []

  const addImages = (thread, picked) => {
    if (!picked.length) {
      return
    }

    const key = thread ?? 'main'
    setPendingImages(prev => ({ ...prev, [key]: [...(prev[key] || []), ...picked] }))
  }

  const removeImage = (thread, index) => {
    const key = thread ?? 'main'
    setPendingImages(prev => ({ ...prev, [key]: (prev[key] || []).filter((_, i) => i !== index) }))
  }

  // Ctrl/⌘-V a screenshot (or any file) into any composer in this room.
  const pasteImages = (thread, event) => {
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

  const dropFiles = event => {
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
  const clarifyAll = useValue($groupClarify)
  const roomClarifies = Object.values(clarifyAll || {})
    .filter(entry => entry?.group === group)
    .sort((a, b) => (a.at || 0) - (b.at || 0))
  const availableMembers = members.filter(member => botSourceStatus(member).available).length
  const availabilityLabel = `${availableMembers} of ${members.length} available`
  const memberNames = members.map(b => displayName(b, botRosterMeta(b, allMeta))).join(', ') || 'No bots in this group chat'

  const header = jsxs('div', {
    className: 'flex items-center gap-2 px-2.5 pt-2.5 pb-2',
    children: [
      jsx(Button, {
        variant: 'ghost',
        size: 'sm',
        onClick: () => (onBack ? onBack() : $groupChatWorkspace.set(null)),
        children: 'Back'
      }),
      // Room picture (set via Group settings) leads the title when present.
      room.image
        ? jsx('img', {
            src: room.image,
            alt: '',
            className: 'size-6 shrink-0 rounded-md object-cover ring-1 ring-(--ui-stroke-secondary)'
          })
        : jsx('span', {
            className:
              'flex size-6 shrink-0 items-center justify-center rounded-md bg-(--chrome-action-hover) text-(--ui-text-tertiary)',
            children: jsx(Codicon, { name: 'organization' })
          }),
      jsx('div', {
        className: 'min-w-0 flex-1 truncate text-sm font-semibold',
        children: group
      }),
      jsx(Tip, {
        label: memberNames,
        children: jsx('span', {
          className: cn(
            'shrink-0 text-[0.65rem] text-(--ui-text-quaternary)',
            members.length > 0 && availableMembers < members.length && 'text-amber-600 dark:text-amber-300'
          ),
          'aria-label': availabilityLabel,
          children: members.length > 0 && availableMembers < members.length ? availabilityLabel : `${members.length} bots`
        })
      }),
      jsx(Tip, {
        label: `Group settings — rename ${group} or set a room picture`,
        children: jsx(Button, {
          variant: 'ghost',
          size: 'sm',
          className: 'shrink-0 text-(--ui-text-tertiary) hover:text-foreground',
          'aria-label': `Group settings for ${group}`,
          onClick: () => setSettingsOpen(true),
          children: jsx(Codicon, { name: 'gear' })
        })
      }),
      jsx(Tip, {
        label: `Disband the ${group} group chat`,
        children: jsx(Button, {
          variant: 'ghost',
          size: 'sm',
          className: 'shrink-0 text-(--ui-text-tertiary) hover:text-destructive',
          'aria-label': `Disband ${group}`,
          onClick: () => setConfirmDisband(true),
          children: jsx(Codicon, { name: 'trash' })
        })
      })
    ]
  })

  const memberDescriptors = () =>
    members.map(b => ({
      ...b,
      title: (b.remoteSource ? '' : allMeta[b.name]?.title) || b.title || ''
    }))

  // Activity disclosure: quiet, collapsed by default. The collapsed row shows
  // the latest event; expanding lists the current run's events newest-first.
  // Events are epoch-tagged, so a superseded run's history drops out of view.
  const activityEvents = currentGroupActivity(group)
  const latestActivity = activityEvents.length ? activityEvents[activityEvents.length - 1] : null
  const activityPanel = jsxs('div', {
    className: 'border-b border-(--ui-stroke-secondary)',
    children: [
      jsxs('button', {
        type: 'button',
        'aria-expanded': activityOpen,
        'aria-controls': `group-activity:${group}`,
        title: activityOpen ? 'Hide room activity' : 'Show room activity',
        className:
          'flex w-full items-center gap-1.5 px-2.5 py-1 text-left text-[0.7rem] text-(--ui-text-quaternary) transition-colors hover:text-foreground',
        onClick: () => setActivityOpen(prev => !prev),
        children: [
          jsx(Codicon, {
            name: activityOpen ? 'chevron-down' : 'chevron-right',
            className: 'shrink-0 text-[0.65rem]'
          }),
          jsx('span', { className: 'shrink-0 font-medium', children: 'Activity' }),
          latestActivity
            ? jsx('span', {
                className: 'min-w-0 flex-1 truncate',
                children: `${groupActivityLabel(latestActivity)} · ${relativeTime(latestActivity.at)}`
              })
            : null
        ]
      }),
      activityOpen
        ? jsx('div', {
            id: `group-activity:${group}`,
            className: 'grid gap-0.5 px-2.5 pb-1.5',
            children: activityEvents.length
              ? [...activityEvents]
                  .reverse()
                  .map((event, i) =>
                    jsxs('div', {
                      className: 'flex items-center gap-1.5 text-[0.7rem]',
                      children: [
                        jsx(Codicon, {
                          name: GROUP_ACTIVITY_GLYPHS[event.kind] || 'circle-outline',
                          className: cn('shrink-0 text-[0.65rem]', groupActivityTone(event.kind))
                        }),
                        jsx('span', {
                          className: cn('min-w-0 flex-1 truncate', groupActivityTone(event.kind)),
                          children: groupActivityLabel(event)
                        }),
                        jsx('span', {
                          className: 'shrink-0 text-[0.625rem] text-(--ui-text-quaternary)',
                          children: relativeTime(event.at)
                        })
                      ]
                    }, `${event.at}:${i}`)
                  )
              : jsx('div', {
                  className: 'px-0.5 pb-0.5 text-[0.625rem] text-(--ui-text-quaternary)',
                  children: 'No activity in this turn yet.'
                })
          })
        : null
    ]
  })

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
      pendingAttachments: { ...(current.pendingAttachments || {}), main: [] }
    }))
    // Main composer = START A NEW THREAD with the whole group (Slack shape).
    // Full descriptors ride into the turn loop: remote members keep their
    // connection fields so their turns route to their own machines.
    const minted = sendToGroupChat(group, memberDescriptors(), text, null, images)

    if (minted) {
      setOpenThreads(prev => ({ ...prev, [minted]: true }))
    } else {
      const restored = restoreGroupComposerDraft(composerKeyRef.current, cleared.revision, before)

      if (restored) {
        setComposerDraft(restored)
      }
    }
  }

  const submitReply = thread => {
    const text = (replyDrafts[thread] || '').trim()
    const images = imagesFor(thread)

    if (!text && !images.length) {
      return
    }

    const before = groupComposerDraftSnapshot(composerKeyRef.current)
    const cleared = updateComposerDraft(current => ({
      ...current,
      pendingAttachments: { ...(current.pendingAttachments || {}), [thread]: [] },
      replies: { ...(current.replies || {}), [thread]: '' }
    }))
    // Reply box = CONTINUE this thread; the member turns it triggers are
    // scoped to it.
    const sent = sendToGroupChat(group, memberDescriptors(), text, thread, images)

    if (sent) {
      setOpenThreads(prev => ({ ...prev, [thread]: true }))
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
  const attachmentRow = thread => {
    const images = imagesFor(thread)

    if (!images.length) {
      return null
    }

    return jsx('div', {
      className: 'flex flex-wrap items-center gap-1.5 px-1 pb-1',
      children: images.map((img, index) =>
        jsxs('div', {
          className:
            'flex items-center gap-1 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary,#181818) px-1 py-0.5',
          children: [
            img.kind === 'pdf' || img.kind === 'file'
              ? jsx(Codicon, {
                  name: img.kind === 'pdf' ? 'file-pdf' : 'file',
                  className: 'text-[0.9rem] text-(--ui-text-tertiary)'
                })
              : jsx('img', { src: img.data, alt: '', className: 'size-6 rounded object-cover' }),
            jsx('span', {
              className: 'max-w-32 truncate text-[0.65rem] text-(--ui-text-tertiary)',
              children: img.name || 'image'
            }),
            jsx('button', {
              type: 'button',
              className: 'cursor-pointer border-0 bg-transparent p-0 text-(--ui-text-quaternary) hover:text-foreground',
              title: 'Remove attachment',
              onClick: () => removeImage(thread, index),
              children: jsx(Codicon, { name: 'close', className: 'text-[0.65rem]' })
            })
          ]
        }, `${img.name || 'img'}:${index}`)
      )
    })
  }

  const attachButton = thread =>
    jsx(Button, {
      type: 'button',
      variant: 'ghost',
      size: 'sm',
      className: 'shrink-0 text-(--ui-text-tertiary) hover:text-foreground',
      title: 'Attach files — every responding bot sees them',
      onClick: () => void pickGroupAttachments().then(picked => addImages(thread, picked)),
      children: jsx(Codicon, { name: 'attach' })
    })

  // One log entry, rendered exactly as before conversation folding existed.
  const renderEntry = (entry, index) => {
                  const isUser = entry.from.kind === 'user'
                  const meta = isUser || entry.from.source ? null : allMeta[entry.from.name]
                  // Match this speaker back to its member descriptor so display
                  // names and disambiguating handles come from the roster (the
                  // primary "default" profile renders as Hermes, remote dupes
                  // carry their @name-device handle) instead of raw profile ids.
                  const member = isUser
                    ? null
                    : members.find(b =>
                        b.name === entry.from.name &&
                        (entry.from.source
                          ? (b.connectionLabel || b.connectionId) === entry.from.source
                          : !b.remoteSource)
                      ) || null
                  const display = isUser ? 'You' : displayName(member || { name: entry.from.name }, meta)
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
                  const { shape, color, image } = isUser
                    ? { shape: null, color: null, image: null }
                    : botAppearance(entry.from.name, meta)
                  const photo = Boolean(image && !isBackfilledFacePng(image))

                  return jsxs('div', {
                    className: cn(
                      'group flex items-start gap-2',
                      isUser ? 'rounded-md bg-(--chrome-action-hover) px-2 py-1.5' : 'px-2 py-1'
                    ),
                    children: [
                      isUser
                        ? null
                        : jsx('div', {
                            className: 'mt-0.5 shrink-0',
                            children: jsx(BotFace, {
                              shape,
                              color,
                              image: photo ? image : null,
                              size: 24,
                              name: entry.from.name
                            })
                          }),
                      jsxs('div', {
                        className: 'min-w-0 flex-1',
                        children: [
                          jsxs('div', {
                            className: 'flex items-center gap-2',
                            children: [
                              isUser
                                ? jsx('span', {
                                    className: 'text-[0.7rem] font-semibold text-foreground',
                                    children: label
                                  })
                                : jsx('button', {
                                    type: 'button',
                                    className:
                                      'cursor-pointer border-0 bg-transparent p-0 text-left text-[0.7rem] font-semibold text-(--ui-accent,#4f9cf9)',
                                    title: revealed ? 'Hide full handle' : 'Show full handle',
                                    onClick: () => setRevealedSpeaker(revealed ? null : entryKey),
                                    children: label
                                  }),
                              jsx('span', {
                                className: 'text-[0.625rem] text-(--ui-text-quaternary)',
                                children: relativeTime(entry.at)
                              }),
                              entry.text.trim()
                                ? jsx('div', {
                                    className:
                                      'ml-auto shrink-0 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100',
                                    children: jsx(CopyButton, {
                                      appearance: 'icon',
                                      buttonSize: 'icon',
                                      stopPropagation: true,
                                      text: entry.text
                                    })
                                  })
                                : null
                            ]
                          }),
                          jsx('div', {
                            className:
                              'text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0 [&_ul]:mb-1 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:mb-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_pre]:overflow-x-auto',
                            // The app shell sets user-select: none globally; message bodies opt
                            // back in so drag-select and ⌘C work in group chat logs.
                            'data-selectable-text': 'true',
                            children: Streamdown ? jsx(Streamdown, { children: entry.text }) : entry.text
                          }),
                          // User attachments: what every responding bot was
                          // shown — image previews, or a named chip for
                          // PDFs/files.
                          Array.isArray(entry.images) && entry.images.length
                            ? jsx('div', {
                                className: 'mt-1 flex flex-wrap items-center gap-1.5',
                                children: entry.images.map((img, imgIndex) =>
                                  img.kind === 'pdf' || img.kind === 'file'
                                    ? jsxs('div', {
                                        className:
                                          'flex items-center gap-1 rounded-md border border-(--ui-stroke-secondary) px-1.5 py-1 text-[0.65rem] text-(--ui-text-tertiary)',
                                        title: img.name || 'attached file',
                                        children: [
                                          jsx(Codicon, { name: img.kind === 'pdf' ? 'file-pdf' : 'file', className: 'text-[0.8rem]' }),
                                          jsx('span', { className: 'max-w-48 truncate', children: img.name || 'attached file' })
                                        ]
                                      }, `${entryKey}:img:${imgIndex}`)
                                    : jsx('img', {
                                        src: img.data,
                                        alt: img.name || 'attached image',
                                        title: img.name || 'attached image',
                                        className:
                                          'max-h-40 max-w-60 rounded-md border border-(--ui-stroke-secondary) object-contain'
                                      }, `${entryKey}:img:${imgIndex}`)
                                )
                              })
                            : null
                        ]
                      })
                    ]
                  }, entryKey)
  }

  // Threads: group entries by thread id (hydration assigned legacy ids, but
  // guard live pre-thread entries too), ordered by last activity — oldest
  // first, so the busiest/newest thread sits at the bottom by the composer.
  // The most recently ACTIVE thread renders open; older ones collapse to a
  // Slack-style summary row unless explicitly opened. Every open thread gets
  // its own reply box, which continues THAT thread.
  const threadsById = new Map()

  for (let i = 0; i < room.log.length; i++) {
    const entry = room.log[i]
    const id = groupThreadOf(entry)
    let bucket = threadsById.get(id)

    if (!bucket) {
      bucket = { entries: [], id, startIndex: i }
      threadsById.set(id, bucket)
    }

    bucket.entries.push({ entry, index: i })
  }

  const threads = [...threadsById.values()].sort(
    (a, b) => (a.entries[a.entries.length - 1].entry.at || 0) - (b.entries[b.entries.length - 1].entry.at || 0)
  )
  const newestThread = threads.length ? threads[threads.length - 1].id : null
  const logChildren = []

  threads.forEach(threadBucket => {
    const { entries, id } = threadBucket
    const head = entries.find(({ entry }) => entry.from.kind === 'user')?.entry || entries[0].entry
    const isNewest = id === newestThread
    const expanded = openThreads[id] ?? isNewest

    if (!expanded) {
      const replies = entries.length - 1
      const headText = stripPreviewMarkdown(head?.text || '').slice(0, 80)

      logChildren.push(
        jsxs('button', {
          type: 'button',
          className:
            'flex w-full items-center gap-2 rounded-md border border-(--ui-stroke-secondary) px-2 py-1.5 text-left text-xs text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover)',
          title: 'Open this thread',
          onClick: () => setOpenThreads(prev => ({ ...prev, [id]: true })),
          children: [
            jsx(Codicon, { name: 'chevron-right', className: 'shrink-0 text-[0.65rem]' }),
            jsx('span', { className: 'min-w-0 flex-1 truncate', children: headText || 'Thread' }),
            jsx('span', {
              className: 'shrink-0 text-[0.625rem] text-(--ui-text-quaternary)',
              children: `${replies} ${replies === 1 ? 'reply' : 'replies'} · ${relativeTime(entries[entries.length - 1].entry.at)}`
            })
          ]
        }, `fold:${id}`)
      )

      return
    }

    // Open thread: a rail-indented block — collapse affordance, its entries,
    // and its own reply box (Slack's "reply in thread").
    const threadRows = []

    if (!isNewest || openThreads[id] !== undefined) {
      threadRows.push(
        jsxs('button', {
          type: 'button',
          className:
            'flex w-full items-center gap-1.5 px-2 pt-1 text-left text-[0.65rem] text-(--ui-text-quaternary) transition-colors hover:text-foreground',
          title: 'Collapse this thread',
          onClick: () => setOpenThreads(prev => ({ ...prev, [id]: false })),
          children: [jsx(Codicon, { name: 'chevron-down', className: 'text-[0.6rem]' }), 'Collapse thread']
        }, `unfold:${id}`)
      )
    }

    for (const { entry, index } of entries) {
      threadRows.push(renderEntry(entry, index))
    }

    // Reply-in-thread: the newest thread's continuation ALSO lives here, so
    // the main composer below can stay "new thread" without ambiguity.
    threadRows.push(
      replyThread === id
        ? jsxs('form', {
            className: 'grid gap-0 px-2 pb-1',
            onSubmit: event => {
              event.preventDefault()
              submitReply(id)
            },
            children: [
              attachmentRow(id),
              jsxs('div', {
                className: 'flex items-center gap-1.5',
                children: [
                  jsx(GroupMentionInput, {
                    'aria-label': 'Reply in thread',
                    autoFocus: true,
                    placeholder: 'Reply in thread…',
                    members,
                    value: replyDrafts[id] || '',
                    onChange: text => setReplyDrafts(prev => ({ ...prev, [id]: text })),
                    onSubmitDraft: () => submitReply(id),
                    onPaste: event => pasteImages(id, event)
                  }),
                  attachButton(id),
                  jsx(Button, {
                    type: 'submit',
                    size: 'sm',
                    disabled: !(replyDrafts[id] || '').trim() && !imagesFor(id).length,
                    children: 'Reply'
                  })
                ]
              })
            ]
          }, `replybox:${id}`)
        : jsx('button', {
            type: 'button',
            className:
              'w-fit px-2 pb-1 text-left text-[0.65rem] text-(--ui-accent,#4f9cf9) transition-colors hover:underline',
            onClick: () => setReplyThread(id),
            children: 'Reply in thread'
          }, `replylink:${id}`)
    )

    logChildren.push(
      jsx('div', {
        className: 'grid gap-1.5 border-l-2 border-(--ui-stroke-secondary) pl-1.5',
        children: threadRows
      }, `thread:${id}`)
    )
  })

  return jsxs('div', {
    className: 'relative flex h-full flex-col',
    onDragOver: event => {
      if ([...(event.dataTransfer?.types || [])].includes('Files')) {
        event.preventDefault()
        setDragOver(true)
      }
    },
    onDragLeave: event => {
      // Only clear when leaving the room container itself, not when the
      // cursor moves between its children.
      if (!event.currentTarget.contains(event.relatedTarget)) {
        setDragOver(false)
      }
    },
    onDrop: dropFiles,
    children: [
      dragOver
        ? jsx('div', {
            className:
              'pointer-events-none absolute inset-0 z-40 flex items-center justify-center border-2 border-dashed border-(--ui-accent,#4f9cf9) text-sm font-medium text-(--ui-accent,#4f9cf9)',
            children: replyThread ? 'Drop to attach to this thread reply' : 'Drop to attach — every responding bot sees it'
          }, 'dropzone')
        : null,
      header,
      activityPanel,
      jsx(ScrollArea, {
        className: 'min-h-0 flex-1',
        children: jsxs('div', {
          className: 'grid gap-1.5 px-2.5 pb-2',
          children: [
            ...(room.log.length
              ? logChildren
              : [
                  jsx('div', {
                    className: 'px-2 py-4 text-center text-xs text-(--ui-text-tertiary)',
                    children: 'Say something — every bot in this group hears the room.'
                  }, 'empty')
                ]),
            ...roomClarifies.map(entry =>
              jsx(GroupClarifyCard, { entry, members }, `clarify:${entry.memberKey}:${entry.requestId}`)
            ),
            room.running
              ? jsx('div', {
                  className: 'px-2 py-1 text-[0.7rem] italic text-(--ui-text-quaternary)',
                  children: roomClarifies.length
                    ? 'Waiting for your answer…'
                    : room.turn
                      ? `${groupSpeakerLabel(room.turn)} is thinking…`
                      : 'The room is working…'
                }, 'working')
              : null,
            // Scroll anchor (#89835): rooms opened at scroll position 0, mid-
            // history. The effect below scrolls this sentinel into view on
            // mount and on log growth — unless the user has scrolled up.
            jsx('div', { ref: bottomSentinelRef, 'aria-hidden': true }, 'bottom-sentinel')
          ]
        })
      }),
      jsx('div', {
        className: 'border-t border-(--ui-stroke-secondary) p-2',
        children: jsxs('form', {
          className: 'grid gap-0',
          onSubmit: event => {
            event.preventDefault()
            submit()
          },
          children: [
            attachmentRow(null),
            jsxs('div', {
              className: 'flex items-center gap-1.5',
              children: [
                jsx(GroupMentionInput, {
                  'aria-label': `Message ${group}`,
                  placeholder: `New thread in ${group}… (@name to direct, @everyone for all)`,
                  members,
                  value: draft,
                  onChange: setDraft,
                  onSubmitDraft: submit,
                  onPaste: event => pasteImages(null, event)
                }),
                attachButton(null),
                jsx(Button, {
                  type: 'submit',
                  size: 'sm',
                  disabled: !draft.trim() && !imagesFor(null).length,
                  children: 'New Thread'
                })
              ]
            })
          ]
        })
      }),
      jsx(GroupChatSettingsDialog, {
        group,
        members,
        open: settingsOpen,
        onClose: () => setSettingsOpen(false)
      }),
      jsx(ConfirmDialog, {
        open: confirmDisband,
        title: 'Disband group chat?',
        description: jsxs('span', {
          children: [
            'This removes the ',
            jsx('span', { className: 'font-medium text-foreground', children: group }),
            ' grouping from its ',
            String(members.length),
            // New rooms title member sessions by roomId, legacy rooms by name —
            // so the copy names the concept, not a literal session title.
            ' bots and clears the shared room log. The bots themselves and their per-group sessions are kept.'
          ]
        }),
        destructive: true,
        confirmLabel: 'Disband',
        busyLabel: 'Disbanding…',
        doneLabel: 'Disbanded',
        onClose: () => setConfirmDisband(false),
        onConfirm: async () => {
          clearGroupComposerDraft(composerKeyRef.current)
          await disbandGroupChat(group, members)
          host.notify({ kind: 'success', message: `Disbanded “${group}”` })
        }
      })
    ]
  })
}

/** Live closers for group-chat MAIN-window tabs, by group name — so a
 *  disband (or the room view's own Back) can retire the tab it opened. */
const groupChatMainTabs = new Map()

/** Reactive shadow of `groupChatMainTabs` membership. The Map itself can't
 *  notify React, and #89788's first fix read it non-reactively: a BotsPane
 *  render that landed between selecting the group and recording its main
 *  tab kept the in-pane room on screen forever (the map write repaints
 *  nothing). Every map mutation goes through the two helpers below so the
 *  rev bump re-evaluates the gate. */
const $groupMainTabsRev = atom(0)

function recordGroupMainTab(group, close) {
  groupChatMainTabs.set(group, close)
  $groupMainTabsRev.set($groupMainTabsRev.get() + 1)
}

function dropGroupMainTab(group) {
  if (groupChatMainTabs.delete(group)) {
    $groupMainTabsRev.set($groupMainTabsRev.get() + 1)
  }
}

/** The in-panel room is the FALLBACK surface, not a second copy: it renders
 *  only while no main-window tab owns the group. On desktops with the door
 *  the room already lives in a main tab, and painting it here too produced
 *  two live panes with independent drafts driving one shared engine (#89788).
 *  The selection atom stays set either way so the roster row still
 *  highlights. Callers must subscribe to `$groupMainTabsRev` (BotsPane does)
 *  so ownership changes re-run this gate. */
function shouldRenderGroupChatInPane(group) {
  return Boolean(group && !groupChatMainTabs.has(group))
}

function closeGroupChatMainTab(group) {
  const close = groupChatMainTabs.get(group)

  dropGroupMainTab(group)

  if ($groupChatWorkspace.get() === group) {
    $groupChatWorkspace.set(null)
  }

  if (typeof close === 'function') {
    try {
      close()
    } catch {
      /* tab already gone */
    }
  }
}

function selectedRosterBot(roster, key) {
  return (Array.isArray(roster) ? roster : []).find(bot => botRosterKey(bot) === key) || null
}

/** A selected owner whose roster row is absent because its SOURCE is down —
 *  not because the bot is gone. Identity comes from the key itself, so the
 *  selection survives a relaunch with that gateway offline and reconciles
 *  onto the live row (same key) when it returns, without duplicating it.
 *
 *  Returns null when the selection is provably invalid instead: a reachable
 *  source that no longer lists the bot, or a source that left the registry
 *  while other sources are live. Unknown (no sources yet) is NOT proof. */
function ghostRosterOwner(key, sources) {
  const { connectionId, name } = parseRosterKey(key)

  if (!name) {
    return null
  }

  const list = Array.isArray(sources) ? sources : []
  const source = sourceByConnection(list).get(connectionId)

  if (source ? source.reachable === true : list.length > 0) {
    return null
  }

  return {
    name,
    connectionId,
    ghost: true,
    remoteSource: connectionId !== 'local',
    connectionKind: source?.kind,
    connectionLabel: source?.label,
    sourceError: source?.error || null,
    sourceMissing: false,
    sourceReachable: false
  }
}

/** Keep the exact selected owner visible through a cold-start outage without
 *  persisting the whole remote roster. The source registry supplies the
 *  gateway identity/status; the source-qualified selection supplies the bot
 *  identity. Once that source answers again, the live row replaces the ghost
 *  (or reconciliation clears it when the bot was actually removed). */
function rosterWithSelectedOwner(roster, sources, key) {
  const rows = Array.isArray(roster) ? roster : []

  if (!key || selectedRosterBot(rows, key)) {
    return rows
  }

  const ghost = ghostRosterOwner(key, sources)

  return ghost ? [...rows, ghost] : rows
}

/** Keep the persisted selection honest against the live roster and seat a
 *  first selection when there is none. PRESENTATION ONLY: it never opens,
 *  prepares, activates, or creates anything — an unreachable owner keeps its
 *  selection rather than falling back onto some other gateway's bot. */
function reconcileRosterSelection(roster, sources, metaByName) {
  if (!$rosterHydrated.get() || !$selectedRosterHydrated.get()) {
    return
  }

  const key = $selectedRosterKey.get()

  if (key) {
    if (selectedRosterBot(roster, key) || ghostRosterOwner(key, sources)) {
      return
    }

    clearSelectedRosterKey(key)
  }

  const first = (Array.isArray(roster) ? roster : []).find(
    bot => !isBotHidden(bot, metaByName) && botSourceStatus(annotateBotSource(bot, sources)).available
  )

  if (first) {
    saveSelectedRosterBot(first)
  }
}

function BotsHomeView() {
  const roster = useValue($lastRoster)
  const sources = useValue($lastSources)
  const selectedKey = useValue($selectedRosterKey)
  const rosterHydrated = useValue($rosterHydrated)
  const selectionHydrated = useValue($selectedRosterHydrated)
  const allMeta = useValue($botMeta)
  const live = selectedRosterBot(roster, selectedKey)

  if (!rosterHydrated || !selectionHydrated) {
    return jsx('div', {
      className: 'flex h-full items-center justify-center',
      'aria-label': 'Loading bots',
      children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
    })
  }

  const ghost = live ? null : ghostRosterOwner(selectedKey, sources)
  const bot = live ? annotateBotSource(live, sources) : ghost

  if (!bot) {
    return jsx('div', {
      className: 'flex h-full items-center justify-center px-6',
      children: jsx(EmptyState, {
        icon: roster.length ? 'hubot' : 'add',
        title: roster.length ? 'Choose a bot or group chat' : 'No bots yet',
        description: roster.length ? 'Pick one from the Bots sidebar.' : 'Create your first bot from the Bots sidebar.'
      })
    })
  }

  const meta = botRosterMeta(bot, allMeta)
  const status = botSourceStatus(bot)
  // A ghost is reconstructed from a persisted owner key while its gateway is
  // offline. That proves the profile name, not its public mention handle.
  const handle = bot.ghost ? '' : botHandle(bot.name, bot)
  const gateway = bot.connectionLabel || (bot.connectionId === 'local' ? 'This device' : 'Hermes gateway')
  const gatewayKind = bot.connectionKind || (bot.connectionId === 'local' ? 'local' : 'remote')
  const { shape, color, image } = botAppearance(bot.name, meta)
  const photo = image && !isBackfilledFacePng(image) ? image : null
  const description = String(meta?.description || bot.description || '').trim()
  const unavailable = !status.available
  const sourceRemoved = status.key === 'missing'
  // Retry re-polls the roster on the bot's OWN source. It never activates or
  // re-routes anything: if the gateway is back, its row reappears under the
  // same key and this view reconciles onto it.
  const retrySource = () => {
    haptic('tap')
    queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
  }

  return jsxs('div', {
    className: 'flex h-full min-h-0 flex-col bg-background',
    children: [
      jsxs('header', {
        className:
          'flex min-w-0 items-center gap-3 border-b border-(--ui-stroke-tertiary) px-5 py-3.5',
        children: [
          jsx(BotFace, { shape, color, image: photo, size: 38, name: bot.name, mood: 'idle' }),
          jsxs('div', {
            className: 'min-w-0 flex-1',
            children: [
              jsx('h1', {
                className: 'truncate text-sm font-semibold text-foreground',
                children: displayName(bot, meta)
              }),
              jsxs('div', {
                className: 'flex min-w-0 items-center gap-1.5 text-xs text-(--ui-text-tertiary)',
                children: [
                  jsx('span', { children: 'Bot' }),
                  handle
                    ? jsx('span', { className: 'truncate font-mono', children: `· @${handle}` })
                    : null
                ]
              })
            ]
          }),
          // Tip wraps ONE element (Radix asChild) — the screen-reader text
          // rides inside the trigger, not beside it.
          jsx(Tip, {
            label: `${gateway} · ${gatewayKind} · ${status.label}`,
            children: jsxs('div', {
              className: 'flex max-w-[45%] items-center gap-1.5 text-xs text-(--ui-text-tertiary)',
              children: [
                jsx('span', { className: 'sr-only', children: `${gateway}, ${status.label}` }),
                jsx(GatewayKindGlyph, { kind: gatewayKind }),
                jsx('span', { className: 'min-w-0 truncate', children: gateway }),
                unavailable
                  ? jsx(Codicon, {
                      name: 'debug-disconnect',
                      className: 'shrink-0 text-amber-600 dark:text-amber-300',
                      'aria-hidden': true
                    })
                  : null
              ]
            })
          })
        ]
      }),
      jsx('main', {
        className: 'flex min-h-0 flex-1 items-center justify-center overflow-y-auto px-6 py-10',
        children: jsxs('div', {
          className: 'flex w-full max-w-2xl flex-col items-center text-center',
          children: [
            jsx(BotFace, { shape, color, image: photo, size: 76, name: bot.name, mood: 'idle' }),
            jsx('h2', {
              className: 'mt-5 text-xl font-semibold text-foreground',
              children: displayName(bot, meta)
            }),
            description
              ? jsx('p', {
                  className: 'mt-2 max-w-xl text-sm leading-6 text-(--ui-text-tertiary)',
                  children: description
                })
              : null,
            jsx('p', {
              className: cn(
                'mt-4 max-w-lg text-xs leading-5',
                unavailable ? 'text-amber-700 dark:text-amber-300' : 'text-(--ui-text-tertiary)'
              ),
              children: unavailable
                ? sourceRemoved
                  ? `${gateway} was removed. Choose another bot from the sidebar.`
                  : `${gateway} is unavailable. Retry when it is back online.`
                : 'Open this bot’s continuous chat. Its background work keeps running when you switch away.'
            }),
            unavailable && !sourceRemoved
              ? jsx(Button, {
                  variant: 'secondary',
                  size: 'sm',
                  className: 'mt-5',
                  onClick: retrySource,
                  children: 'Retry'
                })
              : jsx(Button, {
                  variant: 'secondary',
                  size: 'sm',
                  className: 'mt-5',
                  onClick: () => void openRosterBot(bot),
                  children: 'Open chat'
                })
          ]
        })
      })
    ]
  })
}

function closeBotsHomeWorkspace() {
  if (typeof botsHomeClose !== 'function') {
    return
  }

  const close = botsHomeClose
  botsHomeClose = null
  suppressBotsHomeReopen = true
  // Retiring the tab ends the current attempt's budget: the next open is a
  // new surface, and it gets its own re-front chance. The re-front path sets
  // the latch again AFTER calling us, so this cannot erase its own attempt.
  botsHomeRefrontTried = false

  try {
    close()
  } catch {
    /* workspace already closed */
  } finally {
    suppressBotsHomeReopen = false
  }
}

/** The Bot home needs BOTH the main-area door and pane visibility to behave.
 *  Older shells keep their previous surfaces untouched (no home at all). */
function botsHomeEnabled() {
  return typeof host.openWorkspace === 'function' && typeof host.paneVisibility === 'function'
}

/** True when a session owns the main workspace. Prefers the focused STORED
 *  session (tab focus moves without swapping the gateway socket); bare test
 *  harnesses with neither atom drive $botChatFocused directly. */
function sessionOwnsWorkspace() {
  const focused = host.state?.focusedStoredSessionId?.get?.()

  if (focused !== undefined) {
    return Boolean(focused)
  }

  const active = host.state?.activeSessionId?.get?.()

  return active === undefined ? $botChatFocused.get() : Boolean(active)
}

/** The home tab currently holds the center's active tab slot. */
function botsHomeVisible() {
  if (typeof host.paneVisibility !== 'function') {
    return false
  }

  try {
    return host.paneVisibility(BOTS_HOME_PANE_ID).get() === true
  } catch {
    return false
  }
}

/** A real bot chat owns the center. Cronjobs are BOT-scoped, so this — not
 *  mere Bot Mode visibility — is what may seat the Cronjobs tile: beside the
 *  ownerless home or a group chat it would describe whichever profile the
 *  socket happens to be homed on. While the home tab is fronted the chat is
 *  a hidden sibling layer, so the focused session does NOT count. */
function botChatOwnsWorkspace() {
  return (
    $botsPaneVisible.get() &&
    !$groupChatWorkspace.get() &&
    !botsHomeVisible() &&
    Boolean($openBotChat.get() || sessionOwnsWorkspace())
  )
}

/** May the home OPEN right now? `explicit` is a user gesture aimed at the
 *  home itself (selecting a remote/unavailable owner): it overrides the
 *  focused-session veto — the veto exists so PASSIVE events (boot, restore,
 *  polls) never cover a chat the user left in the center. */
function botsHomeMayOpen(explicit) {
  return (
    $botsPaneVisible.get() &&
    !$groupChatWorkspace.get() &&
    !$openBotChat.get() &&
    (explicit || !sessionOwnsWorkspace())
  )
}

function openBotsHomeWorkspace(explicit = false) {
  if (!botsHomeEnabled() || !botsHomeMayOpen(explicit)) {
    return false
  }

  const selected = selectedRosterBot($lastRoster.get(), $selectedRosterKey.get())
  const ownerKey = selected ? botWorkspaceOwnerKey(selected) : BOTS_HOME_OWNER_KEY
  setBotsWorkspaceOwner(ownerKey, selected)
  // Already open and fronted: nothing to do. Already open but backgrounded
  // (a persisted layout can restore the tab behind the draft): re-open to
  // re-front it. Never stack a second registration — a stale disposer would
  // tear down the newer one. This cannot yank the center from a tab the
  // user just chose: plugin events are sparse (sidebar/group/focus edges),
  // and each of those either legitimately claims the center or cleared it.
  if (botsHomeClose) {
    if (botsHomeVisible()) {
      botsHomeRefrontTried = false

      return true
    }

    // A re-front is a close + re-open, so it REMOUNTS the whole Bots view.
    // That is affordable once, against the backgrounded-tab case above. It is
    // not affordable per signal: the shell does not always answer a reveal
    // with the active slot (revealTreePane returns early for a pane in
    // $hiddenTreePanes without activating it; a minimized zone and a pane the
    // tree never adopted are never visible either). Pinned in one of those
    // states the old code re-fronted on every passive pass — sidebar flips,
    // focus churn and group changes all reach here — and the view strobed.
    // One attempt proves whether this shell will front the tab; if it will
    // not, keep the surface and wait for a signal that changes the answer.
    if (botsHomeRefrontTried && !explicit) {
      return true
    }

    closeBotsHomeWorkspace()
    botsHomeRefrontTried = true
  }

  try {
    botsHomeClose = host.openWorkspace(`${ID}:home`, {
      title: 'Bots',
      minWidth: '24rem',
      render: () => jsx(BotsHomeView, {}),
      // Closing the tab is a decision, not a glitch: drop the handle and
      // leave the center alone. The home returns on the next real signal
      // (Bots tab regains focus, a chat closes, a group is left).
      onClose: () => {
        if (!suppressBotsHomeReopen) {
          botsHomeClose = null
        }
      }
    })

    // The reveal has already either granted the tab its zone's active slot or
    // refused it, so settle the re-front budget on that answer directly rather
    // than waiting for a visibility notification to schedule another pass. A
    // computed store stays silent when the value does not change, so on the
    // shells that refuse, no such pass is coming.
    if (botsHomeVisible()) {
      botsHomeRefrontTried = false
    }

    return typeof botsHomeClose === 'function'
  } catch {
    botsHomeClose = null
    return false
  }
}

/** Passive reconcile. Opens the home only into an ownerless center; closes
 *  it only when a surface with a REAL owner claims the center (bot chat,
 *  group chat) or Bot Mode leaves the screen. The focused-session LEVEL
 *  deliberately does not close an open home — the home may sit over a
 *  focused-but-hidden chat after an explicit selection; the chat reclaims
 *  the center on its focus EDGE (handleWorkspaceFocusChange). */
function syncBotsHomeWorkspace() {
  if (!$botsPaneVisible.get() || $groupChatWorkspace.get() || $openBotChat.get()) {
    closeBotsHomeWorkspace()
    return
  }

  openBotsHomeWorkspace(false)
}

/** An opened bot chat stops owning the center once focus leaves it (closed,
 *  or another session took over). Without this the home could never come
 *  back: $openBotChat would claim ownership for a chat nobody is reading.
 *
 *  The legacy newChat fallback has no registry id to compare — a draft with no
 *  focused session is still that bot's draft, so it only yields once some
 *  session actually takes focus. */
function releaseStaleOpenBotChat(focusedStoredId) {
  const open = $openBotChat.get()

  if (!open) {
    return
  }

  const focused = focusedStoredId === null || focusedStoredId === undefined ? '' : String(focusedStoredId)
  // The focused stored id is the compression-lineage TIP; the claim carries
  // both the durable registry id and the tip it actually opened. Either
  // match keeps the claim — comparing only the registry id released it on
  // the very focus edge the open itself caused (first-click home bounce).
  const owned = [open.openedSessionId, open.openedRegistryId].filter(Boolean)
  const stale = owned.length ? !owned.includes(focused) : Boolean(focused)

  if (stale) {
    $openBotChat.set(null)
  }
}

/** Main-window wrapper: seats the member roster reactively (live roster +
 *  bot meta + the room's stored cross-connection descriptors) so the room
 *  keeps working as members change while the tab is open. Also subscribes to
 *  this pane's visibility (feature-detected host.paneVisibility): retained
 *  panes stay mounted while hidden, so the workspace needs the hidden →
 *  visible edge to re-anchor its log to the bottom (#89835 follow-up). */
function GroupChatMainView({ group }) {
  const allMeta = useValue($botMeta)
  // Subscribe: membership changes ride bot meta AND the room record.
  useValue($groupChats)
  const roster = useValue($lastRoster)
  const members = groupChatMemberBots(group, roster, allMeta)
  // Older SDKs have no paneVisibility: fall back to an always-visible atom so
  // the hook order stays stable and behavior matches the previous build.
  const $visible = useMemo(
    () => (typeof host.paneVisibility === 'function' ? host.paneVisibility(`plugin-workspace:${ID}:group:${slugify(group)}`) : atom(true)),
    [group]
  )
  const visible = useValue($visible)

  return jsx(GroupChatWorkspace, { group, members, visible, onBack: () => closeGroupChatMainTab(group) })
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
function openGroupChat(group) {
  // A room selection supersedes any bot-open transition still hydrating.
  // The in-flight host navigation may complete underneath this workspace,
  // but it may not later close or visually steal the room the user chose.
  botOpenGeneration += 1
  $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: false })
  const ownerKey = groupWorkspaceOwnerKey(group)
  setBotsWorkspaceOwner(ownerKey, null, 'New group conversations start in the group composer.')

  if (typeof host.openWorkspace === 'function') {
    try {
      const close = host.openWorkspace(`${ID}:group:${slugify(group)}`, {
        title: group,
        minWidth: '24rem',
        render: () => jsx(GroupChatMainView, { group }),
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

/** One group chat as one quiet roster row. The room owns one visual identity;
 *  member details stay in its tooltip and workspace instead of competing
 *  with bot avatars in the narrow sidebar. */
function GroupRow({ active, group, members, needsYou, onOpen, onDisband }) {
  const rooms = useValue($groupChats)
  const room = rooms[group] || { log: [] }
  const log = Array.isArray(room.log) ? room.log : []
  const last = log.length ? log[log.length - 1] : null
  const lastAt = groupLastActivity(room)
  // Room previews speak the same handle vocabulary as the roster, mentions
  // and the group prompt: the primary profile is @hermes, not @default.
  const lastFrom = last?.from?.name || ''
  const lastHandle = botHandle(lastFrom || 'bot', members.find(member => member?.name === lastFrom))
  const preview = last
    ? `${last.from?.kind === 'user' ? 'You' : `@${lastHandle}`}: ${stripPreviewMarkdown(last.text) || '…'}`
    : `${members.length} bots`
  const availableMembers = members.filter(member => botSourceStatus(member).available).length
  const availabilityLabel = `${availableMembers} of ${members.length} available`

  const row = jsxs('button', {
    type: 'button',
    onClick: () => {
      haptic('tap')
      onOpen(group)
    },
    className: cn(
      'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
      'hover:bg-(--chrome-action-hover)',
      active && 'bg-(--ui-row-active-background)'
    ),
    'aria-label': `${group}, ${members.length} bots, ${availabilityLabel}`,
    children: [
      jsxs('div', {
        className: 'relative flex w-[34px] shrink-0 items-center justify-center',
        children: [
          room.image
            ? jsx('img', {
                src: room.image,
                alt: '',
                className: cn(
                  'size-8 rounded-md object-cover ring-1 ring-(--ui-stroke-tertiary)',
                  availableMembers === 0 && 'grayscale opacity-60'
                )
              })
            : jsx('span', {
                className: cn(
                  'flex size-8 items-center justify-center rounded-md bg-(--chrome-action-hover) text-(--ui-text-tertiary)',
                  availableMembers === 0 && 'opacity-60'
                ),
                children: jsx(Codicon, { name: 'organization' })
              }),
          members.length > 0 && availableMembers < members.length
            ? jsx(Tip, {
                label: availabilityLabel,
                children: jsx('span', {
                  className:
                    'absolute -bottom-0.5 -right-0.5 flex size-4 items-center justify-center rounded-full bg-(--ui-bg-primary) text-[0.625rem] text-amber-600 ring-1 ring-(--ui-stroke-tertiary) dark:text-amber-300',
                  'aria-label': availabilityLabel,
                  children: jsx(Codicon, { name: 'debug-disconnect' })
                })
              })
            : null
        ]
      }),
      jsxs('div', {
        className: 'min-w-0 flex-1',
        children: [
          jsxs('div', {
            className: 'flex items-baseline justify-between gap-2',
            children: [
              jsx('span', {
                className: 'min-w-0 flex-1 truncate text-[0.8125rem] font-medium',
                children: group
              }),
              needsYou
                ? jsx(Tip, {
                    label: 'A bot in this group chat needs your input',
                    children: jsx(Codicon, {
                      name: 'question',
                      className: 'shrink-0 text-(--ui-accent)',
                      'aria-label': 'Needs your input'
                    })
                  })
                : null,
              lastAt
                ? jsx('span', {
                    className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                    children: relativeTime(lastAt)
                  })
                : null
            ]
          }),
          jsx('div', {
            className: 'min-w-0 truncate text-xs text-(--ui-text-tertiary)',
            children: preview
          })
        ]
      })
    ]
  })

  return jsxs(ContextMenu, {
    children: [
      jsx(ContextMenuTrigger, { asChild: true, children: row }),
      jsxs(ContextMenuContent, {
        children: [
          jsx(ContextMenuItem, {
            onSelect: () => onOpen(group),
            children: 'Open Group Chat'
          }),
          jsx(ContextMenuSeparator, {}),
          jsx(ContextMenuItem, {
            className: 'text-destructive focus:text-destructive',
            onSelect: () => onDisband({ name: group, members }),
            children: 'Delete Group'
          })
        ]
      })
    ]
  })
}

/** Foldable roster heading. It organizes rows visually but never supplies or
 * reconstructs ownership; every action still receives the full bot row. */
function RosterSectionHeader({ collapsed, count, gatewayKind, icon, label, onToggle, status, tip }) {
  const button = jsxs('button', {
    type: 'button',
    'aria-expanded': !collapsed,
    className:
      'mt-1 flex w-full min-w-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary) transition-colors hover:bg-(--chrome-action-hover) hover:text-(--ui-text-secondary)',
    onClick: onToggle,
    children: [
      jsx(Codicon, { name: collapsed ? 'chevron-right' : 'chevron-down', className: 'shrink-0' }),
      gatewayKind
        ? jsx(GatewayKindGlyph, { kind: gatewayKind })
        : jsx(Codicon, { name: icon, className: 'shrink-0' }),
      jsxs('span', {
        className: 'flex min-w-0 items-center gap-1',
        children: [
          jsx('span', { className: 'min-w-0 truncate', children: label }),
          status && !status.available
            ? jsx('span', { className: 'sr-only', children: status.label })
            : null
        ]
      }),
      jsx('span', { className: 'min-w-0 flex-1', 'aria-hidden': true }),
      jsx('span', {
        className: 'shrink-0 font-normal tabular-nums text-(--ui-text-quaternary)',
        children: count
      }),
      status && !status.available
        ? jsx(Codicon, {
            name: 'debug-disconnect',
            className: 'shrink-0 text-amber-600 dark:text-amber-300',
            'aria-hidden': true
          })
        : null
    ]
  })

  return tip ? jsx(Tip, { label: tip, children: button }) : button
}

function GatewaySectionHeading({ collapsed, count, onToggle, option }) {
  const status = botSourceStatus({ sourceError: option?.error, sourceReachable: option?.reachable })
  const label = option?.label || option?.connectionId || 'Current gateway'
  const kind = option?.kind || 'remote'

  return jsx(RosterSectionHeader, {
    collapsed,
    count,
    gatewayKind: kind,
    label,
    onToggle,
    status,
    tip: `${label} · ${kind} · ${status.label}`
  })
}

function BotsPane() {
  const { data, error, isLoading, refetch } = useRoster()
  const gatewayState = useValue(host.state.gateway)
  const gatewayUp = gatewayState === 'open'
  const activeProfile = (useValue(host.state.profile) || 'default').trim() || 'default'
  const [createOpen, setCreateOpen] = useState(false)
  const [groupCreateOpen, setGroupCreateOpen] = useState(false)
  const [editing, setEditing] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const [deletingGroup, setDeletingGroup] = useState(null)
  const [grouping, setGrouping] = useState(null)
  const [query, setQuery] = useState('')
  const [rowKindFilter, setRowKindFilter] = useState('all')
  const [activityFilter, setActivityFilter] = useState('all')
  const [gatewayFilter, setGatewayFilter] = useState('all')
  const [collapsedRosterSections, setCollapsedRosterSections] = useState(() => new Set())
  const hiddenSectionRef = useRef(null)
  const activityToasts = useValue($activityToasts)
  const groupChatName = useValue($groupChatWorkspace)
  // Main-tab ownership is a module Map; this rev subscription makes the
  // shouldRenderGroupChatInPane gate below reactive to tab open/close
  // (#89788 follow-up — without it a stale render could paint the in-pane
  // room beside a live main tab and stick).
  useValue($groupMainTabsRev)
  const groupNeedsYou = useValue($groupNeedsYou)
  const groupRooms = useValue($groupChats)
  const rememberedSources = useValue($lastSources)
  const rosterHydrated = useValue($rosterHydrated)
  const selectionHydrated = useValue($selectedRosterHydrated)
  const selectedRosterKey = useValue($selectedRosterKey)

  // The socket opening (boot, SSH reconnect, sleep/wake) is the signal to
  // retry immediately instead of waiting out the poll interval.
  useEffect(() => {
    if (gatewayUp) {
      void refetch()
    }
  }, [gatewayUp, refetch])
  const allMeta = useValue($botMeta)
  // Messaging-app order: most recent activity first, where "activity" is
  // the newest of (bot created, last message in any of its sessions). A
  // freshly created bot tops the list until another bot gets a message.
  // No special slot for the primary bot — it competes on recency too.
  const activityOf = bot => {
    const created = botRosterMeta(bot, allMeta)?.created || bot.ui_meta?.['hermes-bots']?.created || 0
    const lastMsg = (botActivitySession(bot)?.last_active || 0) * 1000

    return Math.max(created, lastMsg)
  }
  // Pin is a source-qualified Desktop preference, not gateway profile state.
  const isPinned = bot => isBotPinned(bot, allMeta)
  // Resilience (@wesleysimplicio, #13): a failed refresh must not erase a
  // roster the user already had — mixed local+cloud gateways and remotes
  // waking from sleep fail transiently. Render the last good snapshot with
  // a notice; the full error card is reserved for "never had a roster".
  const live = Array.isArray(data?.profiles) ? data.profiles : null
  const source = live ?? (error ? $lastRoster.get() : [])
  const sourceSnapshot = Array.isArray(data?.sources) ? data.sources : rememberedSources
  const sourceWithSelectedOwner = selectionHydrated && rosterHydrated
    ? rosterWithSelectedOwner(source, sourceSnapshot, selectedRosterKey)
    : source
  const roster = sourceWithSelectedOwner.slice().sort((a, b) => {
    const pa = isPinned(a) ? 1 : 0
    const pb = isPinned(b) ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return activityOf(b) - activityOf(a)
  })
  // React Query can briefly report neither loading nor data while the plugin
  // and the persisted connection registry hydrate. Keep that transition in a
  // neutral loading state instead of flashing the first-run "No bots" copy.
  const initialRosterLoading = !data && !error && roster.length === 0
  const activeRosterKeys = new Set(activeBots(roster, activeProfile, gatewayState).map(botRosterKey))
  const gatewayOptions = rosterGatewayOptions(sourceSnapshot, roster)
  const selectedGateway = gatewayOptions.find(option => option.connectionId === gatewayFilter)
  const gatewayFilterExists = gatewayFilter === 'all' || Boolean(selectedGateway)

  useEffect(() => {
    if (!gatewayFilterExists) {
      setGatewayFilter('all')
    }
  }, [gatewayFilterExists])

  const activeSourceRoster = roster.filter(bot => !bot.remoteSource)
  // Hidden rows remain fully alive and recoverable at the bottom. Every
  // non-display consumer continues to receive the complete roster.
  const hiddenExpanded = useValue($showHiddenBots)
  const hiddenBots = roster.filter(bot => isBotHidden(bot, allMeta))
  const visibleRoster = roster.filter(bot => !isBotHidden(bot, allMeta))
  const gatewayRoster = filterBotsByGateway(visibleRoster, gatewayFilter)
  const filteredRoster = filterBots(gatewayRoster, allMeta, query).filter(bot =>
    rosterActivityMatches(
      { activity: activityOf(bot), active: activeRosterKeys.has(botRosterKey(bot)) },
      activityFilter
    )
  )
  const filteredHiddenBots = filterBots(filterBotsByGateway(hiddenBots, gatewayFilter), allMeta, query).filter(bot =>
    rosterActivityMatches(
      { activity: activityOf(bot), active: activeRosterKeys.has(botRosterKey(bot)) },
      activityFilter
    )
  )
  const groupNames = groupChatNames(allMeta, groupRooms)
  const groupRows = groupNames
    .map(name => ({ name, members: groupChatMemberBots(name, roster, allMeta) }))
    .filter(row => groupMatchesRosterFilters(row.name, row.members, allMeta, query, gatewayFilter))
    .map(row => ({
      kind: 'group',
      name: row.name,
      members: row.members,
      pinned: Boolean(groupRooms[row.name]?.pinned),
      activity: groupLastActivity(groupRooms[row.name]),
      active:
        Boolean(
          groupLastActivity(groupRooms[row.name]) &&
          Date.now() - groupLastActivity(groupRooms[row.name]) <= ACTIVE_WINDOW_S * 1000
        ) || row.members.some(member => activeRosterKeys.has(botRosterKey(member)))
    }))
    .filter(row => rowKindFilter !== 'bots' && rosterActivityMatches(row, activityFilter))
  const botRows =
    rowKindFilter === 'groups'
      ? []
      : filteredRoster.map(bot => ({
          kind: 'bot',
          bot,
          pinned: isPinned(bot),
          activity: activityOf(bot),
          active: activeRosterKeys.has(botRosterKey(bot))
        }))
  const sortRosterRows = rows => rows.slice().sort((a, b) => {
    const pa = a.pinned ? 1 : 0
    const pb = b.pinned ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return b.activity - a.activity
  })
  const rosterRows = sortRosterRows([...botRows, ...groupRows])
  const sortedGroupRows = sortRosterRows(groupRows)
  const gatewaySections = rosterGatewaySections(botRows, gatewayOptions, gatewayFilter)
  const showGatewaySections = gatewaySections.sectioned && botRows.length > 0
  const activeFilterCount =
    (rowKindFilter === 'all' ? 0 : 1) +
    (activityFilter === 'all' ? 0 : 1) +
    (gatewayFilter === 'all' ? 0 : 1)
  const hasRosterConstraint = Boolean(query.trim()) || activeFilterCount > 0
  const matchingHiddenBots = rowKindFilter === 'groups' ? [] : filteredHiddenBots
  const showHiddenSection = hiddenBots.length > 0 && (!hasRosterConstraint || matchingHiddenBots.length > 0)
  const showHiddenRows = hiddenExpanded || hasRosterConstraint
  const rosterItemCount = roster.length + groupNames.length
  const allBotsHidden =
    !hasRosterConstraint && visibleRoster.length === 0 && groupNames.length === 0 && hiddenBots.length > 0
  const showRosterSearch =
    gatewayOptions.length > 1 || rosterItemCount >= BOT_ROSTER_SEARCH_THRESHOLD || Boolean(query.trim())
  const showRosterFilters =
    gatewayOptions.length > 1 ||
    groupNames.length > 0 ||
    rosterItemCount >= BOT_ROSTER_SEARCH_THRESHOLD ||
    activeFilterCount > 0
  const showRosterTools = showRosterSearch || showRosterFilters
  const rosterSectionCollapsed = id => !hasRosterConstraint && collapsedRosterSections.has(id)
  const hiddenGatewaySections = rosterGatewaySections(
    matchingHiddenBots.map(bot => ({ kind: 'bot', bot })),
    gatewayOptions,
    gatewayFilter
  )

  const toggleRosterSection = id => {
    setCollapsedRosterSections(previous => {
      const next = new Set(previous)

      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }

      return next
    })
  }

  useEffect(() => {
    if (!hiddenExpanded || hasRosterConstraint) {
      return
    }

    const frame = requestAnimationFrame(() => hiddenSectionRef.current?.scrollIntoView({ block: 'nearest' }))

    return () => cancelAnimationFrame(frame)
  }, [hiddenExpanded, hasRosterConstraint])

  useEffect(() => {
    if (!live) {
      return
    }

    // Offline-owner ghosts belong only to this render. Shared roster state
    // feeds merge caching, group membership, creation, and durable sync. These
    // writes must settle after render: BotsHomeView subscribes to the same
    // atoms, so publishing here used to update it while BotsPane was rendering.
    $lastRoster.set(roster.filter(row => !row?.ghost))
    if (Array.isArray(data?.sources)) {
      $lastSources.set(data.sources)
    }
    mergeServerMeta(activeSourceRoster, data?.fetchedAt || 0)
    pullServerAvatars(activeSourceRoster)
    trackInboundActivity(roster)
    backfillMessagingProtocol(activeSourceRoster)
    // React Query owns the stable server snapshot; derived arrays intentionally
    // follow that snapshot rather than retriggering on their own atom writes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  // The roster has ANSWERED once data or a terminal error exists — that, not
  // row count, is what lets the home stop showing its loading state (an empty
  // answer is a real answer; a pending one must not flash "No bots"). Keep the
  // persisted-selection writes out of render: React may replay a render, but
  // an abandoned render must never become a storage mutation.
  useEffect(() => {
    if (!data && !error) {
      return
    }

    $rosterHydrated.set(true)

    if (selectionHydrated) {
      reconcileRosterSelection(roster, sourceSnapshot, allMeta)
      const selected = selectedRosterBot(roster, $selectedRosterKey.get())

      if ($botsPaneVisible.get() && !$groupChatWorkspace.get() && selected) {
        setBotsWorkspaceOwner(botWorkspaceOwnerKey(selected), selected)
      }
    }
  }, [data, error, selectionHydrated, roster, sourceSnapshot, allMeta])

  const staleNotice = error && !live && roster.length
    ? 'Roster refresh failed — showing the last good list.' + (gatewayUp ? '' : ' Waiting for the gateway to reconnect…')
    : null
  const groupChatMembers = groupChatName ? groupChatMemberBots(groupChatName, roster, allMeta) : []

  if (shouldRenderGroupChatInPane(groupChatName) && groupChatMembers.length) {
    return jsx(GroupChatWorkspace, { group: groupChatName, members: groupChatMembers })
  }

  const renderBotRow = (bot, keyPrefix = '') =>
    jsx(
      BotRow,
      {
        bot,
        onDelete: setDeleting,
        onEdit: setEditing,
        onGroup: setGrouping,
        showHandle: botNeedsHandleLabel(bot, roster, allMeta)
      },
      `${keyPrefix}${botRosterKey(bot)}`
    )

  const renderGroupRow = row =>
    jsx(
      GroupRow,
      {
        active: groupChatName === row.name,
        group: row.name,
        members: row.members,
        needsYou: Boolean(groupNeedsYou[row.name]),
        onOpen: openGroupChat,
        onDisband: setDeletingGroup
      },
      `group:${row.name}`
    )

  const renderGatewaySection = section => {
    const sectionId = `gateway:${section.id}`
    const collapsed = rosterSectionCollapsed(sectionId)

    return jsxs(
      'div',
      {
        className: 'min-w-0',
        children: [
          jsx(GatewaySectionHeading, {
            collapsed,
            count: section.rows.length,
            onToggle: () => toggleRosterSection(sectionId),
            option: section.option
          }),
          collapsed
            ? null
            : jsx('div', {
                className: 'grid min-w-0 gap-0.5',
                children: section.rows.map(row => renderBotRow(row.bot, `${section.id}:`))
              })
        ]
      },
      sectionId
    )
  }

  const renderGroupChatSection = () => {
    const sectionId = 'group-chats'
    const collapsed = rosterSectionCollapsed(sectionId)

    return jsxs(
      'div',
      {
        className: 'min-w-0',
        children: [
          jsx(RosterSectionHeader, {
            collapsed,
            count: sortedGroupRows.length,
            icon: 'organization',
            label: 'Group chats',
            onToggle: () => toggleRosterSection(sectionId),
            tip: `${sortedGroupRows.length} global group chat${sortedGroupRows.length === 1 ? '' : 's'}`
          }),
          collapsed
            ? null
            : jsx('div', {
                className: 'grid min-w-0 gap-0.5',
                children: sortedGroupRows.map(renderGroupRow)
              })
        ]
      },
      sectionId
    )
  }

  const renderHiddenGatewaySection = section =>
    jsxs(
      'div',
      {
        className: 'min-w-0',
        children: [
          jsx('div', {
            className:
              'flex min-w-0 items-center gap-1.5 px-2 py-1 text-[0.625rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)',
            children: [
              jsx(GatewayKindGlyph, { kind: section.option?.kind }),
              jsx('span', {
                className: 'min-w-0 flex-1 truncate',
                children: section.option?.label || section.option?.connectionId || 'Current gateway'
              }),
              jsx('span', { className: 'shrink-0 font-normal tabular-nums', children: section.rows.length })
            ]
          }),
          ...section.rows.map(row => renderBotRow(row.bot, `hidden:${section.id}:`))
        ]
      },
      `hidden-gateway:${section.id}`
    )

  return jsxs('div', {
    className: 'flex h-full flex-col',
    children: [
      jsxs('div', {
        className: 'flex items-center justify-between gap-2 px-2.5 pt-2.5 pb-1.5',
        children: [
          jsx('span', {
            className: 'text-[0.6875rem] font-semibold uppercase tracking-wider text-(--ui-text-quaternary)',
            children: 'Bots'
          }),
          jsxs('div', {
            className: 'flex items-center gap-0.5',
            children: [
              jsx(Tip, {
                label: activityToasts ? 'Activity toasts on — click to silence' : 'Activity toasts off — click to enable',
                children: jsx('button', {
                  type: 'button',
                  className:
                    'flex size-6 items-center justify-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                  onClick: () => setActivityToasts(!activityToasts),
                  children: jsx(Codicon, { name: activityToasts ? 'bell' : 'bell-slash' })
                })
              }),
              jsxs(DropdownMenu, {
                children: [
                  jsx(Tip, {
                    label: 'New…',
                    children: jsx(DropdownMenuTrigger, {
                      asChild: true,
                      children: jsx('button', {
                        type: 'button',
                        'aria-label': 'New bot or group chat',
                        className:
                          'flex size-6 items-center justify-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                        children: jsx(Codicon, { name: 'add' })
                      })
                    })
                  }),
                  jsxs(DropdownMenuContent, {
                    align: 'end',
                    children: [
                      jsxs(DropdownMenuItem, {
                        onSelect: () => setCreateOpen(true),
                        children: [jsx(Codicon, { name: 'hubot', className: 'mr-1.5' }), 'New Bot']
                      }),
                      jsxs(DropdownMenuItem, {
                        disabled: activeSourceRoster.length < 2,
                        onSelect: () => setGroupCreateOpen(true),
                        children: [jsx(Codicon, { name: 'organization', className: 'mr-1.5' }), 'New Group Chat']
                      })
                    ]
                  })
                ]
              })
            ]
          })
        ]
      }),
      jsx(ActiveNowStrip, {
        roster: visibleRoster,
        activeProfile,
        gatewayState,
        metaByName: allMeta,
        // Keep the Active Now strip and sidebar rows on the same exact-owner
        // route: source activation first, then canonical name-registry open.
        onOpen: bot => void openRosterBot(bot)
      }),
      showRosterTools
        ? jsx('div', {
            className: 'flex min-w-0 items-center gap-1 px-2.5 pb-1.5',
            children: [
              showRosterSearch
                ? jsx(
                    SearchField,
                    {
                      'aria-label': 'Search bots and group chats',
                      containerClassName: cn(
                        'min-w-0 flex-1',
                        query ? 'opacity-100!' : 'opacity-50 focus-within:opacity-100'
                      ),
                      inputClassName:
                        'w-full text-[0.75rem] placeholder:text-(--ui-text-tertiary)',
                      placeholder: 'Search bots and group chats…',
                      value: query,
                      onChange: setQuery
                    },
                    'roster-search'
                  )
                : jsx('span', { className: 'min-w-0 flex-1' }, 'roster-search-spacer'),
              showRosterFilters
                ? jsxs(
                    DropdownMenu,
                    {
                      children: [
                      jsx(Tip, {
                        label: activeFilterCount ? `Filters (${activeFilterCount} active)` : 'Filter roster',
                        children: jsx(DropdownMenuTrigger, {
                          asChild: true,
                          children: jsx('button', {
                            type: 'button',
                            'aria-label': activeFilterCount ? `Filter roster, ${activeFilterCount} active` : 'Filter roster',
                            className: cn(
                              'flex size-7 shrink-0 items-center justify-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                              activeFilterCount && 'text-(--ui-accent)'
                            ),
                            children: jsx(Codicon, { name: 'list-filter' })
                          })
                        })
                      }),
                      jsxs(DropdownMenuContent, {
                        align: 'end',
                        children: [
                          ...[
                            ['all', 'Bots and group chats'],
                            ['bots', 'Bots only'],
                            ['groups', 'Group chats only']
                          ].map(([value, label]) =>
                            jsxs(
                              DropdownMenuItem,
                              {
                                onSelect: () => setRowKindFilter(value),
                                children: [
                                  jsx('span', { className: 'min-w-0 flex-1', children: label }),
                                  rowKindFilter === value ? jsx(Codicon, { name: 'check' }) : null
                                ]
                              },
                              `kind:${value}`
                            )
                          ),
                          jsx(DropdownMenuSeparator, {}),
                          ...[
                            ['all', 'Any activity'],
                            ['active', 'Active now'],
                            ['recent', 'Recently active'],
                            ['older', 'Older']
                          ].map(([value, label]) =>
                            jsxs(
                              DropdownMenuItem,
                              {
                                onSelect: () => setActivityFilter(value),
                                children: [
                                  jsx('span', { className: 'min-w-0 flex-1', children: label }),
                                  activityFilter === value ? jsx(Codicon, { name: 'check' }) : null
                                ]
                              },
                              `activity:${value}`
                            )
                          ),
                          gatewayOptions.length > 1 ? jsx(DropdownMenuSeparator, {}) : null,
                          gatewayOptions.length > 1
                            ? jsxs(DropdownMenuItem, {
                                onSelect: () => setGatewayFilter('all'),
                                children: [
                                  jsx(Codicon, { name: 'globe', className: 'mr-1.5' }),
                                  jsx('span', { className: 'min-w-0 flex-1', children: 'All gateways' }),
                                  gatewayFilter === 'all' ? jsx(Codicon, { name: 'check' }) : null
                                ]
                              })
                            : null,
                          ...(gatewayOptions.length > 1
                            ? gatewayOptions.map(option => {
                                const status = botSourceStatus({
                                  sourceError: option.error,
                                  sourceReachable: option.reachable
                                })

                                return jsxs(
                                  DropdownMenuItem,
                                  {
                                    onSelect: () => setGatewayFilter(option.connectionId),
                                    children: [
                                      jsx(GatewayKindGlyph, {
                                        kind: option.kind,
                                        className: cn(
                                          'mr-1.5',
                                          !status.available && 'text-amber-600 dark:text-amber-300'
                                        )
                                      }),
                                      jsx('span', {
                                        className: 'min-w-0 flex-1 truncate',
                                        children: option.label || option.connectionId
                                      }),
                                      jsx('span', {
                                        className: 'text-[0.625rem] tabular-nums text-(--ui-text-quaternary)',
                                        children: option.count
                                      }),
                                      gatewayFilter === option.connectionId ? jsx(Codicon, { name: 'check' }) : null
                                    ]
                                  },
                                  option.connectionId
                                )
                              })
                            : []),
                          activeFilterCount ? jsx(DropdownMenuSeparator, {}) : null,
                          activeFilterCount
                            ? jsx(DropdownMenuItem, {
                                onSelect: () => {
                                  setRowKindFilter('all')
                                  setActivityFilter('all')
                                  setGatewayFilter('all')
                                },
                                children: 'Clear filters'
                              })
                            : null
                        ]
                      })
                      ]
                    },
                    'roster-filters'
                  )
                : null
            ]
          })
        : null,
      staleNotice
        ? jsx('div', {
            className: 'mx-2.5 mb-1 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
            children: staleNotice
          })
        : null,
      (isLoading || initialRosterLoading) && !roster.length
        ? jsx('div', {
            className: 'flex flex-1 items-center justify-center',
            children: jsx(GlyphSpinner, { spinner: 'breathe', className: 'text-(--ui-text-tertiary)' })
          })
        : error && !roster.length
          ? jsxs('div', {
              className: 'grid gap-2 px-3 py-4 text-xs text-(--ui-text-tertiary)',
              children: [
                jsx('div', {
                  children: gatewayUp
                    ? `Roster unavailable: ${error instanceof Error ? error.message : 'gateway error'}. If your gateway predates profiles.list, update Hermes and restart the gateway.`
                    : 'Waiting for the gateway connection… (remote gateways can take a few seconds; retries automatically)'
                }),
                jsx(Button, {
                  variant: 'secondary',
                  size: 'sm',
                  className: 'justify-self-start',
                  onClick: () => void refetch(),
                  children: 'Retry now'
                })
              ]
            })
          : roster.length === 0
            ? jsx(EmptyState, {
                icon: 'hubot',
                title: 'No bots yet',
                description: 'Create your first bot.'
              })
            : allBotsHidden && !hiddenExpanded
              ? jsxs('div', {
                  className: 'grid content-start gap-2 px-3 py-4 text-xs text-(--ui-text-tertiary)',
                  children: [
                    jsxs('div', {
                      className: 'flex items-center gap-1.5 font-medium text-(--ui-text-secondary)',
                      children: [
                        jsx(Codicon, { name: 'eye-closed', className: 'text-(--ui-text-quaternary)' }),
                        'All bots are hidden'
                      ]
                    }),
                    jsx('p', { className: 'leading-relaxed', children: 'They keep working and retain their history.' }),
                    jsx(Button, {
                      variant: 'secondary',
                      size: 'sm',
                      className: 'justify-self-start',
                      onClick: () => $showHiddenBots.set(true),
                      children: 'Show hidden bots'
                    })
                  ]
                })
              : rosterRows.length === 0 && matchingHiddenBots.length === 0
                ? jsx('div', {
                    'aria-live': 'polite',
                    className:
                      'flex flex-1 items-center justify-center px-4 text-center text-xs text-(--ui-text-tertiary)',
                    role: 'status',
                    children: query.trim()
                      ? `No bots or group chats match “${query.trim()}”${selectedGateway ? ` on ${selectedGateway.label}` : ''}`
                      : selectedGateway
                        ? `No bots or group chats match these filters on ${selectedGateway.label}`
                        : 'No bots or group chats match these filters.'
                  })
                : jsx(ScrollArea, {
                    className: 'hermes-bots-roster min-h-0 flex-1',
                    children: jsx('div', {
                      className: 'grid w-full min-w-0 gap-0.5 px-1.5 pb-2',
                      children: [
                        ...(showGatewaySections
                          ? [
                              sortedGroupRows.length ? renderGroupChatSection() : null,
                              ...gatewaySections.sections.map(renderGatewaySection)
                            ].filter(Boolean)
                          : rosterRows.map(row =>
                              row.kind === 'group' ? renderGroupRow(row) : renderBotRow(row.bot)
                            )),
                        showHiddenSection
                          ? jsxs(
                              'div',
                              {
                                ref: hiddenSectionRef,
                                className: 'mt-1 border-t border-(--ui-stroke-tertiary) pt-1',
                                children: [
                                hasRosterConstraint
                                  ? jsxs('div', {
                                      className:
                                        'flex w-full items-center gap-1 px-2 py-1.5 text-[0.6875rem] font-medium text-(--ui-text-tertiary)',
                                      children: [
                                        jsx(Codicon, { name: 'eye-closed' }),
                                        jsx('span', { children: 'Hidden' }),
                                        jsx('span', {
                                          className: 'text-(--ui-text-quaternary)',
                                          children: matchingHiddenBots.length
                                        })
                                      ]
                                    })
                                  : jsxs('button', {
                                      type: 'button',
                                      'aria-expanded': hiddenExpanded,
                                      className:
                                        'flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-left text-[0.6875rem] font-medium text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                                      onClick: () => $showHiddenBots.set(!hiddenExpanded),
                                      children: [
                                        jsx(Codicon, { name: hiddenExpanded ? 'chevron-down' : 'chevron-right' }),
                                        jsx('span', { children: 'Hidden' }),
                                        jsx('span', {
                                          className: 'text-(--ui-text-quaternary)',
                                          children: hiddenBots.length
                                        })
                                      ]
                                    }),
                                showHiddenRows
                                  ? matchingHiddenBots.length
                                    ? hiddenGatewaySections.sectioned
                                      ? hiddenGatewaySections.sections.map(renderHiddenGatewaySection)
                                      : matchingHiddenBots.map(bot => renderBotRow(bot, 'hidden:'))
                                    : jsx('div', {
                                        className: 'px-2 py-2 text-xs text-(--ui-text-quaternary)',
                                        children: 'No hidden bots match these filters.'
                                      })
                                  : null
                                ]
                              },
                              'hidden-section'
                            )
                          : null
                      ]
                    })
                  }),
      jsx(CreateAgentDialog, {
        open: createOpen,
        onClose: () => {
          setCreateOpen(false)
          void refetch()
        },
        roster: activeSourceRoster
      }),
      jsx(CreateGroupChatDialog, {
        open: groupCreateOpen,
        // Full multi-source roster: group chats can seat bots from other
        // registered connections — their turns route to their own machines.
        roster,
        onClose: () => setGroupCreateOpen(false),
        onCreated: groupName => openGroupChat(groupName)
      }),
      jsx(EditProfileDialog, {
        bot: editing,
        open: Boolean(editing),
        onClose: () => {
          setEditing(null)
          void refetch()
        }
      }),
      grouping ? jsx(GroupDialog, { bot: grouping, onClose: () => setGrouping(null) }) : null,
      jsx(ConfirmDialog, {
        open: Boolean(deleting),
        title: 'Delete bot and profile?',
        description: deleting
          ? jsxs('span', {
              children: [
                'This will permanently delete the bot ',
                jsx('span', { className: 'font-medium text-foreground', children: deleting.name }),
                ' and its associated Hermes profile at ',
                jsx('span', { className: 'font-mono text-xs', children: deleting.path }),
                '. This cannot be undone.'
              ]
            })
          : null,
        destructive: true,
        confirmLabel: 'Delete',
        busyLabel: 'Deleting…',
        doneLabel: 'Deleted',
        onClose: () => setDeleting(null),
        onConfirm: async () => {
          if (!deleting) {
            return
          }

          const name = deleting.name
          await deleteBot(deleting)
          await refetch()
          host.notify({ kind: 'success', message: `Deleted profile ${name}` })
        }
      }),
      jsx(ConfirmDialog, {
        open: Boolean(deletingGroup),
        title: 'Delete group chat?',
        description: deletingGroup
          ? `This removes “${deletingGroup.name}” from its bots and clears the shared room log. The bots and their individual chats are kept.`
          : null,
        destructive: true,
        confirmLabel: 'Delete Group',
        busyLabel: 'Deleting…',
        doneLabel: 'Deleted',
        onClose: () => setDeletingGroup(null),
        onConfirm: async () => {
          if (!deletingGroup) return
          await disbandGroupChat(deletingGroup.name, deletingGroup.members)
          host.notify({ kind: 'success', message: `Deleted group “${deletingGroup.name}”` })
        }
      })
    ]
  })
}

// ── plugin ───────────────────────────────────────────────────────────────────

export default {
  id: ID,
  name: 'Bots',
  description: 'Bot Mode — a one-chat-per-agent roster with avatars, routines, group chats, and bot-to-bot messaging. Ships with the app; disable here if unwanted.',
  register(ctx) {
    pluginCtx = ctx
    groupChatSyncDisposed = false
    startFaceClock()
    // The cross-connection relay rides every gateway socket this Desktop
    // holds: roster sync + envelope drain/deliver/reply loops.
    startBotRelay()
    // Disabling the plugin (or a hot reload) must actually stop the clock —
    // before this, the rAF loop + 1Hz document scan ran until app restart.
    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(stopFaceClock)
      ctx.onDispose(stopBotRelay)
    }

    // @-mention autocomplete: typing "@rese…" in ANY composer offers the
    // roster's handles (issue #88060). Reads the roster straight from the
    // query cache — useRoster keeps it ≤5s stale and the popover must answer
    // synchronously per keystroke. Multi-source rosters contribute their
    // precomputed @name-device handles via botHandle. The active profile is
    // excluded (a bot doesn't @ itself); 'default' surfaces as @hermes.
    ctx.register({
      id: 'mention-completions',
      area: COMPOSER_AREAS.atCompletions,
      data: {
        provide: query => {
          const roster = cachedUnionRoster()
          const profiles = Array.isArray(roster?.profiles) ? roster.profiles : []

          if (!profiles.length) {
            return []
          }

          const active = focusedMentionProfile()
          const q = (query || '').toLowerCase()
          const items = []
          const live = {
            name: active,
            connectionId: String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
          }

          for (const profile of profiles) {
            if (!profile?.name || isActiveRosterBot(profile, live)) {
              continue
            }

            const handle = botHandle(profile.name, profile)
            const display = displayName(profile, $botMeta.get()[profile.name])
            // Renamed bots complete on their friendly name — the tag is the
            // renamed slug when one exists, the profile handle otherwise.
            const tag = botMentionTag(profile)

            if (
              q &&
              !tag.toLowerCase().startsWith(q) &&
              !handle.toLowerCase().startsWith(q) &&
              !display.toLowerCase().startsWith(q)
            ) {
              continue
            }

            const source = profile.connectionLabel ? ` · ${profile.connectionLabel}` : ''

            items.push({
              insert: `@${tag}`,
              display: `@${tag}`,
              meta: `Bot · ${display}${source}`
            })
          }

          return items.slice(0, 8)
        }
      }
    })

    // Keyframes for the pet bob — injected because plugin classes aren't in
    // the app's precompiled CSS. Idempotent across hot reloads.
    if (!document.getElementById('hermes-bots-keyframes')) {
      const style = document.createElement('style')
      style.id = 'hermes-bots-keyframes'
      style.textContent = '@keyframes hermes-bots-bob { from { transform: translateY(0); } to { transform: translateY(-3px); } }'
      document.head.appendChild(style)
    }

    // Hydrate persisted avatars/titles. Migration writes v2 only after a
    // provable sole-local topology and deliberately leaves v1 untouched for
    // one-version rollback.
    void migrateBotMeta(ctx.storage).catch(() => undefined)

    // The last selected bot, source-qualified. Restoring it is PRESENTATION
    // ONLY: it paints the Bots home and the roster highlight, and never
    // activates a gateway, opens a chat, or creates a session. The hydrated
    // flag must flip on every settle path — the home holds a loading state
    // until it does, and a storage quirk must not strand it there.
    try {
      Promise.resolve(ctx.storage?.get?.('selected-roster-bot-v1'))
        .then(value => {
          if (typeof value === 'string' && value.trim()) {
            $selectedRosterKey.set(value.trim())
          }
        })
        .catch(() => undefined)
        .finally(() => $selectedRosterHydrated.set(true))
    } catch {
      /* no storage — this window starts with no restored selection */
      $selectedRosterHydrated.set(true)
    }

    // Bot Mode sessions are always hidden now — the old "hide Bot Chats"
    // pref is gone (its stored key is simply ignored). The reconciliation
    // sweep below hides any rows born visible under the old pref.

    // Hydrate the activity-toast pref (default OFF).
    try {
      Promise.resolve(ctx.storage?.get?.('activity-toasts'))
        .then(value => {
          if (typeof value === 'boolean') {
            $activityToasts.set(value)
          }
        })
        .catch(() => undefined)
    } catch {
      /* no storage — default (silent) stays */
    }

    // Hydrate persisted group-chat room logs (epoch/running are runtime-only
    // and always reset — a loop can't survive a window reload anyway).
    try {
      Promise.resolve(ctx.storage?.get?.('group-chats'))
        .then(async value => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            const rooms = {}

            for (const [name, room] of Object.entries(value)) {
              if (room && Array.isArray(room.log)) {
                rooms[name] = {
                  // Pre-thread entries get synthetic thread ids on hydrate so
                  // every UI/engine path can assume entry.thread exists.
                  log: assignLegacyThreads(room.log),
                  watermarks: room.watermarks && typeof room.watermarks === 'object' ? room.watermarks : {},
                  sessions: room.sessions && typeof room.sessions === 'object' ? room.sessions : {},
                  sessionOwners: room.sessionOwners && typeof room.sessionOwners === 'object' ? room.sessionOwners : {},
                  stranded: room.stranded && typeof room.stranded === 'object' ? room.stranded : {},
                  // #93129: rehydrate sticky stop holds with the same shape
                  // guard as the other maps — a held bot stays held across
                  // window restarts until explicitly released.
                  holds: room.holds && typeof room.holds === 'object' ? room.holds : {},
                  members: Array.isArray(room.members) ? room.members : [],
                  roomId: typeof room.roomId === 'string' && room.roomId ? room.roomId : null,
                  image: typeof room.image === 'string' && room.image ? room.image : null,
                  syncRevision: Math.max(0, Number(room.syncRevision || 0)),
                  epoch: 0,
                  running: false
                }
              }
            }

            $groupChats.set({ ...rooms, ...$groupChats.get() })

            // #93492: annotate rows orphaned before this build (their
            // connection was deleted while an older Desktop ran, so no
            // 'removed' push ever swept them). Registry read is
            // feature-detected; when unavailable only the unresolvable-route
            // shape (lost connectionId) is annotated.
            try {
              const registry =
                typeof window !== 'undefined'
                  ? await Promise.resolve(window.hermesDesktop?.connections?.list?.()).catch(() => null)
                  : null
              const liveIds = Array.isArray(registry?.connections)
                ? new Set(registry.connections.map(connection => String(connection?.id || '').trim()).filter(Boolean))
                : null
              const annotated = annotateOrphanedGroupChatMembers($groupChats.get(), liveIds)

              if (annotated.changed) {
                // Per-room updateGroupChat keeps the durable record's full
                // shape (sessionOwners, holds) in storage; sync:false —
                // the scheduleGroupChatServerSync below publishes once.
                for (const [roomName, room] of Object.entries(annotated.rooms)) {
                  if (room !== $groupChats.get()[roomName]) {
                    updateGroupChat(roomName, () => room, { sync: false })
                  }
                }
              }
            } catch {
              /* registry unavailable — the lost-connectionId shape is still safe to render */
            }
          }

          // Receive before publish. A fresh Desktop with no local room cache
          // must hydrate the gateway projection instead of merely avoiding an
          // empty overwrite and then rendering an empty conversation.
          await pullGroupChatServerState().catch(() => false)
          scheduleGroupChatServerSync($groupChats.get())
        })
        .catch(() => undefined)
    } catch {
      /* no storage — rooms start empty */
    }

    // Routines follow the chat you're in: track the focused chat's owner
    // profile (falls back to the live gateway profile on older desktops —
    // see $focusedBotProfile). Keying this off the socket's home alone left
    // the unread-suppression and Routines scope on the wrong bot whenever a
    // focused tab showed another profile's chat.
    // Capture the unbinds: without them a disable → re-enable cycle stacks a
    // duplicate listener per cycle (same survives-disable class as the face
    // clock before its onDispose hook — these kept firing until app restart).
    const unbindProfileListener = bindProfileSync($focusedBotOwner)
    const unbindGatewayListener = host.state.gateway.listen(handleSessionsGatewayTransition)

    // #93492 root fix: the registry pushes a lifecycle event when a
    // connection is removed. The gateway store already disposes the dead
    // sockets; the persisted group-chat rosters referencing that connection
    // were never touched, which is what left panes throwing "Bot X has no
    // connection owner" forever. Annotate (never silently delete) those
    // member rows the moment the connection goes away. Feature-detected:
    // older Electron mains don't emit it, and bare vm test harnesses have
    // no window global.
    let unbindConnectionsChanged = null
    try {
      if (typeof window !== 'undefined') {
        unbindConnectionsChanged =
          window.hermesDesktop?.connections?.onChanged?.(payload => {
            if (payload?.reason === 'removed') {
              sweepGroupChatMembersForRemovedConnection(payload.connectionId)
            }
          }) || null
      }
    } catch {
      /* registry lifecycle push unavailable — hydrate-time annotate still covers it */
    }

    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(() => {
        stopGroupChatServerSync()
        if (typeof unbindProfileListener === 'function') {
          unbindProfileListener()
        }
        if (typeof unbindGatewayListener === 'function') {
          unbindGatewayListener()
        }
        if (typeof unbindConnectionsChanged === 'function') {
          unbindConnectionsChanged()
        }
      })
    }

    // Reconciliation sweep: hide every Bot Mode session we know about, on
    // load and again on each reconnect (a swap can land on a gateway whose
    // rows were created before the always-hidden policy). Deferred a tick so
    // the meta/room storage hydrates above have landed; idempotent after that.
    // (Feature-guarded: bare vm test harnesses have no setTimeout global.)
    const scheduleHideSweep = () => {
      try {
        setTimeout(() => void hideOwnedBotSessions(), 0)
      } catch {
        void hideOwnedBotSessions()
      }
    }
    host.state.gateway.listen(state => {
      if (state === 'open') {
        scheduleHideSweep()
      }
    })
    scheduleHideSweep()

    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'Bots',
      // dock: explicit adoption gesture — CENTER-STACK into the sessions zone
      // so the sidebar grows a SESSIONS | BOTS tab strip instead of splitting
      // two cramped panes down the column. Center is safe now: insertAtGroup
      // pins the zone's header explicitly shown on a center gain (and it
      // stays shown once the zone has stacked), so the sessions pane can
      // never vanish behind a stripless Bots tab — the lone-pane auto-hide
      // trap this dock used to work around with a 'bottom' split.
      // enforce: standing invariant, not a one-shot migration — the pane
      // re-homes into the sessions strip at EVERY boot it isn't already
      // there, whatever tokens or user placement an older install persisted.
      // The one-time heal ('sessions-tab-v1') burned its token even when its
      // guards skipped the move, so exactly the users who had fought the old
      // stacked layout (dragged panes → $userPlacedPanes) stayed stacked
      // forever. Owner's order: SESSIONS | BOTS is always a tab strip.
      // An intra-session drag still sticks until the next launch (the
      // invariant runs at adoption time only — see enforceDockedPanes in the
      // tree store).
      // collapsible: the pane lives in the sessions zone, so it must LEAVE
      // the grid with that zone below the sidebar-collapse breakpoint. The
      // sessions pane collapses alone without this flag. The zone then keeps
      // a stranded BOTS tab on screen. The narrow edge overlay mirrors the
      // zone's tab strip, so the pane stays reachable while collapsed.
      data: { placement: 'left', width: '260px', collapsible: true, hideOnly: true, dock: { pane: 'sessions', pos: 'center', enforce: true } },
      render: () => jsx(BotsPane, {})
    })

    // Routines — its OWN tiling pane splitting the workspace's right edge
    // (NOT the collapsible right sidebar; placement 'right' is that sidebar's
    // role and hides the pane until "Show Right Sidebar").
    //
    // Registered ONLY while Bot Mode is on screen: the pane exists while the
    // Bots pane is visible (its zone's active tab, or a lone pane in a
    // stacked pre-heal layout) and unregisters when the user tabs back to
    // Sessions — no Cronjobs tile squatting beside the chat outside Bot Mode.
    // `ctx.register` returns the disposer that makes this cheap; the tree
    // keeps the pane's spot, so re-registering re-adopts it where it was.
    // host.paneVisibility is feature-detected: older desktops without the SDK
    // export keep the always-registered behavior.
    const registerRoutinesPane = () =>
      ctx.register({
        id: 'routines',
        area: 'panes',
        title: 'Cronjobs',
        data: {
          placement: 'main',
          // Repair persisted layouts that stranded Cronjobs in the Bots tab strip.
          dock: { pane: 'workspace', pos: 'right', enforce: true },
          width: '250px'
        },
        render: () => jsx(RoutinesPane, {})
      })

    if (typeof host.paneVisibility === 'function') {
      // The contribution-scoped pane id (`register` prefixes `${ID}:`).
      const $sidebarVisible = host.paneVisibility(`${ID}:pane`)
      let unregisterRoutines = null

      const syncRoutinesPane = () => {
        if (botChatOwnsWorkspace()) {
          unregisterRoutines ??= registerRoutinesPane()
        } else if (unregisterRoutines) {
          unregisterRoutines()
          unregisterRoutines = null
        }
      }

      // One recompute for both main-area surfaces: they answer the same
      // question (who owns the center) from the same three signals.
      const syncWorkspaceSurfaces = () => {
        syncBotsHomeWorkspace()
        syncRoutinesPane()
      }

      const stopSidebarSync = $sidebarVisible.listen(visible => {
        $botsPaneVisible.set(Boolean(visible))
        if (visible) {
          const group = $groupChatWorkspace.get()
          const selected = selectedRosterBot($lastRoster.get(), $selectedRosterKey.get())
          setBotsWorkspaceOwner(
            group ? groupWorkspaceOwnerKey(group) : selected ? botWorkspaceOwnerKey(selected) : BOTS_HOME_OWNER_KEY,
            group ? null : selected,
            group ? 'New group conversations start in the group composer.' : 'Select a Bot or group first.'
          )
        } else {
          // Strand any owner wake still dialing. Its SDK open will fail the
          // workspace token too; this plugin generation prevents that expected
          // cancellation from repainting Bots home or showing an error after
          // the user deliberately returned to Sessions.
          botOpenGeneration += 1
          host.setWorkspaceScope?.('sessions')
        }
        // A generic composer has no stored-session owner, so passive sync
        // replaces it with the Bot home. A real restored chat keeps the
        // center until the user explicitly selects a Bot owner.
        syncWorkspaceSurfaces()
      })
      const stopGroupSync = $groupChatWorkspace.listen(syncWorkspaceSurfaces)
      // The home tab's visibility flips are the ONLY signal for two real
      // transitions: layout hydration re-asserting a persisted active tab
      // over the home after boot, and the user swapping between the home tab
      // and a chat tab. React on the NEXT tick — the notification arrives
      // mid-layout-mutation, and registering/unregistering panes from inside
      // it would re-enter the tree store.
      const scheduleSurfaceSync = () => {
        try {
          setTimeout(syncWorkspaceSurfaces, 0)
        } catch {
          syncWorkspaceSurfaces()
        }
      }
      const homeVisibleStore = host.paneVisibility(BOTS_HOME_PANE_ID)
      const stopHomeVisibleSync = homeVisibleStore.listen(visible => {
        // Update selection ownership immediately; the deferred pass below may
        // mutate registrations, but the visible row must never lag a frame.
        $botsHomeFronted.set(Boolean(visible))
        scheduleSurfaceSync()
      })
      // Tab focus moves without swapping the gateway socket, so the focused
      // STORED session is the truth about session focus; older shells fall
      // back to the active session id. A RISING edge means a session just
      // claimed the center (opened or refocused): the home yields then — and
      // only then, so an explicitly selected owner can hold the center over
      // a focused-but-hidden chat without the next poll snatching it back.
      const focusStore = host.state.focusedStoredSessionId || host.state.activeSessionId
      const stopFocusSync =
        typeof focusStore?.listen === 'function'
          ? focusStore.listen(id => {
              $botChatFocused.set(Boolean(id))
              releaseStaleOpenBotChat(id)

              if (id) {
                closeBotsHomeWorkspace()
              }

              syncWorkspaceSurfaces()
            })
          : null

      // Proactive reclaim refresh: when the gateway reaps the runtime behind
      // the OPEN bot chat (idle TTL, LRU cap, WS-orphan reap — the mass-reap
      // shape hits every background bot at once), re-resume the canonical
      // chat immediately instead of letting the user's next send eat the
      // stale-id error + recovery retry. Matched on the STORED id (the
      // claim's ids are stored ids; the payload carries both). Best-effort:
      // a failed re-resume (backend still down) leaves the lazy recovery on
      // next send as the backstop. Feature-detected — older shells have no
      // host.onEvent.
      const stopReclaimSync =
        typeof host.onEvent === 'function'
          ? host.onEvent('session.reclaimed', event => {
              const payload = event?.payload || {}
              const stored = String(payload.stored_session_id || '')
              const claim = $openBotChat.get()

              if (!stored || !claim) {
                return
              }

              const owned = [claim.openedSessionId, claim.openedRegistryId].filter(Boolean)

              if (!owned.includes(stored)) {
                return
              }

              const bot = selectedRosterBot($lastRoster.get(), $selectedRosterKey.get())

              if (!bot) {
                return
              }

              const generation = botOpenGeneration
              void openBotCanonicalChat(bot)
                .then(opened => {
                  // A user action while the re-resume ran owns the center now.
                  if (!opened || generation !== botOpenGeneration) {
                    return
                  }

                  $openBotChat.set({
                    key: claim.key,
                    openedRegistryId: opened.registryId,
                    openedSessionId: opened.openedId
                  })
                })
                .catch(() => {
                  /* backend still down — next send recovers via the ladder */
                })
            })
          : null

      $botsPaneVisible.set(Boolean($sidebarVisible.get()))
      $botChatFocused.set(sessionOwnsWorkspace())
      $botsHomeFronted.set(Boolean(homeVisibleStore.get()))
      // A persisted layout can boot directly into Bot Mode while restoring
      // the generic Sessions workspace as the active sibling. Reconcile now,
      // then once more after the layout mutation finishes: the deferred pass
      // remains passive, so a real restored chat is never covered.
      syncWorkspaceSurfaces()
      scheduleSurfaceSync()

      if (typeof ctx.onDispose === 'function') {
        // The registration disposer is already tracked by ctx.register; only
        // the listeners need explicit teardown or they survive plugin disable.
        ctx.onDispose(() => {
          stopSidebarSync()
          stopGroupSync()
          stopHomeVisibleSync()
          stopFocusSync?.()
          stopReclaimSync?.()
          $botsHomeFronted.set(false)
          closeBotsHomeWorkspace()
        })
      }
    } else {
      registerRoutinesPane()
    }

    ctx.register({
      id: 'new-agent',
      area: PALETTE_AREA,
      data: {
        id: `${ID}.new-agent`,
        label: 'New Bot…',
        keywords: ['bot', 'agent', 'profile', 'teammate', 'create'],
        run: () => {
          host.notify({ kind: 'info', message: 'Open the Bots pane and hit “New Bot”.' })
        }
      }
    })

    // @-mention middleware: "@<bot> do the thing" in any chat gets an
    // IDENTIFICATION note — who the user is referring to, resolved against
    // the LIVE roster ("user@example.com" or an unknown @ passes through
    // untouched). The middleware never delivers anything itself: the agent
    // owns messaging via its message_agent tool (Bot Chats), so there is
    // exactly one send path and user text is never forwarded verbatim by
    // the renderer. The composer's @-autocomplete remains the picking aid.
    ctx.register({
      id: 'mention-middleware',
      area: COMPOSER_AREAS.middleware,
      data: {
        handler: async draft => {
          const text = draft.text || ''

          // /new inside a bot's canonical forever-chat would fork the
          // relationship into a scratch session — the one thing Bots mode
          // promises never happens. Reroute to /compact (same felt effect:
          // fresh working context, SAME conversation) and say so. Only
          // guards the canonical chat: Sessions-mode scratchpads on the
          // same profile keep full /new freedom.
          const slashNew = /^\/(new|reset)\s*$/.exec(text.trim())

          if (slashNew) {
            const activeBot = $selectedBot.get()
            // Canonical identity is the profile's "Bot Chat" registry row —
            // read it from the roster cache (canonical_session, resolved
            // server-side by name), matching either the durable row id or
            // the compression-lineage tip currently on screen.
            const roster = $lastRoster.get()
            const row = Array.isArray(roster) ? roster.find(bot => bot?.name === activeBot) : null
            const canonical = row?.canonical_session || null
            const currentId = host.activeSessionId?.get?.() ?? null
            const canonicalIds = [canonical?.id, canonical?.resolved_id].filter(Boolean).map(String)

            if (activeBot && currentId && canonicalIds.includes(String(currentId))) {
              host.notify({
                kind: 'info',
                title: 'This chat never resets',
                message:
                  'Bot chats are one continuous conversation — compacting instead. ' +
                  'For a throwaway session with this agent, use Sessions mode.'
              })

              return { ...draft, text: '/compact' }
            }
          }

          if (!/(^|\s)@[a-z0-9][a-z0-9_-]*/i.test(text)) {
            return draft
          }

          const live = {
            name: focusedMentionProfile(),
            connectionId: String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
          }
          const cached = cachedUnionRoster()
          const roster = Array.isArray(cached?.profiles) ? cached.profiles : null
          let mentionedBots = roster ? resolveRosterMentions(text, roster, live) : []

          if (!roster) {
            try {
              const res = await host.request('profiles.list', { include_sessions: false })
              // Same resolver as the cached path — renamed bots (display_name
              // / ui_meta title) stay taggable when the roster cache is cold.
              mentionedBots = resolveRosterMentions(text, res?.profiles ?? [], live).map(bot => ({ ...bot, remoteSource: false }))
            } catch {
              return draft
            }
          }

          if (!mentionedBots.length) {
            return draft
          }

          // Identification only. Each line names the agent the user's tag
          // resolves to (friendly title + device for cross-connection rows),
          // so the agent knows exactly who "@research-buddy" is. Cross-
          // connection targets carry the '@connection' suffix message_agent
          // resolves against the Desktop-synced relay roster.
          const lines = mentionedBots.map(bot => {
            const handle = botHandle(bot.name, bot)
            const title = String(botRosterMeta(bot, $botMeta.get())?.title || bot.ui_meta?.['hermes-bots']?.title || bot.title || '').trim()
            const target = bot.remoteSource && bot.connectionId ? `${handle}@${bot.connectionId}` : handle
            const where = bot.remoteSource
              ? ` — on ${bot.connectionLabel || bot.connectionId} (message_agent target: "${target}")`
              : ''
            return `@${handle} = agent profile "${bot.name}"${title ? ` ("${title}")` : ''}${where}`
          })
          const note =
            '\n\n[@mentions resolved from the Bot Mode roster — the user is referring to: ' +
            lines.join('; ') +
            '. If they want one of these agents contacted, compose your own message and send it with your message_agent tool (agents on other connected machines are reachable too — the Desktop relays it); never forward the user\u2019s text verbatim. If this session has no message_agent tool, agent messaging is unavailable here — say so.]'

          return { ...draft, text: text + note }
        }      }
    })
  }
}
