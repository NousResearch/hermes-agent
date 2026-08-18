/**
 * Hermes Bot Mode — a "one chat per agent" roster for the Hermes desktop.
 *
 * Left pane "Bots": one row per Hermes profile (a bot = an agent profile) with
 * a customizable avatar (shape + color + eyes, image, or pet). Click opens that
 * bot's chat; right-click → Edit Profile (avatar, title, description).
 * "New Agent" creates a profile — Name / Title / Description with an
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const { McpTab, ToolsetConfigPanel } = sdk
// Keep optional exports feature-detected; test harnesses may strip the SDK namespace.
const SkillsView = typeof sdk === 'undefined' ? undefined : sdk.SkillsView
const Streamdown = typeof sdk === 'undefined' ? undefined : sdk.Streamdown
// Budgeted render loop (fps cap + observability pause + dormancy + teardown).
// Feature-detected: older desktops fall back to the hand-rolled clock below.
const createBudgetedLoop = typeof sdk === 'undefined' ? undefined : sdk.createBudgetedLoop

const ID = 'hermes-bots'
const ROSTER_KEY = [ID, 'roster']
const ROUTINES_KEY = [ID, 'routines']
const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

/** Captured in register() so components can reach plugin storage. */
let pluginCtx = null

/** Live roster snapshot for imperative handlers (context menus). */
const $lastRoster = atom([])

/** Bots with chat activity the user hasn't seen yet (name -> true).
 *  Fed by the roster poll's activity watermark, so it catches EVERY
 *  delivery path: RPC, CLI (bot-to-bot), cron runs, other machines. */
const $botUnread = atom({})

// last_active watermark per bot, seeded on first poll so a fresh mount
// doesn't mark ancient history unread.
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
 *  the watermark for a bot whose chat isn't on screen -> unread + toast. */
function trackInboundActivity(roster) {
  const seeding = !watermarksSeeded
  watermarksSeeded = true

  for (const bot of roster) {
    const ts = bot.last_session?.last_active || 0
    const prev = rosterWatermarks.get(bot.name) || 0
    rosterWatermarks.set(bot.name, Math.max(prev, ts))

    if (seeding || ts <= prev) {
      continue
    }

    // Activity in the bot the user is currently looking at is already
    // visible — never badge the open chat.
    if ($selectedBot.get() === bot.name) {
      continue
    }

    $botUnread.set({ ...$botUnread.get(), [bot.name]: true })

    // Roster-hidden bots stay quiet: the unread flag above accumulates
    // silently (unhiding reveals the badge) but a hidden bot never toasts.
    if ($botMeta.get()[bot.name]?.hidden) {
      continue
    }

    // Toasts are opt-in: the unread badge is always set above, but the
    // per-message notification fires only when the user enabled it.
    if ($activityToasts.get()) {
      const meta = $botMeta.get()[bot.name]
      const label = displayName(bot, meta)
      const preview = (bot.last_session?.preview || '').trim()
      const inbound = /^Message from/i.test(preview)

      host.notify({
        kind: 'info',
        title: inbound ? `\uD83E\uDD16 New message for ${label}` : `${label} has new activity`,
        message: preview.slice(0, 140) || 'Open the chat to see it.'
      })
    }
  }
}

/** Last good cron list, same idea as the roster snapshot. */
const $lastJobs = atom([])

// Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar:
// canonical Bot Chats are plugin-owned forever-chats and group-chat member
// sessions are room plumbing — neither is a scratch conversation, and a
// 6-member room would otherwise dump six identical "Group: ..." rows into
// recents. Backed by the core generic `hidden` session flag (session.create
// hidden:true / session.set_hidden); the Bots pane browses them via
// session.list include_hidden. Older gateways ignore the flag and the
// sessions simply stay visible there.

/** Bot the Routines tile is scoped to. Follows the live gateway profile
 *  (the bot you're actually chatting with) and roster clicks. */
const $selectedBot = atom('default')

/** Optional secondary navigation inside the Bots pane. Primary row clicks still
 * open the bot's canonical chat; this state opens its stored-session browser. */
const $botSessionsWorkspace = atom(null)
const $botSelectedSessions = atom({})
const $sessionsGatewayGeneration = atom(0)

/** Group-chat rooms: { [group]: { log: [{from:{kind,name},text,at}], watermarks:{[member]:idx}, epoch, running } }.
 *  Log + watermarks persist via plugin storage; epoch/running are runtime-only. */
const $groupChats = atom({})
/** Group whose room view is open in the Bots pane (secondary navigation,
 *  same pattern as $botSessionsWorkspace). */
const $groupChatWorkspace = atom(null)
/** Groups whose latest room activity mentions @user — the needs-you badge. */
const $groupNeedsYou = atom({})

function handleSessionsGatewayTransition() {
  $sessionsGatewayGeneration.set($sessionsGatewayGeneration.get() + 1)
  $botSelectedSessions.set({})
  // A gateway swap invalidates any in-flight room drive: bump every room's
  // epoch so running loops bail at their next member boundary.
  const rooms = { ...$groupChats.get() }

  for (const name of Object.keys(rooms)) {
    rooms[name] = { ...rooms[name], epoch: (rooms[name].epoch || 0) + 1, running: false }
  }

  $groupChats.set(rooms)
}

/** Per-bot appearance + display meta, persisted via ctx.storage:
 *  { [botName]: { shape, color, title } } */
const $botMeta = atom({})

async function saveBotMeta(name, patch) {
  const prevMeta = $botMeta.get()[name] || {}
  const next = { ...$botMeta.get(), [name]: { ...prevMeta, ...patch } }
  $botMeta.set(next)

  // Local plugin storage: instant, and the fallback for older gateways.
  try {
    Promise.resolve(pluginCtx?.storage?.set?.('bot-meta', next)).catch(() => undefined)
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
    const { image, pet, ...rest } = next[name] || {}
    serverRequest = Promise.resolve(host.request('profiles.configure', { name, ui_meta: { 'hermes-bots': rest } }))
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
        ? host.request('profiles.set_asset', { name, asset: 'avatar', data: patch.image })
        : host.request('profiles.set_asset', { name, asset: 'avatar', clear: true })
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
  }

  return { serverPersisted: serverOutcome === 'persisted', serverOutcome }
}

// ── hidden bots (right-click → Hide Bot) ────────────────────────────────────
// Hiding is a ROSTER-DISPLAY concern only: a hidden bot keeps working —
// @mentions still resolve, group-chat membership is untouched, its name
// still counts as taken, and an open chat stays open. The flag lives in bot
// meta (`hidden: true`), so it rides the same local-storage + server
// ui_meta pipeline as pins/titles and follows the profile across machines.
// Unhide writes `hidden: false` (never null): a null key survives the local
// `{ ...prev, ...patch }` merge while the server DELETES None keys, and
// that asymmetry lets mergeServerMeta resurrect a stale truthy copy. A
// literal false round-trips identically through both stores.

/** Session-only view toggle: reveal hidden bots (dimmed) in the roster. */
const $showHiddenBots = atom(false)

/** Hidden flag for a roster row. Thin remote-source rows never read local
 *  meta (botRosterMeta returns null for them), so hide is by NAME on the
 *  active source; remote rows of the same name stay visible. */
function isBotHidden(bot, metaByName) {
  return Boolean(botRosterMeta(bot, metaByName)?.hidden)
}

/** Hiding the selected bot re-homes the selection (the Routines pane
 *  follows it): first visible bot wins, then 'default' — unless default is
 *  itself hidden with nothing else visible, in which case the selection
 *  stays put rather than pointing somewhere even less real. */
function fallbackSelectionAfterHide(name) {
  if ($selectedBot.get() !== name) {
    return
  }

  const meta = $botMeta.get()
  const visible = $lastRoster
    .get()
    .filter(bot => !bot.remoteSource && bot.name !== name && !meta[bot.name]?.hidden)

  if (visible.length) {
    $selectedBot.set(visible[0].name)
    return
  }

  if (name !== 'default' && !meta.default?.hidden) {
    $selectedBot.set('default')
  }
}

/** One-time reconciliation: Bot Mode sessions are always hidden, but rooms
 *  and Bot Chats created before this policy (or while the old pref was off)
 *  left visible rows behind. On every plugin load, sweep every session id we
 *  own — canonical chats from bot meta plus each group room's member
 *  sessions — through the core session.set_hidden RPC, then run the
 *  ownership-based sweep for the rows we DON'T know by id. Idempotent (the DB
 *  setter is a no-op on already-hidden rows) and feature-detected: older
 *  gateways lack session.set_hidden and simply keep the rows visible. */
function hideOwnedBotSessions() {
  const canonical = Object.values($botMeta.get())
    .map(m => m && m.chat)
    .filter(Boolean)
  const rooms = Object.values($groupChats.get())
    .flatMap(room => Object.values(room?.sessions || {}))
    .filter(sid => Boolean(sid) && sid !== true)
  const ids = [...new Set([...canonical, ...rooms])]

  const known = Promise.all(
    ids.map(sid =>
      Promise.resolve(host.request('session.set_hidden', { session_id: sid, hidden: true })).catch(() => undefined)
    )
  )

  return Promise.all([known, sweepBotProfileSessions().catch(() => undefined)])
}

// Titles Bot Mode itself mints for its plumbing sessions. Bot-to-bot CLI
// handoffs (`hermes -p <bot> chat --in ~ -c "Bot Chat" --create-if-missing`)
// and mention handoffs create sessions with EXACTLY these titles; the
// "Group: " prefix is the member-session title ensureGroupChatSession has
// used since group chats shipped. Exact/prefix matching is deliberate — a
// user's real conversation inside a bot profile keeps whatever title the
// user gave it and is never touched.
const BOT_MODE_SWEEP_TITLES = new Set(['Bot Chat', 'Agent Inbox'])

function isBotModeSweepTitle(title) {
  const t = String(title || '').trim()
  return BOT_MODE_SWEEP_TITLES.has(t) || t.startsWith('Group: ')
}

/** Ownership-based sweep: the id-based sweep above only covers sessions the
 *  plugin recorded ($botMeta canonical chats, $groupChats member sids), but
 *  Bot Mode sessions are ALSO minted outside the plugin — bot-to-bot CLI
 *  handoffs ("Agent Inbox" / extra "Bot Chat" rows born visible in a bot's
 *  profile) — and those ids the plugin never learns. So: enumerate each
 *  roster bot's OWN profile sessions (only bot profiles — a non-bot profile
 *  is never listed, so its sessions are never touched) and hide any VISIBLE
 *  row whose title is Bot Mode plumbing. session.list without include_hidden
 *  returns only visible rows, which keeps the sweep naturally idempotent.
 *  Remote-source bots route to their own connection via requestForBot.
 *  Feature-detected + fire-and-forget: older gateways without per-profile
 *  session.list / session.set_hidden simply reject and the sweep no-ops. */
async function sweepBotProfileSessions() {
  const cached = $lastRoster.get()
  let roster = Array.isArray(cached) && cached.length ? cached : null

  if (!roster) {
    // Plugin load can run before the Bots pane hydrates $lastRoster — fall
    // back to the active gateway's own profile list (local bots; remote
    // sources get covered by the next sweep once the roster cache exists).
    try {
      const res = await host.request('profiles.list', {})
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
            .filter(row => row && row.id && isBotModeSweepTitle(row.title))
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
    if (bot.has_avatar || avatarPushInflight.has(bot.name)) {
      continue
    }

    const image = $botMeta.get()[bot.name]?.image

    if (image && typeof image === 'string' && image.startsWith('data:')) {
      avatarPushInflight.add(bot.name)
      host
        .request('profiles.set_asset', { name: bot.name, asset: 'avatar', data: image })
        .then(() => queryClient.invalidateQueries({ queryKey: ['hermes-bots', 'roster'] }))
        .catch(() => avatarPushInflight.delete(bot.name))
      continue
    }

    // Vector shape/color face: no image exists anywhere — rasterize the
    // live SVG (tagged data-bot-face) to a PNG and push that, so the
    // inter-agent notices (core #85855/#85888) can show the real pfp.
    const svg = document.querySelector('svg[data-bot-face=' + JSON.stringify(bot.name) + ']')

    if (!svg) {
      continue
    }

    avatarPushInflight.add(bot.name)
    rasterizeSvgToPng(svg, 160)
      .then(png =>
        png
          ? host
              .request('profiles.set_asset', { name: bot.name, asset: 'avatar', data: png })
              .then(() => queryClient.invalidateQueries({ queryKey: ['hermes-bots', 'roster'] }))
          : Promise.reject(new Error('rasterize failed'))
      )
      .catch(() => avatarPushInflight.delete(bot.name))
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
    if (!bot.has_avatar || avatarFetchInflight.has(bot.name)) {
      continue
    }

    if ($botMeta.get()[bot.name]?.image) {
      continue
    }

    avatarFetchInflight.add(bot.name)
    host
      .request('profiles.get_asset', { name: bot.name, asset: 'avatar' })
      .then(res => {
        if (res?.found && res.data) {
          const current = $botMeta.get()
          const mine = current[bot.name] || {}
          // A 160px raster of the vector face is only for inter-agent
          // notices. Do not park it on the roster or the live face dies.
          if (isBackfilledFacePng(res.data) && mine.imageKind !== 'photo' && !mine.pet) {
            return
          }
          $botMeta.set({ ...current, [bot.name]: { ...mine, image: res.data } })

          try {
            Promise.resolve(pluginCtx?.storage?.set?.('bot-meta', $botMeta.get())).catch(() => undefined)
          } catch {
            /* no storage */
          }
        }
      })
      .catch(() => undefined)
      .finally(() => avatarFetchInflight.delete(bot.name))
  }
}

/** Server ui_meta (per roster row) beats local storage for the compact
 *  fields it carries; local-only fields (avatar image data URL, extracted
 *  pet icon) are PRESERVED — the server copy never includes them, so a
 *  naive replace would wipe a just-saved image avatar on the next roster
 *  paint. When server bot metadata exists, an omitted chat is authoritative
 *  deletion; local still fills all gaps for older gateways with no metadata. */
function mergeServerMeta(roster) {
  const local = $botMeta.get()
  let changed = false
  const next = { ...local }

  for (const bot of roster) {
    const server = bot.ui_meta?.['hermes-bots']
    if (server && typeof server === 'object') {
      const mine = next[bot.name] || {}
      const merged = { ...mine, ...server }

      // Local-only fields survive the server overlay.
      if (mine.image) {
        merged.image = mine.image
      }

      // Server metadata is authoritative for the canonical chat pointer.
      // Without this deletion sync, ctx.storage resurrects stale sessions
      // after the server pin is cleared and even after a full app restart.
      if (
        Object.prototype.hasOwnProperty.call(mine, 'chat') &&
        !Object.prototype.hasOwnProperty.call(server, 'chat')
      ) {
        delete merged.chat
      }

      if (JSON.stringify(next[bot.name] || null) !== JSON.stringify(merged)) {
        next[bot.name] = merged
        changed = true
      }
    }
  }

  if (changed) {
    $botMeta.set(next)

    // Persist server reconciliation so a relaunch cannot rehydrate stale
    // local fields that the server intentionally removed.
    try {
      Promise.resolve(pluginCtx?.storage?.set?.('bot-meta', next)).catch(() => undefined)
    } catch {
      /* storage unavailable — reconciliation lasts for this window only */
    }
  }
}

/** Clone a bot: profile (config/skills/SOUL/memory via clone_from) + look.
 *  Name is "<base>-2", "-3", … — first free slot against the live roster. */
async function duplicateBot(bot, roster) {
  const base = bot.name
  let name = null
  for (let n = 2; n < 100; n++) {
    // Truncate the BASE, never the suffix — slicing the joined string chops
    // the "-2" off a max-length name and the candidate collides with the
    // base forever (#19).
    const suffix = `-${n}`
    const candidate = base.slice(0, 64 - suffix.length) + suffix
    if (!roster.some(b => b.name === candidate)) {
      name = candidate
      break
    }
  }

  if (!name) {
    throw new Error('No free name for the duplicate.')
  }

  await host.request('profiles.create', {
    name,
    clone_from: base,
    description: bot.description || ''
  })

  // Same look: avatar shape/color/image and a "(copy)" title so the two
  // are tellable apart in the roster until the user renames. Do not copy
  // chat or created. Those belong to the original bot.
  const meta = $botMeta.get()[base]
  if (meta) {
    const { chat, created, ...look } = meta
    saveBotMeta(name, {
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
  if (typeof host.deleteProfile === 'function') {
    await host.deleteProfile(bot.name)
  } else {
    // Older desktop without the SDK verb — best effort via the CLI.
    const result = await host.request('cli.exec', {
      argv: ['profile', 'delete', bot.name, '--yes']
    })

    if (result?.blocked || result?.code !== 0) {
      throw new Error(result?.hint || result?.output || `Could not delete profile ${bot.name}.`)
    }
  }

  const meta = { ...$botMeta.get() }
  delete meta[bot.name]
  $botMeta.set(meta)

  try {
    await Promise.resolve(pluginCtx?.storage?.set?.('bot-meta', meta))
  } catch {
    /* profile is deleted; stale local appearance is harmless if storage fails */
  }

  const unread = { ...$botUnread.get() }
  delete unread[bot.name]
  $botUnread.set(unread)
  rosterWatermarks.delete(bot.name)
  avatarFetchInflight.delete(bot.name)
  avatarPushInflight.delete(bot.name)

  if ($selectedBot.get() === bot.name) {
    $selectedBot.set('default')
  }

  queryClient.invalidateQueries({ queryKey: ROSTER_KEY })

  if (host.state.profile.get?.() === bot.name && typeof host.newChat === 'function') {
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
  // profile may be null at first (New Agent: the profile isn't created yet).
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

  // Resolve the target profile, creating it on demand for the New Agent flow.
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

/** Shape grid + color swatches, shared by Edit Profile and New Agent.
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
        ? jsxs('div', {
            className: 'grid justify-items-center gap-3',
            children: [
              jsx('div', {
                style: {
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
                  gap: '6px',
                  justifyItems: 'center'
                },
                children: AVATAR_PICKER_SHAPES.map(s =>
                  jsx(
                    'button',
                    {
                      type: 'button',
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

/** Pins to resolve precisely on the next roster poll: {profile: chatId}.
 *  The backend answers "what about THIS conversation" per entry
 *  (preferred_session), so a row's preview can describe the same session its
 *  click opens (hermes-agent#88200). Unknown params are ignored by older
 *  gateways, which simply omit the field. */
function preferredSessionIds(allMeta) {
  const pins = {}
  for (const [name, meta] of Object.entries(allMeta || {})) {
    if (meta?.chat) {
      pins[name] = meta.chat
    }
  }
  return pins
}

function useRoster() {
  const activeConnectionId = useValue(host.state.connectionId)

  return useQuery({
    queryKey: [...ROSTER_KEY, activeConnectionId],
    queryFn: async () => {
      // Rich rows (last_session, ui_meta, has_avatar) come from the ACTIVE
      // gateway's profiles.list — unchanged single-source behavior.
      const pins = preferredSessionIds($botMeta.get())
      const local = await host.request(
        'profiles.list',
        Object.keys(pins).length ? { preferred_session_ids: pins } : {}
      )
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
          return mergeMultiSourceRoster(local, union, activeConnectionId, $lastRoster.get())
        } catch {
          /* older build or roster failure — single-source list stands */
        }
      }

      return local
    },
    refetchInterval: 5000,
    staleTime: 5000,
    // Remote (SSH) gateways connect slowly and drop on sleep/wake; keep
    // retrying instead of latching a terminal error card.
    retry: true,
    retryDelay: attempt => Math.min(15000, 1000 * 2 ** attempt)
  })
}

/** Merge the union agent roster (host.agents) over the active gateway's
 *  profiles.list. Active-source rows — matched by the LIVE connection id,
 *  falling back to the roster's primaryConnectionId, then the legacy
 *  kind==='local' rule on older desktops — are the agents profiles.list
 *  already returned: they only ANNOTATE the rich rows (handle, connection
 *  fields); rich fields stay authoritative and they are NOT duplicated.
 *  Rows from other sources become new roster entries tagged with their
 *  source label so BotRow can badge them and route open/warm through
 *  ensureAgent/warmAgent. Pure — exercised directly by the tests. */
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
  const activeId = liveId || (liveProvided ? '' : String(union?.primaryConnectionId || '').trim())
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

function isActiveRosterBot(bot, active) {
  const activeName = String(active?.name || 'default').trim() || 'default'
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

const REMOTE_DM_TIMEOUT_MS = 180000
const REMOTE_DM_POLL_MS = 2000

/** The remote bot's canonical Bot Chat: pinned stored-id from its profile's
 *  ui_meta first, then resume-by-title, then create. Mirrors
 *  ensureGroupChatSession so DMs land in the ONE forever-chat instead of
 *  minting a fresh "Bot Chat" per mention. */
async function ensureRemoteCanonicalChat(route, profile) {
  let pinned = null

  try {
    const listed = await host.requestProfile(route, 'profiles.list', {})
    const owner = listed?.profiles?.find(p => p.name === profile)
    pinned = owner?.ui_meta?.['hermes-bots']?.chat || null
  } catch {
    /* older remote gateway — title lookup below still works */
  }

  for (const target of [pinned, 'Bot Chat']) {
    if (!target) {
      continue
    }

    try {
      const res = await host.requestProfile(route, 'session.resume', {
        session_id: target,
        profile,
        omit_messages: true
      })

      if (res?.session_id) {
        return { runtime: res.session_id, stored: res.session_key || pinned }
      }
    } catch {
      /* fall through */
    }
  }

  const created = await host.requestProfile(route, 'session.create', {
    profile,
    title: 'Bot Chat',
    // Bot Mode sessions are always hidden from the global sidebar.
    hidden: true
  })

  return { runtime: created?.session_id || null, stored: created?.stored_session_id || null }
}

/** Bounded reply poll on the recipient's session — same shape as a group
 *  member turn: wait for a NEW assistant message after `before`, or time out. */
async function pollRemoteDmReply(route, profile, sessionRef, before) {
  const deadline = Date.now() + REMOTE_DM_TIMEOUT_MS

  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, REMOTE_DM_POLL_MS))

    let state = null

    try {
      state = await host.requestProfile(route, 'session.resume', { session_id: sessionRef, profile })
    } catch {
      continue
    }

    const messages = Array.isArray(state?.messages) ? state.messages : []
    const done = !state?.inflight && !state?.running

    if (messages.length > before && done) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]

        if (msg?.role === 'assistant') {
          const text = typeof msg.content === 'string'
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content.map(p => (typeof p === 'string' ? p : p?.text || '')).join('')
              : msg?.text || ''

          return String(text).trim() || null
        }
      }

      return null
    }
  }

  return null
}

/** Deliver a user mention to bots on OTHER connections: into each bot's
 *  canonical Bot Chat, with the standard sender-attribution prefix (so the
 *  recipient's messaging protocol recognizes an agent-to-agent message), then
 *  relay the reply back as a notification. Sequential and fire-and-forget
 *  from the composer's perspective. */
async function deliverRemoteRosterMentions(bots, userText, sender) {
  const text = String(userText || '').trim()

  if (!text || typeof host.requestProfile !== 'function') {
    return
  }

  const senderName = String(sender?.name || 'the user').trim()
  const senderHandle = String(sender?.handle || senderName).trim()

  for (const bot of bots) {
    const connectionId = String(bot?.connectionId || '').trim()
    const profile = String(bot?.name || '').trim() || 'default'

    if (!connectionId || connectionId === 'local') {
      continue
    }

    const route = { connectionId, mode: 'remote', profile, targetProfile: profile }
    const label = bot.connectionLabel || connectionId

    try {
      const { runtime, stored } = await ensureRemoteCanonicalChat(route, profile)

      if (!runtime) {
        throw new Error('No remote session')
      }

      // Baseline before our submit, so the poll can spot the NEW reply.
      let before = 0

      try {
        const pre = await host.requestProfile(route, 'session.resume', { session_id: stored || runtime, profile })
        before = Array.isArray(pre?.messages) ? pre.messages.length : pre?.message_count || 0
      } catch {
        /* lazy session — zero messages */
      }

      // The delivery prefix is the recipient's cue that an agent (not its
      // human) is talking — same contract as the local CLI handoff.
      await host.requestProfile(route, 'prompt.submit', {
        session_id: runtime,
        text: `Message from \u{1F916} ${senderName} (@${senderHandle}): ${text}`
      })
      host.notify?.({
        kind: 'info',
        title: displayName(bot),
        message: `Messaged @${botHandle(profile, bot)} on ${label} — will relay the reply here.`
      })

      const reply = await pollRemoteDmReply(route, profile, stored || runtime, before)

      if (reply) {
        host.notify?.({
          kind: 'info',
          title: `\u{1F916} ${displayName(bot)} (${label})`,
          message: reply.slice(0, 500)
        })
      } else {
        host.notify?.({
          kind: 'info',
          title: displayName(bot),
          message: `No reply from @${botHandle(profile, bot)} yet — check its Bot Chat on ${label}.`
        })
      }
    } catch (error) {
      host.notifyError?.(error, `Could not reach ${label}`)
    }
  }
}

/** Source-qualified identity for a roster row — the React list key AND the
 *  cross-surface roster identity. Names alone are NOT unique in a
 *  multi-source roster (two connections can both expose 'default');
 *  duplicate keys make React reconciliation repeat whole blocks of the list
 *  on every poll repaint (the Aug 2026 dupe-bots smear). */
function botRosterKey(bot) {
  return `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`
}

// ── cross-connection routing ─────────────────────────────────────────────────
// A bot from another registered connection (remoteSource rows) is reached
// through host.requestProfile with a route descriptor; local bots keep the
// active-gateway door. Feature-detected: older desktops without
// requestProfile simply have no remote routes (callers fall back / disable).

/** Route descriptor for a bot on another connection, or null for the local /
 *  active source (or when the desktop can't route). */
function botConnectionRoute(bot) {
  const id = String(bot?.connectionId || '').trim()

  if (!bot?.remoteSource || !id || id === 'local' || typeof host.requestProfile !== 'function') {
    return null
  }

  const profile = String(bot?.name || '').trim() || 'default'

  return { connectionId: id, mode: 'remote', profile, targetProfile: profile }
}

/** Gateway RPC on the bot's OWN source: requestProfile for remote rows,
 *  the active gateway for local ones. Never activates/foregrounds. */
async function requestForBot(bot, method, params = {}) {
  const route = botConnectionRoute(bot)

  if (route) {
    return host.requestProfile(route, method, params)
  }

  return host.request(method, params)
}

/** Stable per-member identity inside a group room. Local members keep their
 *  bare name (compat with rooms persisted before cross-connection groups);
 *  remote members get the source-qualified key so `dixie` on the Mini and a
 *  local `dixie` never share watermarks or sessions. */
function groupMemberKey(member) {
  return member?.remoteSource ? botRosterKey(member) : member?.name
}

// Bot metadata is scoped to the active gateway until the server exposes a
// union of rich profile rows. Never paint that metadata onto a thin row from
// another source: two `default` agents must not borrow each other's title,
// pin, avatar, group, unread state, or canonical-chat pointer.
function botRosterMeta(bot, metaByName) {
  return bot?.remoteSource ? null : metaByName?.[bot?.name]
}

function showsHandle(name, meta, bot) {
  const display = displayName({ name }, meta)
  return Boolean(name && display.toLowerCase() !== botHandle(name, bot).toLowerCase())
}

// ── canonical bot chat ───────────────────────────────────────────────────────
// Each bot has ONE forever chat, pinned by stored-session id in bot meta
// (meta.chat — synced server-side via ui_meta, so it follows the profile).
// Opening a bot ALWAYS lands there: never "most recent session", which
// drifts whenever the profile is used from the CLI, Sessions mode, or a
// cronjob. The pin only changes through explicit adoption:
//   - grandfather: first open of a bot that already has history pins its
//     current latest session, so continuity starts from the chat in use
//   - fresh bot: opens a draft; when the first message persists a stored
//     session, we adopt that id (empty sessions are pruned server-side, so
//     pre-creating one at enable time is not possible)
//   - recovery: if the pinned id vanishes from the DB (compaction rewrote
//     the lineage), re-pin the newest session carrying the canonical title.

// In-flight creations, keyed by bot name — double-clicking a row must not
// mint two canonical chats.
const canonicalCreations = new Map()

/** Create the bot's ONE forever chat: a real session opened with a kickoff
 *  message (the gateway prunes zero-message sessions, so the chat is born
 *  with the bot introducing itself). Pins the stored id in bot meta and
 *  returns it. */
function createCanonicalChat(name) {
  const inflight = canonicalCreations.get(name)

  if (inflight) {
    return inflight
  }

  const run = (async () => {
    const res = await host.request('session.create', {
      profile: name,
      title: 'Bot Chat',
      // Always born hidden from the global sidebar — Bot Mode sessions are
      // plugin-owned. Core applies this via the generic `hidden` flag
      // (deferred as pending_hidden until the row exists); older gateways
      // ignore the unknown param and it stays visible.
      hidden: true
    })
    const sid = res?.stored_session_id
    const runtime = res?.session_id

    if (sid) {
      saveBotMeta(name, { chat: sid })
    }

    // Mount the session view FIRST, then send the kickoff — submitting into
    // an unmounted session left the intro reply invisible until reopen.
    let opened = false

    if (sid && typeof host.openSession === 'function') {
      try {
        await host.openSession(sid, { profile: name })
        opened = true
      } catch {
        // The stored row may not exist until the kickoff persists it. Retry
        // after prompt.submit below instead of leaving the chat off-screen.
      }
    }

    if (runtime) {
      await new Promise(resolve => window.setTimeout(resolve, 400))

      try {
        await host.request('prompt.submit', { session_id: runtime, text: 'Hey, tell me about yourself!' })

        if (!opened && sid && typeof host.openSession === 'function') {
          await host.openSession(sid, { profile: name })
        }
      } catch {
        // The chat already exists. Keep the pin so the next click
        // opens it instead of making a second Bot Chat.
      }
    }

    return sid || null
  })().finally(() => canonicalCreations.delete(name))

  canonicalCreations.set(name, run)

  return run
}

/** Open the bot's ONE forever chat and return the opened id (or the pin).
 *
 *  Identity rules (hermes-agent#88200 — the row must open the session its
 *  preview describes):
 *  - grandfather: no pin + existing history adopts the previewed session
 *    (`history`, the roster's last_session for this bot) instead of minting
 *    a new empty chat;
 *  - a live pin is verified through the backend's precise preferred_session
 *    resolver (hidden rows still resolve; compression lineages resolve to
 *    the live tip) — never inferred from a paginated, hidden-excluding
 *    session.list window, which misjudged real hidden pins as gone;
 *  - transient lookup failures keep the pin: try the stored id as-is, and
 *    only a rejected open enters recovery. */
async function openBotCanonicalChat(name, pinned, history) {
  if (!pinned) {
    // Grandfather: adopt the conversation the row already previews.
    const adoptId = history?.id
    if (adoptId && typeof host.openSession === 'function') {
      try {
        await host.openSession(adoptId, { profile: name })
        saveBotMeta(name, { chat: adoptId })
        return adoptId
      } catch {
        // Adoption raced a vanishing session — fall through to creation.
      }
    }
    return createCanonicalChat(name)
  }

  // Precise verification. An older gateway ignores the unknown param and
  // omits the key — that reads as a lookup failure below, NOT as a missing
  // session, so legacy backends keep the try-as-is escape hatch.
  let preferred
  let lookupFailed = false
  try {
    const res = await host.request('profiles.list', {
      include_sessions: true,
      preferred_session_ids: { [name]: pinned }
    })
    const row = (res?.profiles ?? []).find(p => p.name === name)
    preferred = row?.preferred_session
    if (preferred === undefined) {
      lookupFailed = true
    }
  } catch {
    lookupFailed = true
  }

  if (lookupFailed) {
    // Transient gateway state (or an older backend): the pin is innocent
    // until proven guilty — try it as-is, and only a rejected open clears.
    try {
      await host.openSession(pinned, { profile: name })
      return pinned
    } catch {
      saveBotMeta(name, { chat: null })
      return createCanonicalChat(name)
    }
  }

  if (preferred) {
    try {
      await host.openSession(preferred.resolved_id || preferred.id, { profile: name })
      return pinned
    } catch (error) {
      // The precise lookup JUST confirmed this session exists, so a failed
      // open is transient (reconnect, backend restart). Clearing the pin or
      // minting a replacement here would fork the bot's forever-chat on
      // every hiccup — report and keep everything as it is.
      host.notifyError?.(error, `Could not open ${name}'s chat — try again`)
      return pinned
    }
  }

  // Definitively gone (db reset, or the lineage was rewritten past
  // recovery): re-anchor on the previewed session when there is one.
  const recoveryId = history?.id
  if (recoveryId && typeof host.openSession === 'function') {
    try {
      await host.openSession(recoveryId, { profile: name })
      saveBotMeta(name, { chat: recoveryId })
      return recoveryId
    } catch {
      // Fall through to a fresh chat.
    }
  }
  saveBotMeta(name, { chat: null })
  return createCanonicalChat(name)
}

async function prepareBotSource(bot, pinnedChat) {
  if (!bot.sourceScoped) {
    return pinnedChat
  }

  if (typeof host.ensureAgent !== 'function') {
    throw new Error('Update Hermes Desktop to chat with agents on other connections.')
  }

  await host.ensureAgent(bot.connectionId, bot.name)

  if (!bot.remoteSource) {
    return pinnedChat
  }

  const liveId = String(typeof host.activeConnectionId === 'function' ? host.activeConnectionId() || '' : '').trim()
  const targetId = String(bot.connectionId || '').trim()

  if (targetId && targetId !== 'local' && liveId !== targetId) {
    throw new Error(`Still on ${liveId || 'this device'}, not ${bot.connectionLabel || targetId}`)
  }

  // Thin rows deliberately omit metadata from the active source. Once their
  // owner is active, recover that source's canonical-chat pointer so
  // same-named agents never reuse or overwrite each other's pin.
  try {
    const refreshed = await host.request('profiles.list', {})
    const owner = refreshed?.profiles?.find(profile => profile.name === bot.name)

    return owner?.ui_meta?.['hermes-bots']?.chat || null
  } catch {
    // Metadata refresh is best-effort; canonical creation remains the fallback.
    return null
  }
}

function displayName(bot, meta) {
  // Only THIN rows from another source trade the friendly name for their
  // connection label — the active gateway's own default must keep reading
  // "Hermes". Annotated active rows carry sourceScoped too, and keying this
  // off sourceScoped renamed the user's main agent to an IP-derived label
  // (community report, Aug 17 2026).
  if (bot?.remoteSource && (bot.name || '').trim().toLowerCase() === 'default' && bot.connectionLabel) {
    return bot.connectionLabel
  }

  if (meta?.title?.trim()) {
    return meta.title.trim()
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
    const display = displayName(bot, botRosterMeta(bot, metaByName)).toLowerCase()
    const profile = (bot.name || '').toLowerCase()
    const handle = botHandle(bot.name, bot).toLowerCase()
    // Multi-source rows also match on their device name ("homelab" finds
    // every bot living on the Homelab connection).
    const sourceLabel = (bot.connectionLabel || '').toLowerCase()
    return (
      display.includes(needle) || profile.includes(needle) || handle.includes(needle) || sourceLabel.includes(needle)
    )
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

/** Group chats that should hold a roster row: every group named in bot meta
 *  (local members) plus every room record that still has stored members or
 *  log — cross-connection rooms whose members can't ride bot-meta. */
function groupChatNames(metaByName, rooms) {
  const names = new Set(knownGroups(metaByName))

  for (const [name, room] of Object.entries(rooms || {})) {
    if ((Array.isArray(room?.members) && room.members.length) || (Array.isArray(room?.log) && room.log.length)) {
      names.add(name)
    }
  }

  return [...names]
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
    bot => !bot.remoteSource && (botRosterMeta(bot, metaByName)?.group || '').trim() === group
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
    remote.push((roster || []).find(bot => botRosterKey(bot) === key) || descriptor)
  }

  return [...local, ...remote]
}

/** Existing group names, alphabetical — feeds the Move-to-group dialog. */
function knownGroups(metaByName) {
  const names = new Set()

  for (const meta of Object.values(metaByName || {})) {
    const group = (meta?.group || '').trim()

    if (group) {
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

/** Transcript form of a room speaker's profile name. The primary profile is
 *  literally named "default" — render it as Hermes (matching displayName and
 *  the @hermes handle) so the main agent never loses its name in rooms. */
function groupSpeakerLabel(name) {
  return (name || '').trim().toLowerCase() === 'default' ? 'Hermes' : name
}

/** Room-log line as a member sees it: `Name (user): …` / `Name: …` /
 *  `Name (you): …`. */
function formatGroupChatLine(entry, viewerName) {
  if (entry.from.kind === 'user') {
    return `${entry.from.name || 'User'} (user): ${entry.text}`
  }

  const suffix = entry.from.name === viewerName ? ' (you)' : ''
  // Cross-connection speakers carry their device so same-named agents on
  // two machines stay tellable apart in every member's transcript.
  const source = entry.from.source ? ` [${entry.from.source}]` : ''

  return `${groupSpeakerLabel(entry.from.name)}${suffix}${source}: ${entry.text}`
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
    '- Reply with ONE short conversational message (1-3 sentences) ONLY if you have something new worth adding: build on what was just said, claim or hand off work, answer a question aimed at you, or report a real result.',
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
function updateGroupChat(group, mutate) {
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
      durable[name] = {
        log: room.log,
        watermarks: room.watermarks,
        sessions: room.sessions || {},
        // Cross-connection member descriptors — remote bots can't ride
        // bot-meta, so the room record carries who they are.
        members: Array.isArray(room.members) ? room.members : []
      }
    }

    Promise.resolve(pluginCtx?.storage?.set?.('group-chats', durable)).catch(() => undefined)
  } catch {
    /* storage unavailable — room survives for this window only */
  }

  return next
}

/** Soft-disband a group chat: clear every member's group assignment (the
 *  grouping is bot-meta, so the disband syncs cross-machine via ui_meta),
 *  drop the room log from the atom + plugin storage, and close the room view
 *  if it's open. The members' per-group gateway sessions ("Group: <name>")
 *  are intentionally KEPT — they stay reachable through each bot's session
 *  browser. */
async function disbandGroupChat(group, memberNames) {
  // Invalidate any in-flight round-robin FIRST: bump the epoch so a running
  // drive bails at its next member boundary instead of appending to a room
  // the user just discarded.
  const all = { ...$groupChats.get() }
  const prior = all[group] || {}

  delete all[group]
  // Keep a runtime-only tombstone while a drive may still be mid-turn; it
  // carries no log and is never persisted, so it can't rehydrate.
  if (prior.running) {
    all[group] = { log: [], watermarks: {}, sessions: {}, epoch: (prior.epoch || 0) + 1, running: false }
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

  // Persist the room map WITHOUT the disbanded room so it can't come back
  // on the next window load.
  try {
    const durable = {}

    for (const [name, room] of Object.entries($groupChats.get())) {
      if (name !== group && Array.isArray(room.log)) {
        durable[name] = { log: room.log, watermarks: room.watermarks, sessions: room.sessions || {} }
      }
    }

    await Promise.resolve(pluginCtx?.storage?.set?.('group-chats', durable))
  } catch {
    /* storage unavailable — the atom reset above still empties the room */
  }

  // Ungroup the members last. saveBotMeta never throws (local storage +
  // best-effort profiles.configure per member), so a flaky gateway can't
  // strand the disband halfway with the room log already gone.
  for (const name of memberNames) {
    await saveBotMeta(name, { group: null })
  }
}

function appendGroupChatEntry(group, from, text) {
  const entry = { from, text: String(text).trim(), at: Date.now() }

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

/** Ensure the member's per-group session exists and return a LIVE runtime
 *  session id for it. Gateway-native: session.create mints the session
 *  (lazy until its first message), session.resume by stored id — or by
 *  title, which also covers rehydrated rooms whose sid was lost — reopens
 *  it after restarts. Cross-connection members route to their OWN source
 *  via requestForBot; the window's gateway never switches. */
async function ensureGroupChatSession(group, member) {
  const title = `Group: ${group}`
  const room = $groupChats.get()[group] || {}
  const key = groupMemberKey(member)
  const known = room.sessions && room.sessions[key]

  // Try resuming what we know (stored sid first, then title lookup).
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
        return { runtime: res.session_id, stored: res.session_key || known }
      }
    } catch {
      /* fall through to create */
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
      return r
    })
  }

  return { runtime: created?.session_id || null, stored }
}

const GROUP_TURN_TIMEOUT_MS = 180000
const GROUP_TURN_POLL_MS = 2000

/** One member turn, gateway-native: submit the room delta as a prompt into
 *  the member's per-group session, then poll the session until a NEW
 *  assistant message lands (or timeout → pass). No shell composition. */
async function runGroupChatMemberTurn(group, member, prompt) {
  const { runtime, stored } = await ensureGroupChatSession(group, member)

  if (!runtime) {
    return null
  }

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

  await requestForBot(member, 'prompt.submit', { session_id: runtime, text: prompt })

  const deadline = Date.now() + GROUP_TURN_TIMEOUT_MS

  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, GROUP_TURN_POLL_MS))

    let state = null

    try {
      state = await requestForBot(member, 'session.resume', {
        session_id: stored || runtime,
        profile: member.name
      })
    } catch {
      continue
    }

    const messages = Array.isArray(state?.messages) ? state.messages : []
    const done = !state?.inflight && !state?.running

    if (messages.length > before && done) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]

        if (msg?.role === 'assistant') {
          const text = typeof msg.content === 'string'
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content.map(p => (typeof p === 'string' ? p : p?.text || '')).join('')
              : msg?.text || ''

          return String(text).trim()
        }
      }

      return null
    }
  }

  return null // timeout — reads as a pass
}

/** Drive one bounded round-robin room turn. Serial — one member at a time.
 *  A newer user send bumps the room epoch; this loop notices at the next
 *  member boundary, bails, and the newest send's own loop takes over. */
async function runGroupChatRounds(group, members) {
  const startEpoch = ($groupChats.get()[group] || {}).epoch || 0
  const isCurrent = () => (($groupChats.get()[group] || {}).epoch || 0) === startEpoch
  let posted = 0

  try {
    for (let round = 0; round < GROUP_CHAT_MAX_ROUNDS; round++) {
      const roomLog = ($groupChats.get()[group] || {}).log || []
      const responders = rotateGroupSpeakers(resolveGroupResponders(roomLog, members), round)
      let spokeThisRound = 0

      for (const member of responders) {
        if (!isCurrent() || posted >= GROUP_CHAT_MAX_MESSAGES) {
          return
        }

        const room = $groupChats.get()[group] || { log: [], watermarks: {} }
        const memberKey = groupMemberKey(member)
        const seen = room.watermarks[memberKey] || 0
        const delta = room.log.slice(seen)

        if (!delta.length) {
          continue
        }

        const prompt = buildGroupChatTurnPrompt({
          groupName: group,
          members,
          viewer: member,
          deltaLines: delta.slice(-GROUP_CHAT_HISTORY_LIMIT).map(e => formatGroupChatLine(e, member.name))
        })

        let reply = null

        try {
          reply = await runGroupChatMemberTurn(group, member, prompt)
        } catch {
          reply = null // a failed turn is a pass, never a room error
        }

        // The member has now seen everything up to the pre-reply log length.
        updateGroupChat(group, r => {
          r.watermarks[memberKey] = r.log.length
          return r
        })

        if (reply !== null && !isGroupPassText(reply)) {
          appendGroupChatEntry(
            group,
            { kind: 'member', name: member.name, ...(member.remoteSource ? { source: member.connectionLabel || member.connectionId } : {}) },
            reply
          )
          // Its own message counts as seen too.
          updateGroupChat(group, r => {
            r.watermarks[memberKey] = r.log.length
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
      updateGroupChat(group, r => {
        r.running = false
        return r
      })
    }
  }
}

/** User send into a group room: append, bump epoch (supersedes any running
 *  loop at its next member boundary), and start the room turn unless one is
 *  already running under the new epoch semantics. */
function sendToGroupChat(group, members, text) {
  const trimmed = String(text || '').trim()

  if (!trimmed || !members.length) {
    return
  }

  $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: false })
  appendGroupChatEntry(group, { kind: 'user', name: 'You' }, trimmed)

  const wasRunning = ($groupChats.get()[group] || {}).running === true

  updateGroupChat(group, room => {
    room.epoch = (room.epoch || 0) + 1
    room.running = true
    return room
  })

  if (!wasRunning) {
    void runGroupChatRounds(group, members).catch(() => {
      updateGroupChat(group, r => {
        r.running = false
        return r
      })
    })
  } else {
    // A loop is live; it bails at its next boundary. Chain the fresh loop
    // after a short settle so exactly one drive owns the room.
    setTimeout(() => {
      void runGroupChatRounds(group, members).catch(() => {
        updateGroupChat(group, r => {
          r.running = false
          return r
        })
      })
    }, 250)
  }
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
    return { fromBot: (match[1] || match[2] || '').trim().toLowerCase() || null }
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

/** Bots that are working right now: the profile the gateway is running a
 *  turn for (busy), plus any bot whose last message landed inside the
 *  liveness window. Pure — output follows the input roster's order, so
 *  presence never reorders or hides the normal list. */
function activeBots(roster, activeProfile, gatewayState, now = Date.now()) {
  return (roster || []).filter(bot => {
    const busyTurn = !bot.remoteSource && bot.name === activeProfile && gatewayState === 'busy'
    const last = bot.last_session?.last_active || 0
    const inWindow = Boolean(last && now / 1000 - last < ACTIVE_WINDOW_S)

    return busyTurn || inWindow
  })
}

// ── bot row ──────────────────────────────────────────────────────────────────

function BotRow({ bot, onDelete, onEdit, onGroup }) {
  const activeProfile = useValue(host.state.profile)
  const meta = botRosterMeta(bot, useValue($botMeta))
  const last = bot.last_session
  const isActive = !bot.remoteSource && bot.name === activeProfile
  const { shape, color, image } = botAppearance(bot.name, meta)
  // Keep user photos/pets. Drop the 160px SVG backfill so the math face can move.
  const photo = Boolean(image && !isBackfilledFacePng(image))
  const gatewayState = useValue(host.state.gateway)
  const activeNow = Boolean(last?.last_active && Date.now() / 1000 - last.last_active < ACTIVE_WINDOW_S)
  // Work pose only when this bot is actually doing something: the active
  // profile while the gateway is busy, or a bot that wrote within the
  // liveness window. Not every bot whenever the gateway is busy.
  const botMood = (isActive && gatewayState === 'busy') || activeNow ? 'work' : 'idle'
  // Subscribe on every render. A source switch turns the same keyed row from
  // thin to rich; conditionally calling useValue here breaks React hook order.
  const unreadByName = useValue($botUnread)
  const unread = !bot.remoteSource && Boolean(unreadByName[bot.name])
  // WHO sent the last message (bot-to-bot DM vs human) — the full stored
  // history lives in the Sessions workspace (context menu), not inline.
  // Preview identity must match click identity (#88200): when the backend
  // resolved the pinned canonical chat, preview THAT session — not the
  // profile's most recent (but unrelated) activity. Liveness checks above
  // keep last_session semantics: any recent activity means the bot is alive.
  const previewSession = bot.preferred_session || last
  const { fromBot } = previewKind(previewSession?.preview)
  // DM previews read like DMs: strip the delivery prefix, keep the message.
  const displayPreview = stripPreviewMarkdown(
    fromBot
      ? (previewSession?.preview || '').replace(A2A_PREFIX_RE, '').trim() || '…'
      : previewSession?.preview || bot.description || 'No conversations yet — say hi'
  )

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

  const open = async () => {
    haptic('tap')
    $selectedBot.set(bot.name)

    if (bot.remoteSource) {
      const handle = botHandle(bot.name, bot)
      host.notify?.({
        kind: 'info',
        title: displayName(bot),
        message: `Stay in this chat and @${handle} to message them. Gateway stays on this device.`
      })
      return
    }

    let pinnedChat = meta?.chat

    if (!bot.remoteSource && $botUnread.get()[bot.name]) {
      const next = { ...$botUnread.get() }
      delete next[bot.name]
      $botUnread.set(next)
    }

    // Activate the owner first so every canonical-chat RPC lands on the
    // backend that owns this bot's state database.
    try {
      pinnedChat = await prepareBotSource(bot, pinnedChat)
    } catch (error) {
      host.notifyError?.(error, `Could not reach ${bot.connectionLabel || 'the remote source'}`)

      return
    }

    try {
      const id = await openBotCanonicalChat(bot.name, pinnedChat, bot.last_session)

      if (id) {
        return
      }
    } catch {
      // Fall through to the older-gateway draft below.
    }

    if (typeof host.newChat === 'function') {
      // Older gateway without profile-scoped session.create — plain draft.
      host.newChat(bot.name)
    } else {
      host.navigate('/')
    }
  }

  const row = jsxs('button', {
    type: 'button',
    onPointerEnter: warm,
    onClick: open,
    className: cn(
      'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
      'hover:bg-(--chrome-action-hover)',
      isActive && 'bg-(--chrome-action-hover)',
      // Hidden bots only render while the header eye toggle is on — dimmed,
      // so the temporary reveal reads as a different state from the roster.
      meta?.hidden && 'opacity-60'
    ),
    children: [
      jsx('div', {
        className: 'shrink-0',
        children: jsx(BotFace, { shape, color, image: photo ? image : null, size: 34, name: bot.name, mood: botMood })
      }),
      jsxs('div', {
        className: 'min-w-0 flex-1',
        children: [
          jsxs('div', {
            className: 'flex items-baseline justify-between gap-2',
            children: [
              jsxs('div', {
                className: 'flex min-w-0 items-baseline gap-1.5 truncate',
                children: [
                  meta?.pinned
                    ? jsx('span', {
                        className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                        title: 'Pinned',
                        children: '📌'
                      })
                    : null,
                  meta?.hidden
                    ? jsx(Codicon, {
                        name: 'eye-closed',
                        className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                        title: 'Hidden from the roster'
                      })
                    : null,
                  jsx('span', {
                    className: 'truncate text-[0.8125rem] font-medium',
                    children: displayName(bot, meta)
                  }),
                  showsHandle(bot.name, meta, bot)
                    ? jsx('span', {
                        className: 'shrink-0 font-mono text-[0.6875rem] text-(--ui-text-quaternary)',
                        children: `@${botHandle(bot.name, bot)}`
                      })
                    : null,
                  bot.remoteSource
                    ? jsx('span', {
                        className:
                          'shrink-0 rounded bg-(--chrome-action-hover) px-1 font-mono text-[0.625rem] text-(--ui-text-tertiary)',
                        title: `Lives on ${bot.connectionLabel}`,
                        children: bot.connectionLabel
                      })
                    : null
                ]
              }),
              unread
                ? jsx('span', {
                    className: 'size-2 shrink-0 rounded-full bg-(--ui-accent,#4f9cf9)',
                    'aria-label': 'unread'
                  })
                : null,
              activeNow
                ? jsx('span', {
                    className: 'hermes-bots-pulse size-1.5 shrink-0 rounded-full bg-(--ui-accent,#4f9cf9)',
                    title: 'Active in the last 90s'
                  })
                : null,
              last
                ? jsx('span', {
                    className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                    children: relativeTime(last.last_active * 1000)
                  })
                : null
            ]
          }),
          jsxs('div', {
            className: 'flex min-w-0 items-center gap-1',
            children: [
              jsx('div', {
                className: fromBot
                  ? 'min-w-0 truncate text-xs italic text-(--ui-accent,#4f9cf9)'
                  : 'min-w-0 truncate text-xs text-(--ui-text-tertiary)',
                children: displayPreview
              }),
              fromBot
                ? jsxs('span', {
                    className:
                      'flex shrink-0 items-center gap-1 rounded-full bg-(--chrome-action-hover) px-1.5 py-px text-[0.625rem] font-medium text-(--ui-accent,#4f9cf9)',
                    title: `Last message came from @${fromBot} (bot-to-bot)`,
                    children: ['🤖', `@${fromBot}`]
                  })
                : null
            ]
          })
        ]
      })
    ]
  })

  // Thin rows from another source are navigation targets only. Their profile
  // metadata is not loaded yet, so edit/delete/pin/group actions would mutate
  // whichever backend happens to be active. A normal click activates the
  // owner; the refreshed rich row then exposes the full context menu.
  if (bot.remoteSource) {
    return row
  }

  return jsxs(ContextMenu, {
    children: [
      jsx(ContextMenuTrigger, { asChild: true, children: row }),
      jsxs(ContextMenuContent, {
        children: [
          jsx(ContextMenuItem, {
            onSelect: () => {
              const pinned = Boolean($botMeta.get()[bot.name]?.pinned)
              saveBotMeta(bot.name, { pinned: !pinned })
              host.notify({
                kind: 'info',
                message: `${displayName(bot, meta)} ${pinned ? 'unpinned' : 'pinned to top'}`
              })
            },
            children: meta?.pinned ? 'Unpin' : 'Pin to top'
          }),
          jsx(ContextMenuItem, {
            onSelect: () => {
              const hidden = Boolean($botMeta.get()[bot.name]?.hidden)
              // `hidden: false` (not null) so unhide round-trips through the
              // server ui_meta merge the same way the local merge sees it.
              saveBotMeta(bot.name, { hidden: !hidden })

              if (!hidden) {
                fallbackSelectionAfterHide(bot.name)
              }

              host.notify({
                kind: 'info',
                message: hidden
                  ? `${displayName(bot, meta)} is back in the roster`
                  : `${displayName(bot, meta)} hidden — use the eye button in the Bots header to see hidden bots`
              })
            },
            children: meta?.hidden ? 'Unhide Bot' : 'Hide Bot'
          }),
          jsx(ContextMenuSeparator, {}),
          jsx(ContextMenuItem, {
            onSelect: () => openBotSessionsWorkspace(bot),
            children: 'Sessions'
          }),
          jsx(ContextMenuItem, { onSelect: () => onEdit(bot), children: 'Edit Profile' }),
          jsx(ContextMenuItem, {
            onSelect: () => onGroup(bot),
            children: meta?.group ? `Group: ${meta.group}…` : 'Move to group…'
          }),
          jsx(ContextMenuItem, {
            onSelect: () => {
              host.notify({ kind: 'info', message: `Duplicating ${displayName(bot, meta)}…` })
              duplicateBot(bot, $lastRoster.get().filter(candidate => !candidate.remoteSource))
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
              $selectedBot.set(bot.name)

              if (typeof host.newChat === 'function') {
                host.newChat(bot.name)
              }
            },
            children: 'New chat with this agent'
          }),
          bot.is_default ? null : jsx(ContextMenuSeparator, {}),
          bot.is_default
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

function useModelOptions() {
  return useQuery({
    queryKey: [ID, 'model-options'],
    queryFn: () => host.request('model.options', { include_unconfigured: true, explicit_only: false, refresh: true }),
    staleTime: 120000,
    retry: false
  })
}

/**
 * Provider + model dropdowns from the gateway's configured inventory — the
 * same data the core model picker shows. `value = {provider, model}`;
 * onChange receives the merged patch.
 */
function ModelPicker({ value, onChange, placeholderModel = 'gateway default' }) {
  const { data, isLoading, error } = useModelOptions()

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
// Shared by Edit Profile and New Agent (edit mode only for skills/toolsets —
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

  if (!loaded) {
    setLoaded(true)
    Promise.all([
      host.request('profiles.describe', { name: bot }),
      host.request('mcp.catalog', { profile: bot }).catch(() => null)
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
  if (SkillsView) {
    return jsxs('div', {
      className: 'grid gap-4',
      children: [
        jsx(ModelPicker, {
          value: { provider: state.provider, model: state.model },
          onChange: patch => setState(prev => ({ ...prev, dirtyModel: true, ...patch }))
        }),
        labeled(
          'Capabilities (applies immediately — skills, tools, MCP)',
          jsx('div', {
            className: 'overflow-hidden rounded-md border border-(--ui-stroke-secondary)',
            style: { height: 460, minHeight: 300, resize: 'vertical', overflow: 'auto' },
            children: jsx(SkillsView, { embedded: true, fixedProfile: bot })
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

  return jsxs('div', {
    className: 'grid gap-4',
    children: [
      jsx(ModelPicker, {
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
              forProfile: bot,
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
                            children: jsx(ToolsetConfigPanel, { toolset: tset.name, profile: bot })
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
                children: jsx(McpTab, { gateway: host.getGateway(), profile: bot })
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
                                      profile: bot,
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
  const payload = { name: bot }
  const applied = {}

  if (state.dirtySoul) {
    payload.soul = ensureMessagingProtocol(state.soul, bot, $lastRoster.get())
  }

  if (state.dirtyModel) {
    const model = state.model.trim()
    const provider = state.provider.trim()

    if (model && provider) {
      payload.model = model
      payload.provider = provider
    } else if (!model && !provider) {
      try {
        const result = await host.request('cli.exec', {
          argv: ['--profile', bot, 'config', 'unset', 'model']
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

  const result = await host.request('profiles.configure', payload)
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
  const meta = bot ? metaAll[bot.name] : null
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
  const currentKey = bot ? `${bot.name}:${open}` : null
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
    const persistence = await saveBotMeta(bot.name, {
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
        await host.request('cli.exec', {
          argv: ['profile', 'describe', bot.name, '--text', desc]
        })
        queryClient.invalidateQueries({ queryKey: ROSTER_KEY })
      } catch (err) {
        host.notifyError(err, 'Saved look locally; description update failed')
      }
    }

    if (adv.loaded && (adv.dirtyModel || adv.dirtySoul || adv.dirtySkills || adv.dirtyToolsets || adv.dirtyMcp)) {
      try {
        const res = await applyAdvancedConfig(bot.name, adv)
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
                  children: jsx(AdvancedProfileConfig, { bot: bot.name, state: adv, setState: setAdv })
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
  const [shape, setShape] = useState('circle')
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
      .then(value => setConnections(Array.isArray(value) ? value : []))
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
    setShape('circle')
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
          ? `Agent "${displayName({ name: slug, title })}" created on ${targetLabel}`
          : `Agent "${displayName({ name: slug, title })}" created`
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
            jsx(DialogTitle, { children: 'New Agent' }),
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
                      // Capabilities tab only exists for the active gateway.
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
                      // the ACTIVE gateway's backend — a remote-target draft
                      // lives elsewhere, so it keeps the staged checklists
                      // (their catalog reads already route to the target).
                      children: (SkillsView && !remoteTarget
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
                                : 'Name the agent first — a draft profile is created when you open this tab (discarded if you cancel).'
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
                                // to the just-created profile. Writes land
                                // immediately — no staging needed.
                                children: jsx(SkillsView, { embedded: true, fixedProfile: createdForCaps })
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
              children: busy ? 'Creating…' : 'Create Agent'
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

async function loadRoutines(profile) {
  // profile scopes cron.manage to that bot's own cron store (core RPC gained an
  // optional `profile` param). Older gateways ignore the unknown param and
  // return the launch-profile store — the [bot:] tag filter in selectRoutineJobs
  // remains the graceful fallback there.
  const scope = profile ? { profile } : {}
  const data = await host.request('cron.manage', { action: 'list', include_disabled: true, ...scope })
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
      host
        .request('cron.manage', { action: 'pause', name: job.job_id, ...scope })
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

function useRoutines(profile) {
  return useQuery({
    queryKey: [...ROUTINES_KEY, profile || ''],
    queryFn: () => loadRoutines(profile),
    refetchInterval: 20000,
    staleTime: 8000
  })
}

function routineCreateTarget(owner, activeBot) {
  return owner || activeBot
}

async function invalidateRoutineOwner(profile) {
  await queryClient.invalidateQueries({
    queryKey: [...ROUTINES_KEY, profile || ''],
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

/** Escape for interpolation INSIDE an existing double-quoted shell string:
 *  keeps ", `, $, and \ literal so free-text titles (which sync from ui_meta)
 *  and gateway profile names can't expand or break out of the quotes. */
function shellDoubleQuote(value) {
  return String(value).replace(/[\\"`$]/g, ch => '\\' + ch)
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

function RoutineRow({ job, profile }) {
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
      await host.request('cron.manage', { action, name: job.job_id, ...(profile ? { profile } : {}) })
      await invalidateRoutineOwner(profile)
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
          jsx('span', {
            'aria-hidden': true,
            className: cn('size-1.5 shrink-0 rounded-full', active ? 'bg-emerald-500' : 'bg-(--ui-text-quaternary)')
          }),
          jsx('span', {
            className: cn('min-w-0 flex-1 truncate text-xs font-medium', !active && 'text-(--ui-text-tertiary)'),
            children: routineTitle(job)
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
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const activeProfile = useValue(host.state.profile)
  const schedule = composeSchedule(sched)

  const reset = () => {
    setName('')
    setInstruction('')
    setSched(defaultScheduleState())
    setContinuity(false)
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
      await host.request('cron.manage', {
        action: 'add',
        name: `[bot:${bot}] ${title}`,
        schedule: schedule.trim(),
        prompt: routinePrompt(bot, title, task, activeProfile),
        ...(bot ? { profile: bot } : {}),
        ...(repeatN ? { repeat: repeatN } : {}),
        ...(continuity ? { continuity: true } : {})
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
              children: `A recurring task ${displayName({ name: bot }, $botMeta.get()[bot])} runs on a schedule. Runs land in its own chat history.`
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

function RoutinesPane() {
  const selected = useValue($selectedBot)
  const gatewayProfile = useValue(host.state.profile)
  // The tile maps to the bot you're chatting with: the live gateway profile
  // is the truth once a chat opens; $selectedBot covers the gap between a
  // roster click and the profile swap landing.
  const bot = (gatewayProfile || selected || 'default').trim() || 'default'
  const meta = useValue($botMeta)[bot]
  const { shape, color, image } = botAppearance(bot, meta)
  const { data, error, isLoading, refetch } = useRoutines(bot)
  const [createOpen, setCreateOpen] = useState(false)
  const [createOwner, setCreateOwner] = useState(null)
  const createTarget = routineCreateTarget(createOwner, bot)

  const openCreate = () => {
    setCreateOwner(bot)
    setCreateOpen(true)
  }

  const view = selectRoutineJobs(data, error, $lastJobs.get(), bot)
  if (view.live) {
    $lastJobs.set(view.live)
  }
  const jobs = view.jobs
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
                jsx(Codicon, { name: 'calendar', className: 'text-[1.6rem] text-(--ui-text-quaternary)' }),
                jsx('div', {
                  className: 'text-xs leading-5 text-(--ui-text-tertiary)',
                  children: filterHint
                    ? filterHint
                    : 'Cronjobs are recurring tasks this agent runs on a schedule.'
                }),
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
                children: jobs.map(job => jsx(RoutineRow, { job, profile: bot }, job.job_id))
              })
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

// ── profile session workspace ────────────────────────────────────────────────

const PROFILE_SESSION_LIST_LIMIT = 200

function openBotSessionsWorkspace(bot) {
  if (bot?.name && NAME_RE.test(bot.name)) {
    $botSessionsWorkspace.set(bot.name)
  }
}

function filterProfileSessions(sessions, query) {
  const needle = String(query || '').trim().toLowerCase()
  const rows = Array.isArray(sessions) ? sessions : []
  if (!needle) return rows
  return rows.filter(session =>
    `${session?.title || ''} ${session?.preview || ''} ${session?.source || ''}`.toLowerCase().includes(needle)
  )
}

function useProfileSessions(botName, gatewayGeneration) {
  return useQuery({
    queryKey: [ID, 'profile-sessions', botName, gatewayGeneration],
    enabled: Boolean(botName),
    // include_hidden: this browser exists precisely to see the profile's own
    // (always-hidden) Bot Mode sessions alongside its regular ones.
    queryFn: () => host.request('session.list', { profile: botName, limit: PROFILE_SESSION_LIST_LIMIT, include_hidden: true }),
    refetchInterval: 8000,
    staleTime: 4000,
    retry: false
  })
}

async function openProfileSession(botName, storedId, gatewayGeneration) {
  const profile = String(botName || '')
  const id = String(storedId || '')
  if (!NAME_RE.test(profile) || !id || gatewayGeneration !== $sessionsGatewayGeneration.get()) return
  if (typeof host.openSession !== 'function') {
    throw new Error('This Hermes Desktop version cannot open stored sessions')
  }
  await host.openSession(id, { profile })
  if (gatewayGeneration !== $sessionsGatewayGeneration.get()) return
  $botSelectedSessions.set({ ...$botSelectedSessions.get(), [profile]: id })
}

function ProfileSessionRow({ session, botName, active, gatewayGeneration }) {
  return jsxs('button', {
    type: 'button',
    'aria-current': active ? 'page' : undefined,
    onClick: () => void openProfileSession(botName, session.id, gatewayGeneration).catch(err => host.notifyError(err, 'Could not open session')),
    className: cn(
      'flex w-full flex-col gap-0.5 overflow-hidden rounded-md px-2 py-1.5 text-left transition-colors',
      'hover:bg-(--chrome-action-hover)',
      active && 'bg-(--ui-row-active-background)'
    ),
    children: [
      jsx('span', {
        className: 'truncate text-[0.8125rem] font-medium',
        children: session.title || 'Untitled session'
      }),
      jsx('div', {
        className: 'truncate text-[0.7rem] text-(--ui-text-tertiary)',
        children: session.preview || session.source || 'No messages yet'
      })
    ]
  })
}

function ProfileSessionsWorkspace({ bot }) {
  const gatewayGeneration = useValue($sessionsGatewayGeneration)
  const { data, isLoading, error } = useProfileSessions(bot.name, gatewayGeneration)
  const selectedByProfile = useValue($botSelectedSessions)
  const [query, setQuery] = useState('')
  const sourceSessions = data?.sessions || []
  const sessions = filterProfileSessions(sourceSessions, query)
  const inventoryBounded = sourceSessions.length >= PROFILE_SESSION_LIST_LIMIT
  const selectedId = selectedByProfile[bot.name] || ''

  const header = jsxs('div', {
    className: 'flex items-center gap-2 px-2.5 pt-2.5 pb-2',
    children: [
      jsx(Button, {
        variant: 'ghost',
        size: 'sm',
        onClick: () => $botSessionsWorkspace.set(null),
        children: 'Back'
      }),
      jsx('div', {
        className: 'min-w-0 flex-1 truncate text-sm font-semibold',
        children: `${displayName(bot, $botMeta.get()[bot.name])} sessions`
      })
    ]
  })

  return jsxs('div', {
    className: 'flex h-full flex-col',
    children: [
      header,
      jsx('div', {
        className: 'px-2 pb-2',
        children: jsx(Input, {
          'aria-label': 'Filter sessions',
          placeholder: 'Filter sessions…',
          value: query,
          onChange: event => setQuery(event.target.value)
        })
      }),
      inventoryBounded
        ? jsx('div', {
            className: 'px-2.5 pb-2 text-[0.65rem] text-(--ui-text-quaternary)',
            children: `Showing the ${PROFILE_SESSION_LIST_LIMIT} most recent sessions.`
          })
        : null,
      isLoading
        ? jsx('div', {
            className: 'flex flex-1 items-center justify-center',
            children: jsx(GlyphSpinner, { spinner: 'breathe' })
          })
        : error
          ? jsx('div', {
              className: 'px-3 py-3 text-xs text-(--ui-text-tertiary)',
              children: 'Could not load sessions for this profile.'
            })
          : jsx(ScrollArea, {
              className: 'min-h-0 flex-1',
              children: jsx('div', {
                className: 'grid gap-0.5 px-1.5 pb-2',
                children: sessions.length
                  ? sessions.map(session => jsx(ProfileSessionRow, {
                      session,
                      botName: bot.name,
                      active: selectedId === session.id,
                      gatewayGeneration
                    }, session.id))
                  : jsx('div', {
                      className: 'px-2 py-3 text-center text-xs text-(--ui-text-tertiary)',
                      children: query.trim()
                        ? inventoryBounded
                          ? `No matching sessions in the ${PROFILE_SESSION_LIST_LIMIT} most recent.`
                          : 'No sessions match that filter.'
                        : 'No stored sessions yet.'
                    })
              })
            })
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
        const meta = metaByName?.[bot.name]
        const { shape, color, image } = botAppearance(bot.name, meta)
        const photo = Boolean(image && !isBackfilledFacePng(image))
        const label = displayName(bot, meta)

        return jsx('button', {
          type: 'button',
          title: `Open ${label}'s chat`,
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
        }, botRosterKey(bot))
      })
    ]
  })
}

/** Assign a bot to a group (or clear it). Existing groups are one-click;
 *  the input creates a new one. The group is a bot-meta field, so it syncs
 *  cross-machine via ui_meta like pin/title. */
function GroupDialog({ bot, onClose }) {
  const meta = useValue($botMeta)
  const [name, setName] = useState('')
  const current = (meta[bot?.name]?.group || '').trim()
  const groups = knownGroups(meta)

  const assign = group => {
    saveBotMeta(bot.name, { group: group || null })
    host.notify({
      kind: 'info',
      message: group
        ? `${displayName(bot, meta[bot.name])} moved to “${group}”`
        : `${displayName(bot, meta[bot.name])} removed from its group`
    })
    onClose()
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
            jsx(DialogTitle, { children: 'Move to group' }),
            jsx(DialogDescription, {
              children: 'Groups render as labeled sections in the Bots roster and sync to every machine.'
            })
          ]
        }),
        groups.length
          ? jsx('div', {
              className: 'flex flex-wrap gap-1.5',
              children: groups.map(group =>
                jsx(Button, {
                  variant: group === current ? 'default' : 'secondary',
                  size: 'sm',
                  onClick: () => assign(group),
                  children: group
                }, group)
              )
            })
          : null,
        jsxs('form', {
          className: 'flex items-center gap-1.5',
          onSubmit: event => {
            event.preventDefault()
            const trimmed = name.trim()

            if (trimmed) {
              assign(trimmed)
            }
          },
          children: [
            jsx(Input, {
              autoFocus: true,
              placeholder: groups.length ? 'New group…' : 'Group name (e.g. Research)',
              value: name,
              onChange: event => setName(event.target.value)
            }),
            jsx(Button, { type: 'submit', size: 'sm', disabled: !name.trim(), children: 'Create' })
          ]
        }),
        current
          ? jsx(Button, {
              variant: 'ghost',
              size: 'sm',
              className: 'justify-self-start',
              onClick: () => assign(null),
              children: `Remove from “${current}”`
            })
          : null
      ]
    })
  })
}

/** Discord-style group chat creation: pick 2+ bots via checkboxes (with
 *  search), name the group, create. Assignment is the existing per-bot
 *  `group` meta field, so the room appears in the roster and syncs
 *  cross-machine via ui_meta exactly like Move-to-group. */
function CreateGroupChatDialog({ open, roster, onClose, onCreated }) {
  const allMeta = useValue($botMeta)
  const [query, setQuery] = useState('')
  const [checked, setChecked] = useState({})
  const [name, setName] = useState('')

  // Reset per open so a cancelled draft doesn't leak into the next one.
  useEffect(() => {
    if (open) {
      setQuery('')
      setChecked({})
      setName('')
    }
  }, [open])

  const selected = roster.filter(bot => checked[botRosterKey(bot)])
  const visible = filterBots(roster, allMeta, query)
  const atCap = selected.length >= GROUP_CHAT_MAX_MEMBERS
  const placeholder = selected.length
    ? selected.map(bot => displayName(bot, botRosterMeta(bot, allMeta))).join(', ')
    : 'Group name'
  const canCreate = selected.length >= 2 && Boolean(name.trim() || selected.length)

  const create = () => {
    const groupName = (name.trim() || placeholder).slice(0, 64)

    if (selected.length < 2 || !groupName) {
      return
    }

    for (const bot of selected) {
      if (!bot.remoteSource) {
        void saveBotMeta(bot.name, { group: groupName })
      }
    }

    // Members from OTHER connections can't ride bot-meta (it is scoped to
    // the active gateway and name-keyed). Their descriptors live on the room
    // record instead, so the roster merge can seat them every refresh.
    const remoteMembers = selected
      .filter(bot => bot.remoteSource)
      .map(bot => ({
        name: bot.name,
        handle: bot.handle || bot.name,
        connectionId: bot.connectionId,
        connectionKind: bot.connectionKind,
        connectionLabel: bot.connectionLabel,
        remoteSource: true,
        sourceScoped: true
      }))

    if (remoteMembers.length) {
      updateGroupChat(groupName, room => {
        room.members = remoteMembers
        return room
      })
    }

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
              children: `Pick 2–${GROUP_CHAT_MAX_MEMBERS} bots. The room lives in the Bots roster and syncs to every machine.`
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
                  const currentGroup = (meta?.group || '').trim()

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
                              currentGroup
                                ? `@${botHandle(bot.name, bot)} · in “${currentGroup}”`
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
                  children: query.trim() ? `No bots match “${query.trim()}”` : 'No bots yet — create agents first.'
                })
          })
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

/** Merged room view for one group: shared timeline with per-member
 *  attribution, a composer that drives the round-robin, and a working
 *  indicator while member turns run. Renders identically in the MAIN chat
 *  window (host.openWorkspace tile) and in the bots panel (older-desktop
 *  fallback); `onBack` is where the Back button routes — the main tile's
 *  closer, or clearing the in-panel workspace atom. */
function GroupChatWorkspace({ group, members, onBack }) {
  const rooms = useValue($groupChats)
  const allMeta = useValue($botMeta)
  const room = rooms[group] || { log: [], running: false }
  const [draft, setDraft] = useState('')
  const [confirmDisband, setConfirmDisband] = useState(false)
  // Click-to-disambiguate: which log entry is showing its speaker's full
  // @handle (the roster's name-device form when names collide across
  // connections). Naturally every speaker just shows its display name.
  const [revealedSpeaker, setRevealedSpeaker] = useState(null)

  const header = jsxs('div', {
    className: 'flex items-center gap-2 px-2.5 pt-2.5 pb-2',
    children: [
      jsx(Button, {
        variant: 'ghost',
        size: 'sm',
        onClick: () => (onBack ? onBack() : $groupChatWorkspace.set(null)),
        children: 'Back'
      }),
      jsx('div', {
        className: 'min-w-0 flex-1 truncate text-sm font-semibold',
        children: `${group} — group chat`
      }),
      // Member faces: the room's roster at a glance, matching each bot's
      // avatar in the sidebar. Falls back to the count for the title tooltip.
      jsx('div', {
        className: 'flex shrink-0 items-center -space-x-1.5',
        title: members.map(b => displayName(b, botRosterMeta(b, allMeta))).join(', '),
        children: members.slice(0, 6).map(b => {
          const bMeta = botRosterMeta(b, allMeta)
          const { shape, color, image } = botAppearance(b.name, bMeta)
          const photo = Boolean(image && !isBackfilledFacePng(image))

          return jsx('div', {
            className: 'rounded-full ring-2 ring-(--ui-bg-primary,#111)',
            children: jsx(BotFace, { shape, color, image: photo ? image : null, size: 20, name: b.name })
          }, botRosterKey(b))
        })
      }),
      jsx('span', {
        className: 'shrink-0 text-[0.65rem] text-(--ui-text-quaternary)',
        children: `${members.length} bots`
      }),
      jsx(Button, {
        variant: 'ghost',
        size: 'sm',
        className: 'shrink-0 text-(--ui-text-tertiary) hover:text-destructive',
        title: `Disband the ${group} group chat`,
        onClick: () => setConfirmDisband(true),
        children: jsx(Codicon, { name: 'trash' })
      })
    ]
  })

  const submit = () => {
    const text = draft.trim()

    if (!text) {
      return
    }

    setDraft('')
    // Full descriptors ride into the turn loop: remote members keep their
    // connection fields so their turns route to their own machines.
    sendToGroupChat(
      group,
      members.map(b => ({
        ...b,
        title: (b.remoteSource ? '' : allMeta[b.name]?.title) || b.title || ''
      })),
      text
    )
  }

  return jsxs('div', {
    className: 'flex h-full flex-col',
    children: [
      header,
      jsx(ScrollArea, {
        className: 'min-h-0 flex-1',
        children: jsxs('div', {
          className: 'grid gap-1.5 px-2.5 pb-2',
          children: [
            ...(room.log.length
              ? room.log.map((entry, index) => {
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
                      'flex items-start gap-2',
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
                            className: 'flex items-baseline gap-2',
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
                              })
                            ]
                          }),
                          jsx('div', {
                            className: 'text-xs text-(--ui-text-secondary) [&_p]:mb-1 [&_p:last-child]:mb-0',
                            children: Streamdown ? jsx(Streamdown, { children: entry.text }) : entry.text
                          })
                        ]
                      })
                    ]
                  }, entryKey)
                })
              : [
                  jsx('div', {
                    className: 'px-2 py-4 text-center text-xs text-(--ui-text-tertiary)',
                    children: 'Say something — every bot in this group hears the room.'
                  }, 'empty')
                ]),
            room.running
              ? jsx('div', {
                  className: 'px-2 py-1 text-[0.7rem] italic text-(--ui-text-quaternary)',
                  children: 'The room is working…'
                }, 'working')
              : null
          ]
        })
      }),
      jsx('div', {
        className: 'border-t border-(--ui-stroke-secondary) p-2',
        children: jsxs('form', {
          className: 'flex items-center gap-1.5',
          onSubmit: event => {
            event.preventDefault()
            submit()
          },
          children: [
            jsx(Input, {
              'aria-label': `Message ${group}`,
              placeholder: `Message ${group}… (@name to direct, @everyone for all)`,
              value: draft,
              onChange: event => setDraft(event.target.value)
            }),
            jsx(Button, { type: 'submit', size: 'sm', disabled: !draft.trim(), children: 'Send' })
          ]
        })
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
            ' bots and clears the shared room log. The bots themselves and their “Group: ',
            group,
            '” sessions are kept — you can still open those from each bot’s session browser.'
          ]
        }),
        destructive: true,
        confirmLabel: 'Disband',
        busyLabel: 'Disbanding…',
        doneLabel: 'Disbanded',
        onClose: () => setConfirmDisband(false),
        onConfirm: async () => {
          await disbandGroupChat(group, members.map(bot => bot.name))
          host.notify({ kind: 'success', message: `Disbanded “${group}”` })
        }
      })
    ]
  })
}

/** Live closers for group-chat MAIN-window tabs, by group name — so a
 *  disband (or the room view's own Back) can retire the tab it opened. */
const groupChatMainTabs = new Map()

function closeGroupChatMainTab(group) {
  const close = groupChatMainTabs.get(group)

  groupChatMainTabs.delete(group)

  if (typeof close === 'function') {
    try {
      close()
    } catch {
      /* tab already gone */
    }
  }
}

/** Main-window wrapper: seats the member roster reactively (live roster +
 *  bot meta + the room's stored cross-connection descriptors) so the room
 *  keeps working as members change while the tab is open. */
function GroupChatMainView({ group }) {
  const allMeta = useValue($botMeta)
  // Subscribe: membership changes ride bot meta AND the room record.
  useValue($groupChats)
  const roster = useValue($lastRoster)
  const members = groupChatMemberBots(group, roster, allMeta)

  return jsx(GroupChatWorkspace, { group, members, onBack: () => closeGroupChatMainTab(group) })
}

/** Open a group chat the Discord way: a tab taking over the MAIN chat window
 *  (host.openWorkspace, newer desktops), falling back to the in-panel room
 *  view on desktops whose SDK predates the main-area door. */
function openGroupChat(group) {
  $groupNeedsYou.set({ ...$groupNeedsYou.get(), [group]: false })

  if (typeof host.openWorkspace === 'function') {
    try {
      const close = host.openWorkspace(`${ID}:group:${slugify(group)}`, {
        title: group,
        minWidth: '24rem',
        render: () => jsx(GroupChatMainView, { group }),
        onClose: () => groupChatMainTabs.delete(group)
      })

      groupChatMainTabs.set(group, close)

      return
    } catch {
      // Fall through to the in-panel room below.
    }
  }

  $groupChatWorkspace.set(group)
}

/** One group chat as ONE roster row — the Discord shape: stacked member
 *  avatars, group name, member count, the newest room line as the preview
 *  (markdown flattened), relative time of the last activity, and the
 *  needs-you badge on the row itself. Sorts into the same recency ordering
 *  as bot rows; clicking opens the room in the main chat window. */
function GroupRow({ group, members, needsYou, onOpen }) {
  const rooms = useValue($groupChats)
  const allMeta = useValue($botMeta)
  const room = rooms[group] || { log: [] }
  const log = Array.isArray(room.log) ? room.log : []
  const last = log.length ? log[log.length - 1] : null
  const lastAt = groupLastActivity(room)
  const preview = last
    ? `${last.from?.kind === 'user' ? 'You' : `@${last.from?.name || 'bot'}`}: ${stripPreviewMarkdown(last.text) || '…'}`
    : 'No messages yet — say hi to the room'
  const faces = members.slice(0, 3)

  return jsxs('button', {
    type: 'button',
    onClick: () => {
      haptic('tap')
      onOpen(group)
    },
    className: cn(
      'flex w-full min-w-0 max-w-full items-center gap-2.5 overflow-hidden rounded-md px-2 py-2 text-left transition-colors',
      'hover:bg-(--chrome-action-hover)'
    ),
    children: [
      // Composite avatar: up to three member faces fanned like Discord's
      // group-DM icon; a bare glyph when the room has no seated members.
      jsx('div', {
        className: 'flex w-[34px] shrink-0 items-center justify-center',
        children: faces.length
          ? jsx('div', {
              className: 'flex items-center -space-x-2.5',
              children: faces.map(member => {
                const meta = member.remoteSource ? null : allMeta[member.name]
                const { shape, color, image } = botAppearance(member.name, meta)

                return jsx(
                  'div',
                  {
                    className: 'rounded-full ring-2 ring-(--ui-bg-primary,#111)',
                    children: jsx(BotFace, {
                      shape,
                      color,
                      image: image && !isBackfilledFacePng(image) ? image : null,
                      size: 20,
                      name: member.name,
                      mood: 'idle'
                    })
                  },
                  botRosterKey(member)
                )
              })
            })
          : jsx(Codicon, { name: 'organization', className: 'text-(--ui-text-tertiary)' })
      }),
      jsxs('div', {
        className: 'min-w-0 flex-1',
        children: [
          jsxs('div', {
            className: 'flex items-baseline justify-between gap-2',
            children: [
              jsxs('div', {
                className: 'flex min-w-0 items-baseline gap-1.5 truncate',
                children: [
                  jsx('span', { className: 'truncate text-[0.8125rem] font-medium', children: group }),
                  jsx('span', {
                    className: 'shrink-0 text-[0.6875rem] text-(--ui-text-quaternary)',
                    children: `${members.length} bots`
                  })
                ]
              }),
              needsYou
                ? jsx('span', {
                    className:
                      'shrink-0 rounded-full bg-(--ui-accent,#4f9cf9) px-1.5 text-[0.6rem] font-semibold text-white',
                    title: 'A bot in this room needs your input',
                    children: 'needs you'
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
  const [grouping, setGrouping] = useState(null)
  const [query, setQuery] = useState('')
  const activityToasts = useValue($activityToasts)
  const sessionsWorkspaceName = useValue($botSessionsWorkspace)
  const groupChatName = useValue($groupChatWorkspace)
  const groupNeedsYou = useValue($groupNeedsYou)
  const groupRooms = useValue($groupChats)

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
    const lastMsg = (bot.last_session?.last_active || 0) * 1000

    return Math.max(created, lastMsg)
  }
  // Pinned bots (right-click → Pin) float to the top as a group; within the
  // pinned group and within the unpinned group, recency still rules. A
  // plain boolean flag in bot-meta (rides ui_meta to every machine).
  const isPinned = bot => Boolean(botRosterMeta(bot, allMeta)?.pinned)
  // Resilience (@wesleysimplicio, #13): a failed refresh must not erase a
  // roster the user already had — mixed local+cloud gateways and remotes
  // waking from sleep fail transiently. Render the last good snapshot with
  // a notice; the full error card is reserved for "never had a roster".
  const live = Array.isArray(data?.profiles) ? data.profiles : null
  const source = live ?? (error ? $lastRoster.get() : [])
  const roster = source.slice().sort((a, b) => {
    const pa = isPinned(a) ? 1 : 0
    const pb = isPinned(b) ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return activityOf(b) - activityOf(a)
  })
  const activeSourceRoster = roster.filter(bot => !bot.remoteSource)
  // Hidden bots (right-click → Hide Bot) drop out of the roster list unless
  // the header eye toggle reveals them. Display-only: every other consumer
  // (mentions, group chats, name-collision checks, merge/avatar/activity
  // sweeps) keeps the FULL roster.
  const showHidden = useValue($showHiddenBots)
  const unreadByName = useValue($botUnread)
  const hiddenBots = roster.filter(bot => isBotHidden(bot, allMeta))
  const hiddenUnread = hiddenBots.some(bot => !bot.remoteSource && unreadByName[bot.name])
  const visibleRoster = showHidden ? roster : roster.filter(bot => !isBotHidden(bot, allMeta))
  const filteredRoster = filterBots(visibleRoster, allMeta, query)
  // Group chats are first-class roster rows (Discord-style): one standalone
  // row per room, competing in the SAME recency ordering as bot rows — a
  // group's activity is its newest room-log line. Pinned bots still lead;
  // groups and unpinned bots interleave by recency below them.
  const needle = query.trim().toLowerCase()
  const groupRows = groupChatNames(allMeta, groupRooms)
    .filter(name => !needle || name.toLowerCase().includes(needle))
    .map(name => ({
      kind: 'group',
      name,
      members: groupChatMemberBots(name, roster, allMeta),
      activity: groupLastActivity(groupRooms[name])
    }))
  const rosterRows = [
    ...filteredRoster.map(bot => ({ kind: 'bot', bot, pinned: isPinned(bot), activity: activityOf(bot) })),
    ...groupRows
  ].sort((a, b) => {
    const pa = a.pinned ? 1 : 0
    const pb = b.pinned ? 1 : 0

    if (pa !== pb) {
      return pb - pa
    }

    return b.activity - a.activity
  })

  if (live) {
    $lastRoster.set(roster)
    mergeServerMeta(activeSourceRoster)
    pullServerAvatars(activeSourceRoster)
    trackInboundActivity(activeSourceRoster)
    backfillMessagingProtocol(activeSourceRoster)
  }

  const staleNotice = error && !live && roster.length
    ? 'Roster refresh failed — showing the last good list.' + (gatewayUp ? '' : ' Waiting for the gateway to reconnect…')
    : null
  const sessionsWorkspaceBot = roster.find(bot => bot.name === sessionsWorkspaceName)

  if (sessionsWorkspaceBot) {
    return jsx(ProfileSessionsWorkspace, { bot: sessionsWorkspaceBot })
  }

  const groupChatMembers = groupChatName ? groupChatMemberBots(groupChatName, roster, allMeta) : []

  if (groupChatName && groupChatMembers.length) {
    return jsx(GroupChatWorkspace, { group: groupChatName, members: groupChatMembers })
  }

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
              // Eye toggle appears only once something is hidden — zero
              // hidden bots means zero extra chrome. It stays visible while
              // hidden rows are revealed, so Unhide is always reachable.
              hiddenBots.length
                ? jsx(Tip, {
                    label: showHidden
                      ? 'Hide hidden bots again'
                      : `Show ${hiddenBots.length} hidden bot${hiddenBots.length === 1 ? '' : 's'}`,
                    children: jsxs('button', {
                      type: 'button',
                      'aria-label': showHidden ? 'Hide hidden bots' : 'Show hidden bots',
                      className: cn(
                        'relative flex size-6 items-center justify-center rounded-md transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground',
                        showHidden ? 'text-foreground' : 'text-(--ui-text-tertiary)'
                      ),
                      onClick: () => $showHiddenBots.set(!showHidden),
                      children: [
                        jsx(Codicon, { name: showHidden ? 'eye' : 'eye-closed' }),
                        hiddenUnread && !showHidden
                          ? jsx('span', {
                              className:
                                'absolute right-0.5 top-0.5 size-1.5 rounded-full bg-(--ui-accent,#4f9cf9)',
                              'aria-label': 'a hidden bot has unread activity'
                            })
                          : null
                      ]
                    })
                  })
                : null,
              jsxs(DropdownMenu, {
                children: [
                  jsx(Tip, {
                    label: 'New…',
                    children: jsx(DropdownMenuTrigger, {
                      asChild: true,
                      children: jsx('button', {
                        type: 'button',
                        'aria-label': 'New agent or group chat',
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
                        children: [jsx(Codicon, { name: 'hubot', className: 'mr-1.5' }), 'New Agent']
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
        onOpen: bot => {
          haptic('tap')
          $selectedBot.set(bot.name)

          if (bot.remoteSource) {
            const handle = botHandle(bot.name, bot)
            host.notify?.({
              kind: 'info',
              title: displayName(bot),
              message: `Stay in this chat and @${handle} to message them. Gateway stays on this device.`
            })
            return
          }

          if ($botUnread.get()[bot.name]) {
            const next = { ...$botUnread.get() }
            delete next[bot.name]
            $botUnread.set(next)
          }

          void (async () => {
            let pinnedChat = botRosterMeta(bot, allMeta)?.chat

            try {
              pinnedChat = await prepareBotSource(bot, pinnedChat)
            } catch (error) {
              host.notifyError?.(error, `Could not reach ${bot.connectionLabel || 'the remote source'}`)

              return
            }

            try {
              const id = await openBotCanonicalChat(bot.name, pinnedChat, bot.last_session)

              if (id) {
                return
              }
            } catch {
              // Fall through to the older-gateway draft below.
            }

            if (typeof host.newChat === 'function') {
              host.newChat(bot.name)
            } else {
              host.navigate('/')
            }
          })()
        }
      }),
      roster.length
        ? jsx('div', {
            className: 'px-2.5 pb-1.5',
            children: jsx(SearchField, {
              'aria-label': 'Search bots',
              containerClassName: 'w-full',
              inputClassName: 'w-full',
              placeholder: 'Search bots…',
              value: query,
              onChange: setQuery
            })
          })
        : null,
      staleNotice
        ? jsx('div', {
            className: 'mx-2.5 mb-1 rounded-md bg-(--chrome-action-hover) px-2 py-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
            children: staleNotice
          })
        : null,
      isLoading && !roster.length
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
                title: 'No agents yet',
                description: 'Create your first teammate.'
              })
            : filteredRoster.length === 0 && rosterRows.length === 0
              ? jsx('div', {
                  'aria-live': 'polite',
                  className:
                    'flex flex-1 items-center justify-center px-4 text-center text-xs text-(--ui-text-tertiary)',
                  role: 'status',
                  children: query.trim()
                    ? `No bots match “${query.trim()}”`
                    : 'All bots are hidden — use the eye button above to show them.'
                })
              : jsx(ScrollArea, {
                  className: 'hermes-bots-roster min-h-0 flex-1',
                  children: jsx('div', {
                    className: 'grid w-full min-w-0 gap-0.5 px-1.5 pb-2',
                    // Flat, Discord-style list: bot rows and group rows
                    // interleaved by recency — no section headers.
                    children: rosterRows.map(row =>
                      row.kind === 'group'
                        ? jsx(
                            GroupRow,
                            {
                              group: row.name,
                              members: row.members,
                              needsYou: Boolean(groupNeedsYou[row.name]),
                              onOpen: openGroupChat
                            },
                            `group:${row.name}`
                          )
                        : jsx(
                            BotRow,
                            { bot: row.bot, onDelete: setDeleting, onEdit: setEditing, onGroup: setGrouping },
                            botRosterKey(row.bot)
                          )
                    )
                  })
                }),
      jsx('div', {
        className: 'border-t border-(--ui-stroke-secondary) p-2',
        children: jsxs(Button, {
          className: 'w-full justify-center gap-1.5',
          variant: 'secondary',
          onClick: () => setCreateOpen(true),
          children: [jsx(Codicon, { name: 'add' }), 'New Agent']
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
    startFaceClock()
    // Disabling the plugin (or a hot reload) must actually stop the clock —
    // before this, the rAF loop + 1Hz document scan ran until app restart.
    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(stopFaceClock)
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
          const roster = queryClient.getQueryData(ROSTER_KEY)
          const profiles = Array.isArray(roster?.profiles) ? roster.profiles : []

          if (!profiles.length) {
            return []
          }

          const active = (host.state.profile.get() || 'default').trim() || 'default'
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

            if (q && !handle.toLowerCase().startsWith(q)) {
              continue
            }

            const display = displayName(profile, $botMeta.get()[profile.name])
            const source = profile.connectionLabel ? ` · ${profile.connectionLabel}` : ''

            items.push({
              insert: `@${handle}`,
              display: `@${handle}`,
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

    // Hydrate persisted avatars/titles. Storage may be sync, async, or
    // absent depending on shell version — normalize through Promise.resolve
    // inside a try so a storage quirk can NEVER fail the plugin load.
    try {
      Promise.resolve(ctx.storage?.get?.('bot-meta'))
        .then(value => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            const live = $botMeta.get()
            const next = { ...value }
            for (const name of Object.keys(live)) {
              next[name] = { ...(value[name] || {}), ...live[name] }
            }
            $botMeta.set(next)
          }
        })
        .catch(() => undefined)
    } catch {
      /* no storage on this shell — defaults stay */
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
        .then(value => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            const rooms = {}

            for (const [name, room] of Object.entries(value)) {
              if (room && Array.isArray(room.log)) {
                rooms[name] = {
                  log: room.log,
                  watermarks: room.watermarks && typeof room.watermarks === 'object' ? room.watermarks : {},
                  sessions: room.sessions && typeof room.sessions === 'object' ? room.sessions : {},
                  members: Array.isArray(room.members) ? room.members : [],
                  epoch: 0,
                  running: false
                }
              }
            }

            $groupChats.set({ ...rooms, ...$groupChats.get() })
          }
        })
        .catch(() => undefined)
    } catch {
      /* no storage — rooms start empty */
    }

    // Routines follow the chat you're in: track the live gateway profile.
    // Capture the unbinds: without them a disable → re-enable cycle stacks a
    // duplicate listener per cycle (same survives-disable class as the face
    // clock before its onDispose hook — these kept firing until app restart).
    const unbindProfileListener = host.state.profile.listen(profile => {
      if (profile && typeof profile === 'string') {
        $selectedBot.set(profile)
      }
    })
    const unbindGatewayListener = host.state.gateway.listen(handleSessionsGatewayTransition)

    if (typeof ctx.onDispose === 'function') {
      ctx.onDispose(() => {
        if (typeof unbindProfileListener === 'function') {
          unbindProfileListener()
        }
        if (typeof unbindGatewayListener === 'function') {
          unbindGatewayListener()
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
      data: { placement: 'left', width: '260px', dock: { pane: 'sessions', pos: 'center', enforce: true } },
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
          dock: { pane: 'workspace', pos: 'right' },
          width: '250px'
        },
        render: () => jsx(RoutinesPane, {})
      })

    if (typeof host.paneVisibility === 'function') {
      // The contribution-scoped pane id (`register` prefixes `${ID}:`).
      const $botsPaneVisible = host.paneVisibility(`${ID}:pane`)
      let unregisterRoutines = null

      const syncRoutinesPane = visible => {
        if (visible) {
          unregisterRoutines ??= registerRoutinesPane()
        } else if (unregisterRoutines) {
          unregisterRoutines()
          unregisterRoutines = null
        }
      }

      const stopRoutinesSync = $botsPaneVisible.listen(syncRoutinesPane)
      syncRoutinesPane($botsPaneVisible.get())

      if (typeof ctx.onDispose === 'function') {
        // The registration disposer is already tracked by ctx.register; only
        // the listener needs explicit teardown or it survives plugin disable.
        ctx.onDispose(stopRoutinesSync)
      }
    } else {
      registerRoutinesPane()
    }

    ctx.register({
      id: 'new-agent',
      area: PALETTE_AREA,
      data: {
        id: `${ID}.new-agent`,
        label: 'New Agent…',
        keywords: ['bot', 'agent', 'profile', 'teammate', 'create'],
        run: () => {
          host.notify({ kind: 'info', message: 'Open the Bots pane and hit “New Agent”.' })
        }
      }
    })

    // @-mention middleware: "@<bot> do the thing" in any chat becomes an
    // explicit handoff instruction the active agent's SOUL.md knows how to
    // execute. Names are validated against the LIVE roster so
    // "user@example.com" or an unknown @ passes through untouched.
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
            const meta = activeBot ? $botMeta.get()[activeBot] : null
            const pinnedId = meta?.chat || null
            const currentId = host.activeSessionId?.get?.() ?? null

            if (activeBot && pinnedId && currentId && String(currentId) === String(pinnedId)) {
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
            name: (host.state.profile.get() || 'default').trim() || 'default',
            connectionId: String(host.state.connectionId?.get?.() || host.activeConnectionId?.() || 'local')
          }
          const cached = typeof queryClient !== 'undefined' && queryClient && typeof queryClient.getQueryData === 'function'
            ? queryClient.getQueryData(ROSTER_KEY)
            : null
          const roster = Array.isArray(cached?.profiles) ? cached.profiles : null
          let mentionedBots = roster ? resolveRosterMentions(text, roster, live) : []

          if (!roster) {
            let names = []
            try {
              const res = await host.request('profiles.list', { include_sessions: false })
              names = (res?.profiles ?? []).map(p => p.name)
            } catch {
              return draft
            }

            const prose = text.replace(/```[\s\S]*?```/g, ' ').replace(/`[^`\n]*`/g, ' ')
            const mentioned = []

            for (const match of prose.matchAll(/(^|\s)@([a-z0-9][a-z0-9_-]*)/gi)) {
              let name = match[2].toLowerCase()

              if (name === 'hermes' && !names.includes('hermes') && names.includes('default')) {
                name = 'default'
              }

              if (names.includes(name) && name !== live.name && !mentioned.includes(name)) {
                mentioned.push(name)
              }
            }

            mentionedBots = mentioned.map(name => ({ name }))
          }

          if (!mentionedBots.length) {
            return draft
          }

          const localMentions = mentionedBots.filter(bot => !bot.remoteSource)
          const remoteMentions = mentionedBots.filter(bot => bot.remoteSource)

          const activeMeta = $botMeta.get()[live.name]
          const senderName = displayName({ name: live.name, title: activeMeta?.title }, activeMeta)

          if (remoteMentions.length && typeof host.requestProfile === 'function') {
            void deliverRemoteRosterMentions(remoteMentions, text, {
              name: senderName,
              handle: botHandle(live.name)
            })
          }
          let note = ''

          if (localMentions.length) {
            note +=
              '\n\n[@mention handoff — for each mentioned agent (' + localMentions.map(bot => botHandle(bot.name, bot)).join(', ') + '): ' +
              'COMPOSE a message from you (' + senderName + ') to that agent conveying what the user wants — do not forward this text verbatim (avoid double quotes in your composed message). Send it with exactly one terminal call, run with background=true AND notify_on_complete=true (the recipient may take minutes; the user must not be blocked):\n' +
              localMentions.map(bot => '`hermes -p ' + shellQuote(bot.name) + ' chat --in ~ -c "Bot Chat" --create-if-missing -Q -q "Message from 🤖 ' + shellDoubleQuote(senderName) + ' (@' + shellDoubleQuote(botHandle(live.name)) + '): <your composed message>"`').join('\n') +
              '\nAfter dispatching, tell the user the message was sent and END YOUR TURN — do not wait or poll; when the background process completes, its notification carries the reply — relay it then, attributed to that agent. ' +
              'Relay the reply back to the user, attributed to that agent.]'
          }

          if (remoteMentions.length) {
            const labels = remoteMentions.map(bot => `@${botHandle(bot.name, bot)} (${bot.connectionLabel || bot.connectionId})`).join(', ')
            note +=
              '\n\n[@mention — stay on this device. Desktop is delivering to ' + labels +
              ' over Connections in the background. Do not run hermes -p for them and do not switch Gateway. Tell the user they were messaged here; when a reply lands, relay it attributed to that agent.]'
          }

          return { ...draft, text: text + note }
        }      }
    })
  }
}
