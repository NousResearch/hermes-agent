/**
 * The profile operations that reach the gateway rather than the screen:
 * two-way avatar sync, duplicate, and delete.
 *
 * They sit below every surface because both the roster and the create dialog
 * drive them — the dialog deletes its own draft profile on cancel, the roster
 * deletes for real — and neither can own them without the other importing a
 * surface.
 */

import { forgetSessionUnread, host, queryClient } from '@hermes/plugin-sdk'

import { isBackfilledFacePng } from './avatar-image'
import {
  $focusedBotOwner,
  $openBotChat,
  $selectedBot,
  clearSelectedRosterBot,
  focusedRosterOwner,
  rosterWatermarks
} from './bot-state'
import { ensureBotMetadata } from './canonical-chat'
import {
  $botMeta,
  BOT_META_V1_KEY,
  botMetaKey,
  botMetaWriteAt,
  botRosterKey,
  botSelectionKey,
  commitBotMetaV2,
  isDefaultBot,
  persistBotMetaSnapshot,
  ROSTER_KEY,
  saveBotMeta
} from './data'
import { botConnectionRoute, botRouteKey, requestForBot } from './routing'
import { getPluginCtx } from './shared'
import type { RosterRow } from './types'

/** Fetch server-side avatars for roster rows flagged has_avatar when the
 *  local cache doesn't already have an image for them. Fire-and-forget. */
const avatarFetchInflight = new Set<string>()
const avatarPushInflight = new Set<string>()

/** Backfill: local meta has art the server lacks -> profiles.set_asset.
 *  Server-side avatars power the inter-agent notice pfp (core #85855) and
 *  cross-machine roster art, so local-only images are a bug, not a state. */
function pushLocalAvatars(roster: RosterRow[]) {
  for (const bot of roster) {
    const key = botMetaKey(bot)

    if (bot.has_avatar || avatarPushInflight.has(key)) {
      continue
    }

    const image = $botMeta.get()[key]?.image

    if (image && typeof image === 'string' && image.startsWith('data:')) {
      avatarPushInflight.add(key)

      const request = bot.sourceScoped
        ? requestForBot(bot, 'profiles.set_asset', {
            name: bot.name,
            asset: 'avatar',
            data: image
          })
        : host.request('profiles.set_asset', {
            name: bot.name,
            asset: 'avatar',
            data: image
          })

      Promise.resolve(request)
        .then(() =>
          queryClient.invalidateQueries({
            queryKey: ['hermes-bots', 'roster']
          })
        )
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
              ? requestForBot(bot, 'profiles.set_asset', {
                  name: bot.name,
                  asset: 'avatar',
                  data: png
                })
              : host.request('profiles.set_asset', {
                  name: bot.name,
                  asset: 'avatar',
                  data: png
                })
            ).then(() =>
              queryClient.invalidateQueries({
                queryKey: ['hermes-bots', 'roster']
              })
            )
          : Promise.reject(new Error('rasterize failed'))
      )
      .catch(() => avatarPushInflight.delete(key))
  }
}

/** Serialize an inline SVG and draw it to a canvas -> PNG data URL. */
function rasterizeSvgToPng(svgEl: Element, size: number): Promise<null | string> {
  return new Promise(resolve => {
    try {
      // `Node.cloneNode` is declared to return `Node`, which has no
      // setAttribute — the clone of an Element is always an Element.
      const clone = svgEl.cloneNode(true) as Element
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
          // A null 2d context throws here on purpose: the catch below resolves
          // null, which is the same "no raster" answer the caller expects.
          canvas.getContext('2d')!.drawImage(img, 0, 0, size, size)
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

/** `profiles.get_asset` reply for the `avatar` asset. */
interface ProfilesGetAssetResult {
  /** Data URL. */
  data?: string
  found?: boolean
}

export function pullServerAvatars(roster: RosterRow[]) {
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
      ? requestForBot(bot, 'profiles.get_asset', {
          name: bot.name,
          asset: 'avatar'
        })
      : host.request('profiles.get_asset', {
          name: bot.name,
          asset: 'avatar'
        })

    Promise.resolve(assetRequest as Promise<ProfilesGetAssetResult>)
      .then(res => {
        if (res?.found && res.data) {
          const current = $botMeta.get()
          const mine = current[key] || {}

          // A 160px raster of the vector face is only for inter-agent
          // notices. Do not park it on the roster or the live face dies.
          if (isBackfilledFacePng(res.data) && mine.imageKind !== 'photo' && !mine.pet) {
            return
          }

          $botMeta.set({
            ...current,
            [key]: {
              ...mine,
              image: res.data
            }
          })
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
export function mergeServerMeta(roster: RosterRow[], fetchedAt = 0) {
  const local = $botMeta.get()
  let changed = false

  const next = {
    ...local
  }

  for (const bot of roster) {
    const server = bot.ui_meta?.['hermes-bots']

    if (server && typeof server === 'object') {
      const key = botMetaKey(bot)

      if (fetchedAt && fetchedAt < (botMetaWriteAt.get(key) || 0)) {
        continue
      }

      const mine = next[key] || {}

      const merged = {
        ...mine,
        ...server
      }

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
      persistBotMetaSnapshot(
        next,
        roster.some(bot => bot.sourceScoped)
      )
    } catch {
      /* storage unavailable — reconciliation lasts for this window only */
    }
  }
}

/** Clone a bot: profile (config/skills/SOUL/memory via clone_from) + look.
 *  Name is "<base>-2", "-3", … — first free slot against the live roster. */
export async function duplicateBot(bot: RosterRow, roster: RosterRow[]) {
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

    if (
      !roster.some(
        // A truthy ownerKey is minted from ownerRoute, so the route is present
        // on every path that reads it — a correlation TS can't follow.
        b => b.name === candidate && (!ownerKey || botMetaKey(b)?.startsWith(`${ownerRoute!.connectionId}::`))
      )
    ) {
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
    await saveBotMeta(
      {
        ...bot,
        name,
        // TODO(bot-mode-types): botConnectionRoute() returns null for an unrouted bot, so this
        // synthesized row can carry `route: null`, which RosterRow['route'] (ProfileRoute |
        // undefined) does not admit. Benign at runtime — every read of it is optional-chained —
        // but the assertion below is covering for a domain type that is too narrow.
        route: ownerRoute,
        sourceScoped: Boolean(ownerRoute)
      } as RosterRow,
      {
        ...look,
        title: meta.title ? `${meta.title} (copy)` : ''
      }
    )
  }

  return name
}

/** `cli.exec` reply, as the legacy delete path reads it. */
interface CliExecResult {
  blocked?: boolean
  code?: number
  hint?: string
  output?: string
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
export async function deleteBot(bot: RosterRow) {
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

    const result: CliExecResult = await host.request('cli.exec', {
      argv: ['profile', 'delete', bot.name, '--yes']
    })

    if (result?.blocked || result?.code !== 0) {
      throw new Error(result?.hint || result?.output || `Could not delete profile ${bot.name}.`)
    }
  }

  const meta = {
    ...$botMeta.get()
  }

  delete meta[botMetaKey(bot)]
  $botMeta.set(meta)

  try {
    if (route) {
      await commitBotMetaV2(getPluginCtx()?.storage, meta)
    } else {
      await Promise.resolve(getPluginCtx()?.storage?.set?.(BOT_META_V1_KEY, meta))
    }
  } catch {
    /* profile is deleted; stale local appearance is harmless if storage fails */
  }

  // Both ids: the marker may have been filed under either tip of the chat's
  // compression lineage.
  forgetSessionUnread([bot.canonical_session?.id, bot.canonical_session?.resolved_id], bot.name)
  rosterWatermarks.delete(botSelectionKey(bot))
  avatarFetchInflight.delete(botMetaKey(bot))
  avatarPushInflight.delete(botMetaKey(bot))

  if ($selectedBot.get() === botSelectionKey(bot)) {
    $selectedBot.set('default')
  }

  clearSelectedRosterBot(bot)

  if ($openBotChat.get()?.key === botRosterKey(bot)) {
    $openBotChat.set(null)
  }

  queryClient.invalidateQueries({
    queryKey: ROSTER_KEY
  })
  const activeOwner = focusedRosterOwner($focusedBotOwner.get?.())

  const deletedOwnerIsActive = route
    ? activeOwner?.connectionId === route.connectionId && activeOwner?.name === route.profile
    : activeOwner?.name === bot.name

  if (deletedOwnerIsActive && typeof host.newChat === 'function') {
    host.newChat('default')
  }
}
