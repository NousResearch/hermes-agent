/**
 * Persistence for session tiles (the chat area's tabs). Tabs are stored per
 * backend connection and per profile VIEW — a concrete profile, or "All
 * profiles" — so each view brings back its own tabs.
 *
 * Every window shares this localStorage area. Writes merge the buckets this
 * window changed into what is on disk instead of replacing the whole map, so
 * a second window (peer, HUD) cannot erase tabs it never saw.
 */
import { backendScopeKey } from '@hermes/shared'

import type { HermesConnection } from '@/global'
import { readJson, writeJson } from '@/lib/storage'

import { ALL_PROFILES, normalizeProfileKey } from './profile'
import { TILE_PANE_PREFIX } from './session-focus'
import type { SessionTile } from './session-states'
import { isBrowserWindow, isHudWindow, isMainWindow, isSecondaryWindow } from './windows'

// Tiles are persisted per connection and profile view: same-named profiles on
// two backends own different sessions. Switching either scope swaps the
// visible set, with runtime bindings dropped so tiles re-resume on their own
// gateway.
const TILES_KEY = 'hermes.desktop.sessionTiles.v2'
const LEGACY_TILES_KEY = 'hermes.desktop.sessionTiles.v1'
export const BOTS_TILE_BUCKET = '__bots_workspace__'

/** Persisted placement — `dir` + strip slot (`before`) + dock `anchor` so a
 *  restart / profile swap re-adopts tiles in the same order, not all stacked
 *  right of workspace. */
export type StoredTile = Pick<
  SessionTile,
  | 'anchor'
  | 'before'
  | 'dir'
  | 'ownerProfile'
  | 'ownerRoute'
  | 'storedSessionId'
  | 'workspaceMode'
  | 'workspaceOwnerKey'
  | 'workspaceTabTitle'
>

export const toStored = (t: SessionTile): StoredTile => ({
  anchor: t.anchor,
  before: t.before,
  dir: t.dir,
  ...(t.ownerProfile ? { ownerProfile: t.ownerProfile } : {}),
  ...(t.ownerRoute ? { ownerRoute: t.ownerRoute } : {}),
  storedSessionId: t.storedSessionId,
  ...(t.workspaceMode ? { workspaceMode: t.workspaceMode } : {}),
  ...(t.workspaceOwnerKey ? { workspaceOwnerKey: t.workspaceOwnerKey } : {}),
  ...(t.workspaceTabTitle ? { workspaceTabTitle: t.workspaceTabTitle } : {})
})

function parseTileList(value: unknown): StoredTile[] {
  return Array.isArray(value)
    ? value
        .filter((t): t is SessionTile => Boolean(t && typeof (t as SessionTile).storedSessionId === 'string'))
        .map(t => {
          const raw = t as SessionTile

          return {
            // #108679: a tile whose anchor is its OWN pane id is
            // self-referential — the re-dock target can never exist (the
            // pane is not in the tree at adoption time), so the dock falls
            // through to an arbitrary same-placement neighbor instead of the
            // recorded layout. Rewrite it to the workspace anchor at load,
            // the same surface an anchorless tile re-docks against.
            anchor:
              typeof raw.anchor === 'string' && raw.anchor !== `${TILE_PANE_PREFIX}${raw.storedSessionId}`
                ? raw.anchor
                : undefined,
            before: typeof raw.before === 'string' || raw.before === null ? raw.before : undefined,
            dir: raw.dir,
            ownerProfile: typeof raw.ownerProfile === 'string' ? normalizeProfileKey(raw.ownerProfile) : undefined,
            ownerRoute:
              raw.ownerRoute &&
              typeof raw.ownerRoute.connectionId === 'string' &&
              typeof raw.ownerRoute.profile === 'string'
                ? {
                    connectionId: raw.ownerRoute.connectionId,
                    mode: raw.ownerRoute.mode,
                    profile: raw.ownerRoute.profile,
                    ...(typeof raw.ownerRoute.targetProfile === 'string'
                      ? { targetProfile: raw.ownerRoute.targetProfile }
                      : {})
                  }
                : undefined,
            storedSessionId: raw.storedSessionId,
            workspaceMode: raw.workspaceMode === 'bots' ? 'bots' : 'sessions',
            workspaceOwnerKey:
              raw.workspaceMode === 'bots' && typeof raw.workspaceOwnerKey === 'string'
                ? raw.workspaceOwnerKey
                : undefined,
            workspaceTabTitle: typeof raw.workspaceTabTitle === 'string' ? raw.workspaceTabTitle : undefined
          }
        })
    : []
}

export function loadTilesByProfile(): Record<string, StoredTile[]> {
  const byProfile: Record<string, StoredTile[]> = Object.create(null)
  const parsed = readJson<unknown>(TILES_KEY)

  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    for (const [profile, list] of Object.entries(parsed as Record<string, unknown>)) {
      const tiles = parseTileList(list)
      // Existing profile-only buckets belong to the local connection. New
      // remote buckets carry the same backend scope key as the socket pool;
      // never collapse them back into a same-named local profile on reload.
      const separator = profile.lastIndexOf('::')
      const isScopedBucket = profile.startsWith('conn:') && separator > 'conn:'.length

      const key =
        profile === BOTS_TILE_BUCKET
          ? BOTS_TILE_BUCKET
          : isScopedBucket
            ? backendScopeKey(
                profile.slice('conn:'.length, separator),
                normalizeProfileKey(profile.slice(separator + 2))
              )
            : normalizeProfileKey(profile)

      if (tiles.length > 0) {
        const sessionTiles = tiles.filter(tile => tile.workspaceMode !== 'bots')
        const botTiles = tiles.filter(tile => tile.workspaceMode === 'bots')

        for (const tile of sessionTiles) {
          // Old profile-only buckets can contain remote tiles whose owner was
          // already recorded. Preserve those tabs on upgrade; unknown owners
          // stay local rather than being guessed onto another backend.
          const scope = isScopedBucket ? key : backendScopeKey(tile.ownerRoute?.connectionId, key)
          byProfile[scope] = [...(byProfile[scope] ?? []), tile]
        }

        if (botTiles.length > 0) {
          byProfile[BOTS_TILE_BUCKET] = [...(byProfile[BOTS_TILE_BUCKET] ?? []), ...botTiles]
        }
      }
    }
  }

  // Migrate a v1 flat list into the default profile, then retire the key.
  const legacy = parseTileList(readJson<unknown>(LEGACY_TILES_KEY))

  if (legacy.length > 0) {
    const key = normalizeProfileKey('default')
    const sessionTiles = legacy.filter(tile => tile.workspaceMode !== 'bots')
    const botTiles = legacy.filter(tile => tile.workspaceMode === 'bots')

    byProfile[key] = [...(byProfile[key] ?? []), ...sessionTiles]
    byProfile[BOTS_TILE_BUCKET] = [...(byProfile[BOTS_TILE_BUCKET] ?? []), ...botTiles]
  }

  if (byProfile[BOTS_TILE_BUCKET]?.length) {
    byProfile[BOTS_TILE_BUCKET] = [
      ...new Map(byProfile[BOTS_TILE_BUCKET].map(tile => [tile.storedSessionId, tile])).values()
    ]
  }

  writeJson(LEGACY_TILES_KEY, null)

  return byProfile
}

export const tileConnectionScopeId = (connection: HermesConnection | null): null | string => {
  const id = connection?.connectionId?.trim()

  if (id) {
    return id
  }

  // Older direct remotes have no registry id. Keep them separate from local
  // and from each other instead of writing into the local profile bucket.
  return connection?.mode === 'remote' ? `url:${connection.baseUrl || 'remote'}` : null
}

const TILE_CONNECTION_KEY = 'hermes.desktop.sessionTiles.connection'
const TILE_ACTIVE_KEY = 'hermes.desktop.sessionTiles.active.v1'
const TILE_VIEWS_SEEDED_KEY = 'hermes.desktop.sessionTiles.viewsSeeded'
export const VIEW_MAIN_KEY = 'hermes.desktop.sessionTiles.viewMain.v1'

/** Per-view memory keyed by tile scope: front panes and the main tab's chat. */
const VIEW_MEMORY_KEYS = [TILE_ACTIVE_KEY, VIEW_MAIN_KEY] as const

function rewriteViewMemory(edit: (stored: Record<string, unknown>) => void): void {
  if (!isTileStoreWriter()) {
    return
  }

  for (const key of VIEW_MEMORY_KEYS) {
    const raw = readJson<unknown>(key)

    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      continue
    }

    const stored: Record<string, unknown> = { ...(raw as Record<string, unknown>) }
    edit(stored)
    writeJson(key, Object.keys(stored).length === 0 ? null : stored)
  }
}

/** A deleted profile's view memory goes with its tabs, or a later profile of
 *  the same name opens on the old one's front tabs and main chat. */
export function dropViewMemory(scope: string): void {
  rewriteViewMemory(stored => {
    delete stored[scope]
  })
}

/** A renamed profile keeps its view memory under the new name. */
export function migrateViewMemory(from: string, to: string): void {
  rewriteViewMemory(stored => {
    if (from in stored) {
      stored[to] ??= stored[from]
      delete stored[from]
    }
  })
}

/** Only the window that owns the tab strip writes tab state. A secondary /
 *  browser pop-out holds no tiles and the HUD is a full renderer that never
 *  shows the strip; a write from either would only replace the primary's set. */
export const isTileStoreWriter = (): boolean => !isSecondaryWindow() && !isBrowserWindow() && !isHudWindow()

/** Write only `buckets` from `byProfile`, merged into what is on disk now.
 *  A bucket missing from `byProfile` is removed. */
export function persistTileBuckets(byProfile: Record<string, StoredTile[]>, buckets: readonly string[]): void {
  if (!isTileStoreWriter() || buckets.length === 0) {
    return
  }

  const raw = readJson<unknown>(TILES_KEY)
  const stored: Record<string, unknown> = raw && typeof raw === 'object' && !Array.isArray(raw) ? { ...raw } : {}

  for (const bucket of new Set(buckets)) {
    if (byProfile[bucket]?.length) {
      stored[bucket] = byProfile[bucket]
    } else {
      delete stored[bucket]
    }
  }

  writeJson(TILES_KEY, Object.keys(stored).length === 0 ? null : stored)
}

/** Replace the stored map with the normalized in-memory one. Run once at load
 *  so later merge writes never meet legacy (un-normalized) keys on disk. */
export function persistAllTileBuckets(byProfile: Record<string, StoredTile[]>): void {
  if (isTileStoreWriter()) {
    writeJson(TILES_KEY, Object.keys(byProfile).length === 0 ? null : byProfile)
  }
}

/** The bucket as another window may have left it. `undefined` = not stored. */
export function readStoredTileBucket(bucket: string): StoredTile[] | undefined {
  const raw = readJson<Record<string, unknown>>(TILES_KEY)

  return raw && typeof raw === 'object' && bucket in raw ? parseTileList(raw[bucket]) : undefined
}

/** Connection whose tabs were last on screen. Boot shows them before the
 *  (remote) connection publishes; otherwise the strip starts on the local
 *  bucket and tabs opened during boot land in the wrong one. */
export function rememberedTileConnectionId(): null | string {
  const stored = readJson<{ id?: unknown }>(TILE_CONNECTION_KEY)

  return typeof stored?.id === 'string' && stored.id ? stored.id : null
}

/** Main window only: a peer window on another backend must not decide which
 *  tabs the main window boots on. */
export function rememberTileConnectionId(id: null | string): void {
  if (isMainWindow() && rememberedTileConnectionId() !== id) {
    writeJson(TILE_CONNECTION_KEY, id ? { id } : null)
  }
}

/** Front pane of each tab zone, per view, so coming back to a view fronts the
 *  same tab in every zone. */
export function rememberedActivePanes(scope: string): string[] {
  const stored = readJson<Record<string, unknown>>(TILE_ACTIVE_KEY)
  const panes = stored && typeof stored === 'object' ? stored[scope] : undefined

  return Array.isArray(panes) ? panes.filter((pane): pane is string => typeof pane === 'string') : []
}

export function rememberActivePanes(scope: string, panes: readonly string[]): void {
  if (!isTileStoreWriter()) {
    return
  }

  const raw = readJson<Record<string, unknown>>(TILE_ACTIVE_KEY)
  const stored: Record<string, unknown> = raw && typeof raw === 'object' && !Array.isArray(raw) ? { ...raw } : {}

  if (panes.length > 0) {
    stored[scope] = [...panes]
  } else {
    delete stored[scope]
  }

  writeJson(TILE_ACTIVE_KEY, Object.keys(stored).length === 0 ? null : stored)
}

/** Before views, "All profiles" showed whatever bucket the live gateway
 *  profile owned. Seed each connection's All bucket from that bucket once, so
 *  an upgrade does not open All with an empty strip. */
export function seedAllProfilesBuckets(
  byProfile: Record<string, StoredTile[]>,
  gatewayProfile: string,
  showAllProfiles: boolean
): void {
  if (!isTileStoreWriter() || readJson<unknown>(TILE_VIEWS_SEEDED_KEY) !== null) {
    return
  }

  writeJson(TILE_VIEWS_SEEDED_KEY, 1)

  if (!showAllProfiles) {
    return
  }

  const profile = normalizeProfileKey(gatewayProfile)

  for (const [bucket, tiles] of Object.entries(byProfile)) {
    const separator = bucket.lastIndexOf('::')
    const isScoped = bucket.startsWith('conn:') && separator > 'conn:'.length
    const bucketProfile = isScoped ? bucket.slice(separator + 2) : bucket
    const connection = isScoped ? bucket.slice('conn:'.length, separator) : null
    const allBucket = backendScopeKey(connection, ALL_PROFILES)

    if (bucket !== BOTS_TILE_BUCKET && bucketProfile === profile && !byProfile[allBucket]) {
      byProfile[allBucket] = [...tiles]
    }
  }
}
