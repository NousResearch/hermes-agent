import { normalizeProfileKey } from './profile'
import { TILE_PANE_PREFIX } from './session-focus'
import type { SessionTile } from './session-states'

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

export function parseTileList(value: unknown): StoredTile[] {
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
