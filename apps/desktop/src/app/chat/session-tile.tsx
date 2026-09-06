/**
 * SESSION TILES — a stored session rendered as a layout-tree pane BESIDE the
 * main thread (multi-session tiling). A tile IS the real chat surface: the
 * same ChatView/ChatBar/Thread tree the primary session renders, mounted
 * under a tile `SessionView` (its session's slice of `$sessionStates`) and a
 * tile `ComposerScope` (own attachment chips, own focus-bus key). Actions
 * (submit/slash/steer/edit/reload/restore/stop) come from
 * `useSessionTileActions`, all writing through the wiring cache.
 *
 * Lifecycle: `openSessionTile(storedId)` -> `watchSessionTiles` registers a
 * pane contribution docked right of the main zone -> tree adoption lands it
 * -> the pane mounts and asks the delegate for a live runtime id. Closing
 * the pane (tab Close) removes the tile + its zone; tiles persist across
 * restarts and re-resume on boot.
 */

import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from 'react'

import { resolveStoredSession } from '@/app/session/hooks/use-session-actions/utils'
import { CenteredThreadSpinner } from '@/components/assistant-ui/thread/status'
import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, closeTreePane, moveTreePane, setTreeGroupTabStrip } from '@/components/pane-shell/tree/store'
import { $workspaceOwnerLabels, workspaceOwnerTitle } from '@/components/pane-shell/workspace-scope'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import type { ChatMessage } from '@/lib/chat-messages'
import { NEW_SESSION_TITLE, sessionTitle } from '@/lib/chat-runtime'
import { draftTitleFor } from '@/store/composer'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $projectTree } from '@/store/projects'
import {
  $cronSessions,
  $gatewayState,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  sessionPinId
} from '@/store/session'
import { isSessionRemovalPending } from '@/store/session-removal'
import {
  $sessionStates,
  $sessionTileDelegateRevision,
  $sessionTiles,
  closeSessionTile,
  patchSessionTile,
  type SessionTile,
  sessionTileDelegate
} from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import type { SessionDragPayload } from './composer/inline-refs'
import { paneMirror } from './pane-mirror'
import { SessionChat } from './session-chat'
import { SessionDraftTitle } from './session-draft-title'
import { startSessionDrag } from './session-drag'
import { SessionStatusDot } from './session-status-dot'
import { tileOwnerRoute } from './session-tile-owner'
import { buildSessionView, type SessionView } from './session-view'
import { SessionContextMenu } from './sidebar/session-actions-menu'


const NO_MESSAGES: ChatMessage[] = []

export function sessionTileResumeFailure(
  message: string,
  durableSessionFound: boolean | undefined,
  tileStillUnbound: boolean
): string | undefined {
  if (!tileStillUnbound) {
    return undefined
  }

  if (!/session not found|\b404\b/i.test(message)) {
    return message
  }

  if (durableSessionFound) {
    return 'Session is still available — retry resuming it.'
  }

  return 'Session unavailable — you can retry resuming it.'
}

/** Should this tile dispatch a `session.resume`?
 *
 *  - The gateway must be OPEN: persisted tiles mount at boot while it is still
 *    connecting, and an ungated resume rejected there latched every restored
 *    tile into the error card.
 *  - A bound runtime, a latched error, or an in-flight attempt means there is
 *    nothing to do.
 *  - A removal-pending session is skipped for the same reason the primary's
 *    `resumeSession` skips it: a 4001 racing a delete unbinds this tile's
 *    runtime and re-arms the effect against an id that is already gone. The
 *    resume would 404 and latch an error card for a chat the user deleted;
 *    `closeSessionTile` lands moments later. */
export function shouldResumeSessionTile(opts: {
  gatewayOpen: boolean
  removalPending: boolean
  resuming: boolean
  runtimeId: null | string | undefined
  tileError: string | undefined
}): boolean {
  return !opts.removalPending && opts.gatewayOpen && !opts.runtimeId && !opts.tileError && !opts.resuming
}

/** The tile's SessionView: the shared non-primary shape, with the runtime id
 *  read from the tile registry. */
const buildTileView = (storedSessionId: string): SessionView =>
  buildSessionView(
    'tile',
    computed($sessionTiles, tiles => tiles.find(t => t.storedSessionId === storedSessionId)?.runtimeId ?? null),
    storedSessionId
  )

/** The tile's chat is the shared one; the tile only owns where it lives, its
 *  owner route, and how it resumes. */
function TileChat({ runtimeId, storedSessionId, view }: { runtimeId: string; storedSessionId: string; view: SessionView }) {
  // Owner ladder, same as useSessionTileActions (session-tile-actions.ts:99-103).
  // Recomputed when the tile store or any owner-bearing session list changes,
  // NOT on every render: this component re-renders per streamed token, and the
  // lookup spreads three arrays before scanning them.
  const tiles = useStore($sessionTiles)
  const sessionRows = useStore($sessions)
  const cronRows = useStore($cronSessions)
  const messagingRows = useStore($messagingSessions)

  const ownerRoute = useMemo(() => {
    const rows = cronRows.length || messagingRows.length ? [...sessionRows, ...cronRows, ...messagingRows] : sessionRows

    return tileOwnerRoute(tiles, rows, storedSessionId)
  }, [cronRows, messagingRows, sessionRows, storedSessionId, tiles])

  const onRetryResume = useCallback(() => patchSessionTile(storedSessionId, { error: undefined }), [storedSessionId])

  const onRuntimeBound = useCallback(
    (recovered: string) => patchSessionTile(storedSessionId, { error: undefined, runtimeId: recovered }),
    [storedSessionId]
  )

  return (
    <SessionChat
      onRetryResume={onRetryResume}
      onRuntimeBound={onRuntimeBound}
      ownerRoute={ownerRoute}
      runtimeId={runtimeId}
      storedSessionId={storedSessionId}
      view={view}
    />
  )
}

export function SessionTilePane({ storedSessionId }: { storedSessionId: string }) {
  const tiles = useStore($sessionTiles)
  const tile = tiles.find(t => t.storedSessionId === storedSessionId)
  const ownerRoute = tile?.ownerRoute
  const runtimeId = tile?.runtimeId ?? null
  const gatewayOpen = useStore($gatewayState) === 'open'
  const delegateRevision = useStore($sessionTileDelegateRevision)
  const resumingRef = useRef(false)
  const view = useMemo(() => buildTileView(storedSessionId), [storedSessionId])

  const storedSessionStillExists = useCallback(
    () => $sessions.get().some(s => sessionMatchesStoredId(s, storedSessionId)),
    [storedSessionId]
  )

  // A tab-strip "+"/⌘T tab is created UNLISTED — its session stays out of
  // $sessions (no sidebar clutter) until it's actually used, so the tab shows
  // "New session". The moment this tile has a message, pull its row into
  // $sessions via the lightweight by-id lookup so the tab (and a sidebar row)
  // resolve the real title. `resolveStoredSession` no-ops when it's already
  // listed, and 404s harmlessly for an in-memory draft that hasn't persisted a
  // turn yet — so we retry across that brief persist lag and stop as soon as it
  // lands (a global turn-complete refresh may beat us to it).
  const hasMessages = useStore(view.$messagesEmpty) === false

  useEffect(() => {
    const alreadyListed = storedSessionStillExists

    if (!runtimeId || !hasMessages || alreadyListed()) {
      return
    }

    let cancelled = false
    let timer: number | undefined

    const attempt = (remaining: number) => {
      if (cancelled || alreadyListed()) {
        return
      }

      void resolveStoredSession(storedSessionId, ownerRoute)
        .then(resolved => {
          if (cancelled || resolved || remaining <= 0) {
            return
          }

          timer = window.setTimeout(() => attempt(remaining - 1), 500)
        })
        .catch(() => undefined)
    }

    attempt(6)

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [hasMessages, ownerRoute, runtimeId, storedSessionId, storedSessionStillExists])

  // Gating lives in shouldResumeSessionTile (unit-tested there).
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (
      !shouldResumeSessionTile({
        gatewayOpen,
        removalPending: isSessionRemovalPending(storedSessionId),
        resuming: resumingRef.current,
        runtimeId,
        tileError: tile?.error
      })
    ) {
      return
    }

    const delegate = sessionTileDelegate()

    if (!delegate) {
      return
    }

    resumingRef.current = true

    delegate
      .resumeTile(storedSessionId)
      .then(id => patchSessionTile(storedSessionId, { error: undefined, runtimeId: id }))
      .catch(async (err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)

        if (!/session not found|\b404\b/i.test(message)) {
          patchSessionTile(storedSessionId, { error: message })

          return
        }

        // A recents page is not authoritative, and resolveStoredSession()
        // intentionally treats transient probe failures like a miss. Await it
        // before releasing this resume attempt, then fail safe: a tile may be
        // retried by the user, but must never be deleted on an inconclusive
        // reconnect-time lookup.
        const durableSession = await resolveStoredSession(storedSessionId, ownerRoute).catch(() => undefined)
        const current = $sessionTiles.get().find(candidate => candidate.storedSessionId === storedSessionId)
        const error = sessionTileResumeFailure(message, Boolean(durableSession), Boolean(current && !current.runtimeId))

        if (error) {
          patchSessionTile(storedSessionId, { error })
        }
      })
      .finally(() => {
        resumingRef.current = false
      })
  }, [delegateRevision, gatewayOpen, ownerRoute, runtimeId, storedSessionId, tile?.error])

  // The gateway (re)opening invalidates any latched error — it likely came
  // from a not-yet-open gateway or the previous connection. Clearing it
  // retriggers the resume effect: one bounded auto-retry per (re)connect,
  // mirroring the primary path's became-open resync.
  useEffect(() => {
    if (gatewayOpen && tile?.error) {
      patchSessionTile(storedSessionId, { error: undefined })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gatewayOpen, storedSessionId])

  if (tile?.error) {
    return (
      <div className="grid h-full place-items-center p-4">
        <div className="max-w-[24rem] space-y-2 text-center font-mono text-[11px]">
          <div className="text-(--ui-danger,#f87171)">Couldn't open this session</div>
          <div className="break-words text-(--ui-text-quaternary)">{tile.error}</div>
          <Button onClick={() => patchSessionTile(storedSessionId, { error: undefined })} size="sm" variant="outline">
            Retry
          </Button>
        </div>
      </div>
    )
  }

  if (!runtimeId) {
    // The SAME session loader the primary thread shows (Thread's
    // loading === 'session' branch) — one loading language everywhere.
    return (
      <div className="relative h-full">
        <CenteredThreadSpinner />
      </div>
    )
  }

  return <TileChat runtimeId={runtimeId} storedSessionId={storedSessionId} view={view} />
}

// ---------------------------------------------------------------------------
// Tile -> pane contribution sync (call once from the app root).
// ---------------------------------------------------------------------------

/** Resolve a tile's stored row: the recents list first, then the project
 *  tree. A session opened as a tab from a project group is often older than
 *  the paginated recents page, so it has no `$sessions` row at all until new
 *  activity lands it there — resolving through the tree keeps its tab titled
 *  and tinted instead of a grey "Session" placeholder. */
export function tileStoredRow(storedSessionId: string): SessionInfo | undefined {
  const match = (s: SessionInfo) => sessionMatchesStoredId(s, storedSessionId)

  return (
    $sessions.get().find(match) ??
    $projectTree
      .get()
      .flatMap(p => [...p.repos.flatMap(r => r.groups.flatMap(g => g.sessions)), ...(p.previewSessions ?? [])])
      .find(match)
  )
}

/** One-shot by-id title fill for restored tiles that never mount (#94167).
 *  A restored background tab has no runtimeId and does not mount its pane, so
 *  the resolution effect above never runs; when its row is outside the recents
 *  page and project tree, `tileTitle()` reads "New session" until first click.
 *  `resolveStoredSession` upserts the row into `$sessions`, which the tab strip
 *  already watches — nothing is persisted. Runs once the gateway can answer. */
export function startUnrestoredTileTitleBackfill(lookup = resolveStoredSession): () => void {
  const run = () => {
    if ($gatewayState.get() !== 'open') {
      return
    }

    off()

    for (const tile of $sessionTiles.get()) {
      if (!tile.runtimeId && !tile.workspaceTabTitle && !tileStoredRow(tile.storedSessionId)) {
        void lookup(tile.storedSessionId, tile.ownerRoute).catch(() => undefined)
      }
    }
  }

  const off = $gatewayState.listen(run)
  run()

  return off
}

/** The tab's REGISTERED name. Deliberately the bare placeholder for a draft
 *  rather than its live composer title (`tabTitle` renders that): re-registering
 *  per keystroke would re-render the strip, and holding the draft's text here
 *  would let the registered name already match the row that lands on send —
 *  skipping the re-register that hands the tab back to this string. */
function tileTitle(storedSessionId: string): string {
  const stored = tileStoredRow(storedSessionId)
  const explicit = $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)?.workspaceTabTitle

  return stored ? sessionTitle(stored) : explicit || NEW_SESSION_TITLE
}

/** The tab's CAPTION: a bot chat's owner name over the canonical stored title
 *  (#99152). The menu keeps `tileTitle` — rename/delete show the real row. */
function tileCaption(storedSessionId: string): string {
  return workspaceOwnerTitle(
    tileTitle(storedSessionId),
    $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)
  )
}

/** The `@session` link payload for a tile tab drag — id + owning profile + title.
 *  Resolved at drag time, so an unsent tab drags under its draft name. */
function tileDragPayload(storedSessionId: string): SessionDragPayload {
  const stored = tileStoredRow(storedSessionId)
  const tile = $sessionTiles.get().find(candidate => candidate.storedSessionId === storedSessionId)

  const title = stored
    ? sessionTitle(stored)
    : tile?.workspaceTabTitle || draftTitleFor(storedSessionId) || NEW_SESSION_TITLE

  return { id: storedSessionId, profile: stored?.profile ?? '', title: workspaceOwnerTitle(title, tile) }
}

// ---------------------------------------------------------------------------
// Close confirmation — a BUSY tab (streaming, or blocked on clarify/approval
// input) doesn't close silently.
// ---------------------------------------------------------------------------

/** Stored id awaiting close confirmation (null = no dialog). */
const $confirmCloseTile = atom<null | string>(null)

/** The tile closer, gated: a quiet session closes immediately; a busy or
 *  input-blocked one asks first. One state read — the tile's runtime slice. */
export function requestCloseSessionTile(storedSessionId: string): void {
  const runtimeId = $sessionTiles.get().find(t => t.storedSessionId === storedSessionId)?.runtimeId
  const state = runtimeId ? $sessionStates.get()[runtimeId] : undefined

  if (state?.busy || state?.awaitingResponse || state?.needsInput) {
    $confirmCloseTile.set(storedSessionId)
  } else {
    closeSessionTile(storedSessionId)
  }
}

/** Mounted once at the shell root: the "Close running tab?" confirmation. */
export function SessionTileCloseConfirm() {
  const { t } = useI18n()
  const storedSessionId = useStore($confirmCloseTile)

  return (
    <ConfirmDialog
      confirmLabel={t.zones.closeRunningConfirm}
      description={t.zones.closeRunningBody}
      destructive
      onClose={() => $confirmCloseTile.set(null)}
      onConfirm={() => {
        if (storedSessionId) {
          closeSessionTile(storedSessionId)
        }
      }}
      open={storedSessionId !== null}
      title={t.zones.closeRunningTitle}
    />
  )
}

/** Layout reset → every session tile collapses into the MAIN zone as a tab
 *  after the workspace (the primary session stays the first tab), the "smart"
 *  reset: N scattered tiles become one tab bar over the chat instead of
 *  re-docking to their old edges.
 *
 *  Runs BEFORE generic adoption (see registerLayoutResetHandler) — the tiles
 *  aren't in the fresh tree yet, so each `moveTreePane` ADDS the tile into the
 *  workspace group as a tab (append). The main group id is re-read each pass
 *  because appending returns a new tree. */
export function stackSessionTilesIntoMain(): void {
  for (const tile of $sessionTiles.get()) {
    const tree = $layoutTree.get()
    const mainGroup = tree ? findGroupOfPane(tree, 'workspace')?.id : null

    if (mainGroup) {
      moveTreePane(`session-tile:${tile.storedSessionId}`, { groupId: mainGroup, pos: 'center' })
    }
  }
}

/** The three scalars the tab menu actually renders, derived from the stored
 *  row. Subscribing to `$sessions` + `$projectTree` wholesale re-rendered
 *  every tab's menu wrapper on ANY session-list or tree churn (polls, title
 *  updates in other sessions) — for a context menu that's almost never open.
 *  Same class as the TreeGroup fix (#72245): derive narrowly, bail out unless
 *  the derived values change. */
function useTileMenuRow(storedSessionId: string): { pinId: string; profile?: string; title: string } {
  const cache = useRef<{ key: string; value: { pinId: string; profile?: string; title: string } } | null>(null)

  const subscribe = useCallback((onChange: () => void) => {
    const offSessions = $sessions.listen(onChange)
    const offTree = $projectTree.listen(onChange)

    return () => {
      offSessions()
      offTree()
    }
  }, [])

  return useSyncExternalStore(subscribe, () => {
    const stored = tileStoredRow(storedSessionId)
    const pinId = stored ? sessionPinId(stored) : storedSessionId
    const title = tileTitle(storedSessionId)
    const profile = stored?.profile
    const key = `${pinId}\u0000${title}\u0000${profile ?? ''}`

    if (cache.current?.key !== key) {
      cache.current = { key, value: { pinId, profile, title } }
    }

    return cache.current.value
  })
}

/** A session TAB's context menu: the full session verb set (pin, copy id, new
 *  window, branch, rename, archive, delete) — the SAME menu a sidebar row
 *  gets, targeted through the tile delegate (whose verbs are generic over
 *  stored ids, primary included). The wrapper stops the contextmenu from also
 *  opening the zone strip's menu. Shared by tile tabs AND the main tab. */
export function SessionTabMenu({
  children,
  onClose,
  onHideTabBar,
  storedSessionId,
  tabPaneId
}: {
  children: React.ReactElement
  /** Close this tab (tiles; the main tab passes nothing). */
  onClose?: () => void
  /** Hide the zone's tab bar (main tab only — the sticky bar's off switch). */
  onHideTabBar?: () => void
  storedSessionId: string
  /** Layout-tree pane id — powers the Close-others/right/all verbs. */
  tabPaneId: string
}) {
  const { pinId, profile, title } = useTileMenuRow(storedSessionId)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const pinned = pinnedSessionIds.includes(pinId)

  return (
    <span className="contents" onContextMenu={event => event.stopPropagation()}>
      <SessionContextMenu
        onArchive={() => void sessionTileDelegate()?.archiveSession(storedSessionId)}
        onBranch={() => void sessionTileDelegate()?.branchSession(storedSessionId)}
        onClose={onClose}
        onDelete={() => void sessionTileDelegate()?.deleteSession(storedSessionId)}
        onHideTabBar={onHideTabBar}
        onPin={() => (pinned ? unpinSession(pinId) : pinSession(pinId))}
        pinned={pinned}
        profile={profile}
        sessionId={storedSessionId}
        surface="tab"
        tabPaneId={tabPaneId}
        title={title}
      >
        {children}
      </SessionContextMenu>
    </span>
  )
}

/** The MAIN tab's menu: the same session verbs targeting the primary's loaded
 *  session, plus Close (the tab empties to a fresh draft — the workspace pane
 *  itself never leaves the tree) and the bar's off switch (the bar sticky-shows
 *  once a tab is ever gained; this is the explicit way back). A fresh draft has
 *  no session — no menu. */
export function WorkspaceTabMenu({ children }: { children: React.ReactElement }) {
  const selected = useStore($selectedStoredSessionId)

  const hideTabBar = () => {
    const tree = $layoutTree.get()
    const group = tree ? findGroupOfPane(tree, 'workspace') : null

    if (group) {
      setTreeGroupTabStrip(group.id, 'never')
    }
  }

  if (!selected) {
    return children
  }

  return (
    <SessionTabMenu
      onClose={() => closeTreePane('workspace')}
      onHideTabBar={hideTabBar}
      storedSessionId={selected}
      tabPaneId="workspace"
    >
      {children}
    </SessionTabMenu>
  )
}

/** Keep pane contributions mirroring `$sessionTiles` (+ titles from
 *  `$sessions`). Tiles dock against main on the chosen edge, flex width. */
export const watchSessionTiles = paneMirror<SessionTile>({
  source: $sessionTiles,
  // $projectTree: a tile whose session is older than the recents page resolves
  // its title through the tree, which loads after the tiles register. (The tab's
  // status dot subscribes to color/state itself, so it needs no `also` entry.)
  also: [$sessions, $projectTree, $workspaceOwnerLabels],
  key: t => t.storedSessionId,
  prefix: 'session-tile',
  dir: t => t.dir,
  anchor: t => t.anchor,
  before: t => t.before,
  minWidth: '20rem',
  title: tileCaption,
  // The tab's status dot — the SAME primitive the sidebar row renders, keyed by
  // the stored id, so a session's status/color can never disagree between the
  // two surfaces. Self-subscribing (live state + resolved color), so the strip
  // needn't re-sync when it changes.
  tabLead: storedSessionId => (
    <SessionStatusDot session={tileStoredRow(storedSessionId)} storedSessionId={storedSessionId} />
  ),
  // Until the first turn lists a row there is no title to register, so the tab
  // takes its name from the composer instead — live, without re-registering.
  tabTitle: storedSessionId =>
    tileStoredRow(storedSessionId) ||
    $sessionTiles.get().some(tile => tile.storedSessionId === storedSessionId && tile.workspaceTabTitle) ? null : (
      <SessionDraftTitle scope={storedSessionId} />
    ),
  render: storedSessionId => <SessionTilePane storedSessionId={storedSessionId} />,
  tabWrap: (storedSessionId, tab) => (
    <SessionTabMenu
      onClose={() => requestCloseSessionTile(storedSessionId)}
      storedSessionId={storedSessionId}
      tabPaneId={`session-tile:${storedSessionId}`}
    >
      {tab}
    </SessionTabMenu>
  ),
  // A tile's tab drags like a sidebar row — stack / split / drop-to-link — with
  // its tap (activate) preserved. Always takes the drag.
  tabDrag: (storedSessionId, event, onTap) => {
    startSessionDrag(tileDragPayload(storedSessionId), event, { onTap })

    return true
  },
  close: requestCloseSessionTile
})
