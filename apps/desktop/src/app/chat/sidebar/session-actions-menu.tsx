import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { Fragment, useEffect, useRef, useState } from 'react'

import { stampHueClass } from '@/app/chat/session-stamp'
import { openSession } from '@/app/open-session'
import {
  closeAllTreeTabs,
  closeOtherTreeTabs,
  closeTreeTabsToRight,
  reloadTreePane,
  treeTabCloseTargets
} from '@/components/pane-shell/tree/store'
import {
  type ActionItemSpec,
  ActionsContextMenu,
  ActionsMenu,
  type MenuKit,
  renderActionItem
} from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ColorSwatches } from '@/components/ui/color-swatches'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { DropdownMenuSearch } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { renameSession } from '@/hermes'
import { useI18n } from '@/i18n'
import { searchEmoji } from '@/lib/emoji-index'
import { triggerHaptic } from '@/lib/haptics'
import { isSubmitEnter } from '@/lib/ime'
import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { exportSession } from '@/lib/session-export'
import { cn } from '@/lib/utils'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $projectTree, moveSessionToProject, projectIdForCwd, projectRootCwd } from '@/store/projects'
import {
  $activeSessionId,
  $connection,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  markSessionRead,
  sessionMatchesStoredId,
  sessionPinId,
  setSessions
} from '@/store/session'
import { $sessionColorOverrides, setSessionColorOverride } from '@/store/session-color'
import {
  $deletedStampPresets,
  $stampColorOverrides,
  $stampPresets,
  addStampTitle,
  applySessionStamps,
  deleteStampPreset,
  hasStampLabel,
  isEmojiStamp,
  normalizeSessionStamp,
  normalizeSessionStamps,
  restoreStampPresets,
  SESSION_STAMP_EMOJI,
  SESSION_STAMP_EMOJI_SEARCH_LIMIT,
  SESSION_STAMP_LIMIT,
  SESSION_STAMP_MAX_LENGTH,
  setStampColor,
  STAMP_SWATCHES,
  stampColorFor,
  stampLabels,
  toggleSessionStamp
} from '@/store/session-stamp'
import { $sessionTiles, closeAllOpenSessionTiles } from '@/store/session-states'
import { ackStoredSessionId } from '@/store/session-unread'
import { canOpenSessionInTerminal, canOpenSessionWindow, openSessionInTerminal } from '@/store/windows'

import type { SessionTitleResponse } from '../../types'

// Rename a session, preferring the gateway's session.title RPC over REST.
//
// A freshly *branched* session (and any brand-new chat) lives only in the
// gateway's in-memory _sessions map keyed by its RUNTIME id — no row is
// persisted to state.db until the first turn. REST PATCH /api/sessions/{id}
// resolves against the stored sessions table, so it 404s ("Session not found")
// on these runtime-only sessions. The session.title RPC resolves the live
// runtime session AND persists the row on demand, so it succeeds where REST
// cannot. This mirrors the /title slash command's fix (use-prompt-actions.ts).
//
// We only take the RPC path for the ACTIVE/selected session: its runtime id is
// known ($activeSessionId) and it lives on the active gateway, so there is no
// profile-routing ambiguity. Every other row (already persisted, possibly on a
// background profile) keeps the REST path, which handles profile scoping and a
// non-empty title is required by the RPC (it rejects clears), so clears stay on
// REST too.
export async function renameSessionPreferringRpc(
  storedSessionId: string,
  title: string,
  profile?: string
): Promise<{ title?: string }> {
  const isActiveRow = storedSessionId === $selectedStoredSessionId.get()
  const runtimeId = isActiveRow ? $activeSessionId.get() : null
  const gateway = activeGateway()

  if (title && runtimeId && gateway) {
    try {
      const result = await gateway.request<SessionTitleResponse>('session.title', {
        session_id: runtimeId,
        title
      })

      return { title: result?.title ?? title }
    } catch (err) {
      // Fall through to REST — e.g. the socket is mid-reconnect. REST still
      // works for any session that already has a persisted row. Log so a
      // genuine RPC-side failure (which then surfaces a REST 404 for the
      // runtime id) is at least diagnosable instead of silently swallowed.
      console.warn('session.title RPC rename failed; falling back to REST', err)
    }
  }

  return renameSession(storedSessionId, title, profile)
}

interface SessionActions {
  sessionId: string
  title: string
  pinned?: boolean
  /** Backend-derived read state — drives the Mark as unread/read label. */
  unread?: boolean
  profile?: string
  /** The row's current durable stamps (`sessions.stamps`), when the caller
   *  already has them — the sidebar row does, because it draws the chips. A
   *  caller that omits them still gets the right check marks and Clear row: the
   *  submenu falls back to reading the loaded rows. */
  stamps?: string[]
  onPin?: () => void
  /** Toggle the persisted read-state watermark for this row. */
  onToggleUnread?: () => void
  onBranch?: () => void
  onArchive?: () => void
  onDelete?: () => void
  /** Close this surface (a tile tab) — omitted where nothing closes (sidebar
   *  rows, the main tab). */
  onClose?: () => void
  /** TAB surfaces: the session is already a tab, so "Open in new tab" is
   *  nonsense there — sidebar rows/dropdowns keep it. */
  surface?: 'row' | 'tab'
  /** The tab's layout-tree pane id (`session-tile:<id>` or `workspace`) — enables
   *  the Close-others / to-the-right / all tab verbs. Tab surfaces only. */
  tabPaneId?: string
  /** The MAIN tab's escape hatch: hide the zone's tab bar (it sticky-shows
   *  once a tab is ever gained; this is the explicit off switch). */
  onHideTabBar?: () => void
}

// The color picker inside the session menu's Appearance submenu. Its own
// component so only an OPEN submenu subscribes to the stores (not every row's
// menu). Reads/writes the override keyed by the DURABLE id so a color survives
// compression; clearing falls back to the inherited project color.
function SessionColorSwatches({ sessionId }: { sessionId: string }) {
  const { t } = useI18n()
  const overrides = useStore($sessionColorOverrides)
  const session = useStore($sessions).find(s => sessionMatchesStoredId(s, sessionId))
  const durableId = session ? sessionPinId(session) : sessionId

  return (
    <ColorSwatches
      clearIcon="circle-slash"
      clearLabel={t.sidebar.projects.noColor}
      onChange={color => setSessionColorOverride(durableId, color)}
      swatches={PROFILE_SWATCHES}
      value={overrides[durableId] ?? null}
    />
  )
}

// The project list inside the session menu's "Move to project" submenu. Its own
// component so only an OPEN submenu subscribes to the stores (same reasoning as
// SessionColorSwatches). Re-homes the session's workspace at the target
// project's root — the fix for a chat created in the wrong folder. The current
// owner and folderless projects (the Home bucket) are excluded: there is
// nothing to move into.
function MoveToProjectItems({ kit, sessionId, profile }: { kit: MenuKit; sessionId: string; profile?: string }) {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const tree = useStore($projectTree)
  const session = useStore($sessions).find(s => sessionMatchesStoredId(s, sessionId))
  const cwd = session?.cwd?.trim() || ''
  const currentProjectId = cwd ? projectIdForCwd(cwd) : null
  const targets = tree.filter(node => node.id !== currentProjectId && !node.isNoProject && projectRootCwd(node))

  if (targets.length === 0) {
    return <kit.Item disabled>{p.moveNoProjects}</kit.Item>
  }

  return (
    <>
      {targets.map(node => (
        <kit.Item
          key={node.id}
          onSelect={() => {
            triggerHaptic('selection')
            moveSessionToProject(sessionId, node.id, profile)
              .then(() => notify({ durationMs: 2_000, kind: 'success', message: p.movedTo(node.label) }))
              .catch(err => notifyError(err, p.moveFailed))
          }}
        >
          {node.label}
        </kit.Item>
      ))}
    </>
  )
}

// The stamp picker inside the session menu's "Stamp" submenu: the titles the
// user keeps, the clear row (only while the session carries something) and the
// two text doors. Its own component so only an OPEN submenu subscribes to the
// stores (same reasoning as SessionColorSwatches). The row's own `stamps` prop
// wins where the caller has it — it is what the chips beside the title are drawn
// from, so a check mark can never disagree with what the list is showing; the
// lookup is the fallback for the surfaces that pass nothing.
//
// Every title row TOGGLES: clicking one the session does not carry adds it (at
// the end, so order = the order it was built), clicking one it does carry takes
// it off and leaves the rest alone. At SESSION_STAMP_LIMIT the session is full,
// so the rows that would add a label go disabled and a hint row says why —
// better than a row that silently does nothing.
//
// Everything that CHANGES something happens INSIDE this menu: a title's colour
// swatches and both text inputs expand in place instead of opening a dialog. A
// Radix menu closes on select and on focus leaving its content, so a dialog took
// the whole submenu down with it and made the user re-navigate after every
// change; the panels keep the menu exactly where it is.
//
// Each title row carries two affordances, revealed while the row is highlighted
// (Radix focuses the item under the pointer, so `focus-within` covers pointer
// AND keyboard movement): a dot that opens that title's colour panel, and a ✕
// that takes the title off the menu. Deleting a title is never a data change — a
// session already carrying "Hold" keeps reading "Hold" everywhere — and the
// restore row that appears while anything is off is the way back.
const STAMP_ROW_ACTION =
  'shrink-0 rounded p-0.5 text-(--ui-text-tertiary) opacity-0 transition-opacity group-focus-within/stamp:opacity-100 group-hover/stamp:opacity-100 hover:text-foreground focus-visible:opacity-100'

/** Which in-menu panel is open: a title's colours, a new title, an emoji picked
 *  off the bundled catalog, or a one-off custom label for this session. */
type StampPanel =
  | { kind: 'add' }
  | { kind: 'color'; label: string }
  | { kind: 'custom' }
  | { kind: 'emoji' }
  | null

/** A nested control inside a menu row: the ROW must not take the press. Radix
 *  resolves a click from the pointer sequence, not from `click` alone, so the
 *  pointer events are stopped too — without them the action ran AND the menu
 *  closed (or the row's own select fired). */
function stopRowSelect(event: React.SyntheticEvent): void {
  event.preventDefault()
  event.stopPropagation()
}

function SessionStampItems({
  kit,
  profile,
  sessionId,
  stamps
}: {
  kit: MenuKit
  profile?: string
  sessionId: string
  stamps?: string[]
}) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const session = useStore($sessions).find(s => sessionMatchesStoredId(s, sessionId))
  // The caller's own list first (it is the one the chips are drawn from), else
  // the loaded row's. `stampLabels` is the ONE reader, so a row that carries only
  // the older singular label still shows up here as a stamp.
  const current = stamps?.length ? normalizeSessionStamps(stamps) : stampLabels(session)
  const presets = useStore($stampPresets)
  const deleted = useStore($deletedStampPresets)
  const colors = useStore($stampColorOverrides)
  const [panel, setPanel] = useState<StampPanel>(null)
  const [typed, setTyped] = useState('')
  // The Emoji panel's query and, once a query exists, its hits over the bundled
  // catalog. `null` while the first answer is in flight — the panel says so
  // rather than flashing an empty grid.
  const [emojiQuery, setEmojiQuery] = useState('')
  const [emojiHits, setEmojiHits] = useState<null | string[]>(null)
  const full = current.length >= SESSION_STAMP_LIMIT

  useEffect(() => {
    const query = emojiQuery.trim()

    if (panel?.kind !== 'emoji' || !query) {
      setEmojiHits(null)

      return
    }

    // Bounded window per keystroke: a search that is no longer the latest must
    // not paint over the newer one (the index is in memory after the first
    // load, but the FIRST one reads a bundled JSON file off disk).
    let live = true

    const timer = window.setTimeout(() => {
      searchEmoji(query, SESSION_STAMP_EMOJI_SEARCH_LIMIT)
        .then(entries => {
          if (live) {
            setEmojiHits(entries.map(entry => entry.emoji))
          }
        })
        .catch(() => {
          // An unreadable catalog is an empty result, not a spinner that never
          // resolves: the curated grid is one Escape away.
          if (live) {
            setEmojiHits([])
          }
        })
    }, 120)

    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [emojiQuery, panel?.kind])

  const commitTyped = (kind: 'add' | 'custom') => {
    const next = normalizeSessionStamp(typed)

    // An empty submit writes NOTHING: with a list behind the menu, "no label"
    // is what the Clear row is for, and guessing that an empty input means
    // "take everything off" would be a destructive default.
    if (!next) {
      setPanel(null)

      return
    }

    // "Add stamp title…" names a MENU entry that stays for the next session too;
    // "Custom stamp…" is a one-off label for this session. Both then toggle it on
    // through the same write path as the rows above.
    if (kind === 'add') {
      addStampTitle(next)
    }

    void toggleStamp(sessionId, profile, next, current, r)
    setTyped('')
    setPanel(null)
  }

  const textPanel = (kind: 'add' | 'custom', placeholder: string) => (
    <DropdownMenuSearch
      key={kind}
      maxLength={SESSION_STAMP_MAX_LENGTH}
      onKeyDown={event => {
        // Enter belongs to this input, not to the highlighted menu row: a
        // stopped key never reaches Radix's own Enter handling.
        if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
          event.preventDefault()
          event.stopPropagation()
          commitTyped(kind)
        } else if (event.key === 'Escape') {
          event.preventDefault()
          event.stopPropagation()
          setPanel(null)
        }
      }}
      onValueChange={value => setTyped(value.slice(0, SESSION_STAMP_MAX_LENGTH))}
      placeholder={placeholder}
      value={typed}
    />
  )

  /**
   * Tapping an emoji goes on this session NOW and joins the menu's own titles,
   * so the emoji actually in use are one tap next time. That is "Add stamp
   * title…" with a chooser — the panel IS that door, drawn instead of typed —
   * and it shares the ONE write path, so the haptic, the notification and the
   * optimistic row patch are identical to a title row's.
   */
  const commitEmoji = (emoji: string) => {
    addStampTitle(emoji)
    void toggleStamp(sessionId, profile, emoji, current, r)
  }

  // The Emoji panel: a search over the bundled catalog (offline — lib/emoji-index
  // is also what the composer's `:shortcode:` completions read) and a grid of what
  // it found, with the curated set standing in before anything is typed.
  //
  // Buttons, not menu items: a row per emoji would put the whole catalog into the
  // menu's keyboard navigation, and Radix leaves a plain child of the menu content
  // alone — so a tap here stamps without closing the submenu (the colour swatches
  // above are the same shape). That is what makes picking two emoji in a row
  // painless, and it is why a picked emoji shows its own state in place instead of
  // the row's trailing check.
  const query = emojiQuery.trim()
  const emojiGrid = query ? emojiHits : [...SESSION_STAMP_EMOJI]

  const emojiPanel = (
    <>
      <DropdownMenuSearch
        aria-label={r.stampEmojiSearch}
        key="emoji"
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.preventDefault()
            event.stopPropagation()
            setPanel(null)
            setEmojiQuery('')
          } else if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
            // Enter belongs to the query, not to the menu row highlighted behind
            // it: a stopped key never reaches Radix's own Enter handling.
            event.preventDefault()
            event.stopPropagation()
          }
        }}
        onValueChange={setEmojiQuery}
        placeholder={r.stampEmojiSearch}
        value={emojiQuery}
      />
      {emojiGrid !== null && emojiGrid.length > 0 && (
        <div
          className="dt-portal-scrollbar grid max-h-40 grid-cols-8 gap-0.5 overflow-y-auto px-2 pt-1 pb-1.5"
          data-stamp-emoji-grid
        >
          {emojiGrid.map(emoji => {
            const isCurrent = hasStampLabel(current, emoji)

            return (
              <button
                // Names the emoji it stamps with, so the grid is answerable to a
                // screen reader and to a test by a stable label.
                aria-label={r.stampEmojiPick(emoji)}
                aria-pressed={isCurrent}
                className={cn(
                  'grid size-7 place-items-center rounded-md text-[1.0625rem] leading-none transition hover:bg-(--ui-control-hover-background)',
                  isCurrent && 'bg-[color-mix(in_srgb,var(--ui-accent)_18%,transparent)]'
                )}
                // A full session still shows what it carries; only ADDING is out
                // of reach (the hint row above says why).
                disabled={full && !isCurrent}
                key={emoji}
                onClick={() => commitEmoji(emoji)}
                title={emoji}
                type="button"
              >
                {emoji}
              </button>
            )
          })}
        </div>
      )}
      {query && !emojiGrid && (
        <div className="px-2 pt-0.5 pb-1.5 text-[0.6875rem] text-(--ui-text-tertiary)" data-stamp-emoji-loading>
          {t.common.loading}
        </div>
      )}
      {query && emojiGrid?.length === 0 && (
        <div className="px-2 pt-0.5 pb-1.5 text-[0.6875rem] text-(--ui-text-tertiary)" data-stamp-emoji-empty>
          {r.stampEmojiEmpty}
        </div>
      )}
    </>
  )

  return (
    <>
      {full && (
        // A plain div child of the menu content: no item, so it cannot be
        // selected — it is the explanation for the rows that just went inert.
        <div className="px-2 pt-1 pb-1.5 text-[0.6875rem] text-(--ui-text-tertiary)" data-stamp-limit>
          {r.stampLimit(SESSION_STAMP_LIMIT)}
        </div>
      )}
      {presets.map(preset => {
        const isCurrent = hasStampLabel(current, preset)
        const color = stampColorFor(preset, colors)
        const panelOpen = panel?.kind === 'color' && panel.label === preset
        // An emoji title gets no colour control: the glyph is drawn by the
        // platform's colour emoji font, so a picked colour would be a choice the
        // user cannot see. The ✕ to take it off the menu stays.
        const emojiTitle = isEmojiStamp(preset)

        return (
          <Fragment key={preset}>
            <kit.Item
              // Explicit name: the row's own text is the title, and the buttons
              // inside it must not become part of what the row is called.
              aria-label={preset}
              className="group/stamp"
              // A full session still offers the labels it carries (click to take
              // one off); only ADDING a fourth is impossible.
              disabled={full && !isCurrent}
              onSelect={event => {
                event.preventDefault()
                void toggleStamp(sessionId, profile, preset, current, r)
              }}
            >
              {preset}
              {/* Mirrors how the app marks a current choice in a menu row (a
                  trailing check on the selected row only — base-branch-picker,
                  kanban's board switcher). */}
              {isCurrent && (
                <Codicon className="ml-auto shrink-0 text-(--ui-accent)" name="check" size="0.8rem" />
              )}
              {!emojiTitle && (
                <button
                  aria-label={r.stampColor(preset)}
                  aria-pressed={panelOpen}
                  className={cn(STAMP_ROW_ACTION, !isCurrent && 'ml-auto', panelOpen && 'opacity-100')}
                  onClick={event => {
                    stopRowSelect(event)
                    triggerHaptic('selection')
                    setPanel(panelOpen ? null : { kind: 'color', label: preset })
                  }}
                  onPointerDown={stopRowSelect}
                  onPointerUp={stopRowSelect}
                  title={r.stampColor(preset)}
                  type="button"
                >
                  <span
                    className={cn('block size-2 rounded-full bg-current', !color && stampHueClass(preset))}
                    style={color ? { color } : undefined}
                  />
                </button>
              )}
              <button
                aria-label={r.stampRemove(preset)}
                // No colour button to carry the right-alignment on an emoji row.
                className={cn(STAMP_ROW_ACTION, !isCurrent && emojiTitle && 'ml-auto')}
                onClick={event => {
                  stopRowSelect(event)
                  triggerHaptic('selection')
                  deleteStampPreset(preset)
                }}
                onPointerDown={stopRowSelect}
                onPointerUp={stopRowSelect}
                title={r.stampRemove(preset)}
                type="button"
              >
                <Codicon name="close" size="0.75rem" />
              </button>
            </kit.Item>
            {panelOpen && (
              <div className="px-2 py-1.5" data-stamp-color-panel={preset}>
                <ColorSwatches
                  clearLabel={r.stampColorReset}
                  onChange={next => setStampColor(preset, next)}
                  swatches={STAMP_SWATCHES}
                  value={color}
                />
              </div>
            )}
          </Fragment>
        )
      })}
      {panel?.kind === 'add' ? (
        textPanel('add', r.stampAddPlaceholder)
      ) : (
        <kit.Item
          disabled={full}
          onSelect={event => {
            event.preventDefault()
            triggerHaptic('selection')
            setTyped('')
            setPanel({ kind: 'add' })
          }}
        >
          <Codicon name="add" size="0.875rem" />
          <span>{r.stampAdd}</span>
        </kit.Item>
      )}
      {panel?.kind === 'emoji' ? (
        emojiPanel
      ) : (
        <kit.Item
          disabled={full}
          onSelect={event => {
            event.preventDefault()
            triggerHaptic('selection')
            setEmojiQuery('')
            setPanel({ kind: 'emoji' })
          }}
        >
          <Codicon name="smiley" size="0.875rem" />
          <span>{r.stampEmoji}</span>
        </kit.Item>
      )}
      {deleted.length > 0 && (
        <kit.Item
          onSelect={event => {
            event.preventDefault()
            triggerHaptic('selection')
            restoreStampPresets()
          }}
        >
          <Codicon name="history" size="0.875rem" />
          <span>{r.stampRestore}</span>
        </kit.Item>
      )}
      {current.length > 0 && (
        <kit.Item
          onSelect={event => {
            event.preventDefault()
            void clearStamps(sessionId, profile, r)
          }}
        >
          <Codicon name="circle-slash" size="0.875rem" />
          <span>{r.stampClear}</span>
        </kit.Item>
      )}
      <kit.Separator />
      {panel?.kind === 'custom' ? (
        textPanel('custom', r.stampCustomPlaceholder)
      ) : (
        <kit.Item
          disabled={full}
          onSelect={event => {
            event.preventDefault()
            triggerHaptic('selection')
            // Empty: the panel ADDS a label now. Seeding it with a label the
            // session already carries would make a submit toggle that one off,
            // which is not what "Custom stamp…" says it does.
            setTyped('')
            setPanel({ kind: 'custom' })
          }}
        >
          <Codicon name="edit" size="0.875rem" />
          <span>{r.stampCustom}</span>
        </kit.Item>
      )}
    </>
  )
}

/** The copy the stamp writes report back with — one shape for both helpers. */
interface StampCopy {
  stampCleared: string
  stampRemoved: (label: string) => string
  stampSaved: (label: string) => string
}

/**
 * The ONE write path for the Stamp submenu: toggle *label* on the session, then
 * a brief beat naming what actually landed — added, taken off, or "no stamps
 * left" when that was the last one.
 *
 * A refusal (the backend rejected the write) is already reported by
 * `applySessionStamps`, which also puts the row back — reporting it here as well
 * would double-notify the user. An over-cap click never reaches here at all: the
 * row is disabled while the session is full.
 */
async function toggleStamp(
  sessionId: string,
  profile: string | undefined,
  label: string,
  current: string[],
  r: StampCopy
): Promise<boolean> {
  const removing = hasStampLabel(current, label)

  triggerHaptic('selection')

  const ok = await toggleSessionStamp(sessionId, profile, label)

  if (ok) {
    notify({
      durationMs: 2_000,
      kind: 'success',
      message: removing
        ? (current.length > 1 ? r.stampRemoved(label) : r.stampCleared)
        : r.stampSaved(label)
    })
  }

  return ok
}

/** The Clear row: every label off the session in one write. */
async function clearStamps(
  sessionId: string,
  profile: string | undefined,
  r: StampCopy
): Promise<boolean> {
  triggerHaptic('selection')

  const ok = await applySessionStamps(sessionId, profile, [])

  if (ok) {
    notify({ durationMs: 2_000, kind: 'success', message: r.stampCleared })
  }

  return ok
}

function useSessionActions({
  sessionId,
  title,
  pinned = false,
  unread = false,
  profile,
  stamps,
  onPin,
  onToggleUnread,
  onBranch,
  onArchive,
  onDelete,
  onClose,
  onHideTabBar,
  surface = 'row',
  tabPaneId
}: SessionActions) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [renameOpen, setRenameOpen] = useState(false)
  // The rename item opens a Dialog. When a menu closes, Radix restores focus to
  // its trigger — for a sidebar row that trigger is the row's own <button>, so
  // focus lands there instead of the dialog's input: Space then activates the
  // row (selecting the session) and the arrow keys move the list rather than
  // the caret. Suppress that one restore so the dialog keeps focus; every other
  // action leaves the restore alone (it's the correct behavior for them). Mirrors
  // the project menu's appearance-popover guard.
  const suppressCloseFocusRef = useRef(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const tiles = useStore($sessionTiles)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const isRemote = useStore($connection)?.mode === 'remote'
  // The row's finished-unread dot is cleared by opening the session (main or
  // tile) — this menu item is the explicit escape hatch for the rest.
  const isUnread = useStore($unreadFinishedSessionIds).includes(sessionId)

  // Already showing as a tab somewhere (a tile, or loaded in main — main IS
  // a tab): offering "Open in new tab" again is noise.
  const alreadyTabbed = sessionId === selectedStoredSessionId || tiles.some(tile => tile.storedSessionId === sessionId)

  const spec = (partial: Omit<ActionItemSpec, 'onSelect'> & { onSelect: () => void }): ActionItemSpec => partial

  // OPEN — where else this session can go. A tab surface IS a tab already,
  // so it only offers the window hop (and its own Close, below).
  const openItems: ActionItemSpec[] = [
    ...(surface === 'row' && !alreadyTabbed
      ? [
          spec({
            disabled: !sessionId,
            icon: 'browser',
            label: r.openInNewTab,
            onSelect: () => {
              triggerHaptic('selection')
              // Stack into the MAIN zone as a tab (center dock; the strip
              // sticky-shows on gain) — the door to the tab bar. Focuses first
              // if the session is already on screen.
              openSession(sessionId, () => undefined, 'tab')
            }
          })
        ]
      : []),
    ...(canOpenSessionWindow()
      ? [
          spec({
            disabled: !sessionId,
            icon: 'link-external',
            label: r.newWindow,
            onSelect: () => {
              triggerHaptic('selection')
              openSession(sessionId, () => undefined, 'window')
            }
          })
        ]
      : []),
    // The user's OWN terminal, not the in-app pane: resumes the session in the
    // TUI. Hidden on a remote connection — the emulator we'd open runs on this
    // machine while the session (and its runtime) lives on the remote host.
    ...(canOpenSessionInTerminal() && !isRemote
      ? [
          spec({
            disabled: !sessionId,
            icon: 'terminal',
            label: r.openInTerminal,
            onSelect: () => {
              triggerHaptic('selection')

              // Read the row lazily: subscribing every row's menu to $sessions
              // would re-render the whole sidebar on each session update.
              const cwd =
                $sessions
                  .get()
                  .find(s => sessionMatchesStoredId(s, sessionId))
                  ?.cwd?.trim() || undefined

              void openSessionInTerminal(sessionId, { cwd, profile })
            }
          })
        ]
      : [])
  ]

  // IDENTITY — name/mark/reference the session.
  const identityItems: ActionItemSpec[] = [
    spec({
      disabled: !sessionId,
      icon: 'edit',
      label: r.rename,
      onSelect: () => {
        triggerHaptic('selection')
        // Keep focus off the row trigger so it lands in the dialog input.
        suppressCloseFocusRef.current = true
        setRenameOpen(true)
      }
    }),
    spec({
      disabled: !onPin,
      icon: 'pin',
      label: pinned ? r.unpin : r.pin,
      onSelect: () => {
        triggerHaptic('selection')
        onPin?.()
      }
    }),
    // One read-state item, driven by BOTH unread sources: the transient
    // finished-unread dot (isUnread) and the backend watermark (unread).
    // "Mark as read" clears whichever is lit; "Mark as unread" arms the
    // persisted watermark so the dot survives restarts.
    spec({
      disabled: !sessionId || (!onToggleUnread && !isUnread),
      // Closed envelope = unread, open envelope = read (codicon has mail and
      // mail-read, but no mail-unread glyph — verified against the font css).
      icon: unread || isUnread ? 'mail-read' : 'mail',
      label: unread || isUnread ? r.markRead : r.markUnread,
      onSelect: () => {
        triggerHaptic('selection')

        if (unread || isUnread) {
          // Clear the transient family dot immediately (and ack the persisted
          // watermark/marker so a list refresh doesn't repaint it)…
          markSessionRead(sessionId)
          ackStoredSessionId(sessionId)

          // …and retire the persisted watermark when the row carries one.
          if (unread) {
            onToggleUnread?.()
          }
        } else {
          onToggleUnread?.()
        }
      }
    })
  ]

  // WORK — derive/extract from the session.
  const workItems: ActionItemSpec[] = [
    spec({
      disabled: !onBranch,
      // Fork glyph to match the inline message action's GitFork icon
      // (assistant-message.tsx). NB: this codicon font has no `git-fork`
      // glyph (only `git-fork-private`); `repo-forked` is the fork icon.
      icon: 'repo-forked',
      label: r.branchFrom,
      onSelect: () => {
        triggerHaptic('selection')
        onBranch?.()
      }
    }),
    spec({
      disabled: !sessionId,
      icon: 'cloud-download',
      label: r.export,
      onSelect: () => {
        triggerHaptic('selection')
        void exportSession(sessionId, { profile, title })
      }
    })
  ]

  // TAB — verbs that act on the strip (tabs only; a row isn't a tab).
  const closeTargets = surface === 'tab' && tabPaneId ? treeTabCloseTargets(tabPaneId) : null

  const tabItems: ActionItemSpec[] =
    surface === 'tab'
      ? [
          ...(tabPaneId
            ? [
                spec({
                  icon: 'refresh',
                  label: t.zones.reload,
                  onSelect: () => {
                    triggerHaptic('selection')
                    reloadTreePane(tabPaneId)
                  }
                })
              ]
            : []),
          ...(onClose
            ? [
                spec({
                  disabled: false,
                  icon: 'close',
                  label: t.common.close,
                  onSelect: () => {
                    triggerHaptic('selection')
                    onClose()
                  }
                })
              ]
            : []),
          ...(tabPaneId
            ? [
                spec({
                  disabled: !closeTargets?.others,
                  icon: 'close-all',
                  label: t.zones.closeOthers,
                  onSelect: () => {
                    triggerHaptic('selection')
                    closeOtherTreeTabs(tabPaneId)
                  }
                }),
                spec({
                  disabled: !closeTargets?.right,
                  icon: 'arrow-right',
                  label: t.zones.closeToRight,
                  onSelect: () => {
                    triggerHaptic('selection')
                    closeTreeTabsToRight(tabPaneId)
                  }
                }),
                spec({
                  disabled: !closeTargets?.all,
                  icon: 'clear-all',
                  label: t.zones.closeAll,
                  onSelect: () => {
                    triggerHaptic('selection')
                    // Persist-close session tiles before dismissing the
                    // remaining tree panes, or Bot Mode rehydrates them
                    // from the shared tile bucket (#94137).
                    closeAllOpenSessionTiles(tabPaneId)
                    closeAllTreeTabs(tabPaneId)
                  }
                })
              ]
            : [])
        ]
      : []

  // DANGER — put it away / destroy it (delete stays last, destructive-red).
  const dangerItems: ActionItemSpec[] = [
    spec({
      disabled: !onArchive,
      icon: 'archive',
      label: r.archive,
      onSelect: () => {
        triggerHaptic('selection')
        onArchive?.()
      }
    }),
    {
      className: 'text-destructive focus:text-destructive',
      disabled: !onDelete,
      icon: 'trash',
      label: t.common.delete,
      onSelect: () => {
        triggerHaptic('warning')

        // Deleting is irreversible (the CLI path asks y/N; the desktop used to
        // fire instantly on click). Gate it behind an explicit confirm — see
        // #61470. The dialog owns the delete call, so every surface that routes
        // through this menu (sidebar rows, tab menus, the chat header) gets the
        // guard for free.
        if (onDelete) {
          setDeleteOpen(true)
        }
      },
      variant: 'destructive'
    }
  ]

  const renderItems = (kit: MenuKit) => (
    <>
      {openItems.map(item => renderActionItem(kit, item))}
      {openItems.length > 0 && <kit.Separator />}
      {identityItems.map(item => renderActionItem(kit, item))}
      <kit.Sub>
        <kit.SubTrigger disabled={!sessionId}>
          <Codicon name="tag" size="0.875rem" />
          <span>{r.stamp}</span>
        </kit.SubTrigger>
        <kit.SubContent>
          <SessionStampItems kit={kit} profile={profile} sessionId={sessionId} stamps={stamps} />
        </kit.SubContent>
      </kit.Sub>
      <kit.Sub>
        <kit.SubTrigger disabled={!sessionId}>
          <Codicon name="symbol-color" size="0.875rem" />
          <span>{t.sidebar.projects.menuAppearance}</span>
        </kit.SubTrigger>
        <kit.SubContent className="p-2">
          <SessionColorSwatches sessionId={sessionId} />
        </kit.SubContent>
      </kit.Sub>
      <CopyButton
        appearance={kit.copyAppearance}
        disabled={!sessionId}
        errorMessage={r.copyIdFailed}
        iconClassName="size-3.5 text-current"
        key={r.copyId}
        label={r.copyId}
        onCopyError={err => notifyError(err, r.copyIdFailed)}
        text={sessionId}
      />
      <kit.Separator />
      {workItems.map(item => renderActionItem(kit, item))}
      <kit.Sub>
        <kit.SubTrigger disabled={!sessionId}>
          <Codicon name="folder" size="0.875rem" />
          <span>{t.sidebar.projects.moveToProject}</span>
        </kit.SubTrigger>
        <kit.SubContent>
          <MoveToProjectItems kit={kit} profile={profile} sessionId={sessionId} />
        </kit.SubContent>
      </kit.Sub>
      {tabItems.length > 0 && (
        <>
          <kit.Separator />
          {tabItems.map(item => renderActionItem(kit, item))}
        </>
      )}
      <kit.Separator />
      {dangerItems.map(item => renderActionItem(kit, item))}
      {onHideTabBar && (
        <>
          <kit.Separator />
          {renderActionItem(kit, {
            disabled: false,
            icon: 'eye-closed',
            label: r.hideTabBar,
            onSelect: () => {
              triggerHaptic('selection')
              onHideTabBar()
            }
          })}
        </>
      )}
    </>
  )

  const renameDialog = (
    <RenameSessionDialog
      currentTitle={title}
      onOpenChange={setRenameOpen}
      open={renameOpen}
      profile={profile}
      sessionId={sessionId}
    />
  )

  // Consumed once per close: when rename was the action that closed the menu,
  // block Radix's focus-restore to the trigger so the dialog input keeps focus.
  const onCloseAutoFocus = (event: Event) => {
    if (suppressCloseFocusRef.current) {
      suppressCloseFocusRef.current = false
      event.preventDefault()
    }
  }

  const deleteDialog = (
    <DeleteSessionDialog
      onConfirm={() => {
        onDelete?.()
      }}
      onOpenChange={setDeleteOpen}
      open={deleteOpen}
      sessionTitle={title}
    />
  )

  return { deleteDialog, onCloseAutoFocus, renameDialog, renderItems }
}

interface DeleteSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  sessionTitle: string
}

// Thin wrapper over ConfirmDialog — the single choke point for every session
// delete entry point (sidebar rows, tab menus, the chat header). Deleting a
// session is irreversible and the desktop used to fire it instantly on click
// (#61470); this mirrors the CLI's y/N guard. onConfirm is the fire-and-forget
// delete call; ConfirmDialog owns the busy/done beat and Enter-to-confirm.
function DeleteSessionDialog({ open, onOpenChange, onConfirm, sessionTitle }: DeleteSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row

  return (
    <ConfirmDialog
      busyLabel={r.deleting}
      confirmLabel={t.common.delete}
      description={r.deleteDesc(sessionTitle)}
      destructive
      doneLabel={r.deleted}
      onClose={() => onOpenChange(false)}
      onConfirm={onConfirm}
      open={open}
      title={r.deleteTitle}
    />
  )
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof ActionsMenu>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { t } = useI18n()

  const { deleteDialog, onCloseAutoFocus, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ActionsMenu
        align={align}
        ariaLabel={t.sidebar.row.sessionActions}
        contentClassName="w-40"
        items={renderItems}
        onCloseAutoFocus={onCloseAutoFocus}
        sideOffset={sideOffset}
      >
        {children}
      </ActionsMenu>
      {renameDialog}
      {deleteDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  children: React.ReactNode
}

export function SessionContextMenu({ children, ...actions }: SessionContextMenuProps) {
  const { t } = useI18n()

  const { deleteDialog, onCloseAutoFocus, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ActionsContextMenu
        ariaLabel={t.sidebar.row.sessionActions}
        contentClassName="w-40"
        items={renderItems}
        onCloseAutoFocus={onCloseAutoFocus}
      >
        {children}
      </ActionsContextMenu>
      {renameDialog}
      {deleteDialog}
    </>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
  profile?: string
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle, profile }: RenameSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [value, setValue] = useState(currentTitle)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(currentTitle)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [currentTitle, open])

  const submit = async () => {
    const next = value.trim()

    if (!sessionId || submitting) {
      return
    }

    if (next === currentTitle.trim()) {
      onOpenChange(false)

      return
    }

    setSubmitting(true)

    try {
      const result = await renameSessionPreferringRpc(sessionId, next, profile)
      const finalTitle = result.title || next || ''
      setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
      notify({ durationMs: 2_000, kind: 'success', message: r.renamed })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, r.renameFailed)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.renameTitle}</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event)) {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={r.untitledPlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
