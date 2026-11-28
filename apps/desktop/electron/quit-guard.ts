// Quitting with a turn in flight kills the backend mid-tool-call: the work is
// lost, and anything the agent had half-written to disk stays half-written.
// Renderers publish what they're running; the main process asks before it lets
// that go. The decision + copy live here (pure, testable) so main.ts only owns
// the IPC and the dialog call.
//
// That's only true for a backend the app owns: an SSH-managed server the app
// starts and stops. The local profile gateway is attached, not owned (`hermes
// gateway ensure`): it keeps cron, messaging adapters and the in-flight turn
// running after the app quits, exactly like a remote URL or Hermes Cloud
// backend, so those prompts say so instead of warning about lost work (#79579).

const MAX_LISTED = 4

export interface ActiveWork {
  /** Titles of sessions running a turn. Untitled sessions contribute a count only. */
  titles: string[]
  /** Running turns, including untitled ones — always >= titles.length. */
  count: number
  /** Names of files with unsaved Files-pane edits. Absent when there are none. */
  unsavedFiles?: string[]
}

const MAX_UNSAVED_FILES = 50

function cleanStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .filter((item): item is string => typeof item === 'string')
        .map(item => item.trim())
        .filter(Boolean)
    : []
}

function withUnsaved(work: ActiveWork, files: Iterable<string>): ActiveWork {
  const unique = [...new Set(files)].slice(0, MAX_UNSAVED_FILES)

  return unique.length > 0 ? { ...work, unsavedFiles: unique } : work
}

export const NO_ACTIVE_WORK: ActiveWork = { count: 0, titles: [] }

/** Coerce an IPC payload from an untrusted renderer into an ActiveWork. */
export function normalizeActiveWork(payload: unknown): ActiveWork {
  if (!payload || typeof payload !== 'object') {
    return NO_ACTIVE_WORK
  }

  const raw = payload as { count?: unknown; titles?: unknown; unsavedFiles?: unknown }
  const titles = cleanStrings(raw.titles)
  const count = typeof raw.count === 'number' && Number.isFinite(raw.count) ? Math.max(0, Math.floor(raw.count)) : 0

  return withUnsaved({ count: Math.max(count, titles.length), titles }, cleanStrings(raw.unsavedFiles))
}

/**
 * The part of a report main's last-seen cache may keep. Turns fall back to it
 * (a webContents can reload mid-turn), but a closed window's unsaved files are
 * gone with the window and must not make a later quit ask about them.
 */
export function turnsOnly(work: ActiveWork): ActiveWork {
  return { count: work.count, titles: work.titles }
}

/** Merge every window's report into one. Windows can show the same session. */
export function mergeActiveWork(reports: Iterable<ActiveWork>): ActiveWork {
  const titles: string[] = []
  const unsaved: string[] = []
  let count = 0

  for (const report of reports) {
    count = Math.max(count, report.count)

    for (const title of report.titles) {
      if (!titles.includes(title)) {
        titles.push(title)
      }
    }

    unsaved.push(...(report.unsavedFiles ?? []))
  }

  return withUnsaved({ count: Math.max(count, titles.length), titles }, unsaved)
}

function bulletList(items: readonly string[], total: number = items.length): string[] {
  const listed = items.slice(0, MAX_LISTED)
  const remaining = total - listed.length
  const lines = listed.map(item => `• ${item}`)

  if (remaining > 0) {
    lines.push(remaining === 1 ? '• 1 more' : `• ${remaining} more`)
  }

  return lines
}

function unsavedMessage(files: readonly string[]): string {
  return files.length === 1
    ? 'You have unsaved changes in 1 file.'
    : `You have unsaved changes in ${files.length || 'some'} files.`
}

export type UnsavedEditsAction = 'leave' | 'update'

// Inspired by Claude Cowork (desktop changelog v7.37095.0, 2026-11-24): an
// edit in the Files pane must not vanish without a warning when the window
// closes or reloads, or the app relaunches to update. Leaving a window is
// answered from its own will-prevent-unload; quitting rides quitPromptFor.
export function unsavedEditsPrompt(files: readonly string[], action: UnsavedEditsAction): QuitPrompt {
  const consequence =
    action === 'update'
      ? 'Restarting to update discards these edits. Save them first if you want to keep them.'
      : 'Leaving discards these edits. Save them first if you want to keep them.'

  return {
    buttons: action === 'update' ? ['Cancel', 'Discard and Update'] : ['Keep Editing', 'Discard Changes'],
    detail: [...bulletList(files), files.length > 0 ? '' : null, consequence]
      .filter(line => line !== null)
      .join('\n')
      .trim(),
    message: unsavedMessage(files)
  }
}

function unsavedQuitPrompt(work: ActiveWork, files: readonly string[], backendOwned: boolean): QuitPrompt {
  const running =
    work.count < 1
      ? null
      : `Hermes is also still working on ${work.count === 1 ? '1 chat' : `${work.count} chats`}. ` +
        (backendOwned ? 'Quitting stops the agent mid-turn.' : 'The agent keeps running on the gateway.')

  return {
    buttons: ['Cancel', 'Quit Without Saving'],
    detail: [...bulletList(files), '', 'Quitting discards these edits.', running]
      .filter(line => line !== null)
      .join('\n'),
    message: unsavedMessage(files)
  }
}

export interface QuitPrompt {
  /** [cancel, confirm]: index 0 keeps the app open, index 1 quits. */
  buttons: readonly [string, string]
  detail: string
  message: string
}

export interface BackendOwnershipInput {
  /**
   * Backends this app stops when it quits: spawned local children plus
   * SSH-managed servers, across the primary and every pooled profile.
   */
  ownedBackendCount: number
  /** What the primary profile resolves to; null means the local profile gateway (attached, not owned). */
  primaryRouteKind: 'cloud' | 'remote' | 'ssh' | null
}

/**
 * Whether quitting takes the agent down with the app. Only SSH backends are
 * started and stopped by the app. The local gateway, a remote URL and a cloud
 * backend all outlive it; but any server the app itself manages (another
 * window's SSH connection) might be where the turn is running, so it counts.
 */
export function backendOwnedByApp({ ownedBackendCount, primaryRouteKind }: BackendOwnershipInput): boolean {
  return primaryRouteKind === 'ssh' || ownedBackendCount > 0
}

/**
 * The confirmation to show, or null when quitting should just proceed.
 *
 * `quittingForHandoff` covers the update / swap / uninstall relaunches: those
 * are the app replacing itself, not the user walking away, and a modal there
 * would strand the detached script waiting on a PID that never exits.
 *
 * `backendOwned` (see backendOwnedByApp) picks the copy: an owned backend dies
 * with the app, a remote/cloud one keeps working after it closes.
 */
export function quitPromptFor(
  work: ActiveWork,
  quittingForHandoff: boolean,
  backendOwned: boolean = true
): null | QuitPrompt {
  const unsaved = work.unsavedFiles ?? []

  if (quittingForHandoff || (work.count < 1 && unsaved.length < 1)) {
    return null
  }

  if (unsaved.length > 0) {
    return unsavedQuitPrompt(work, unsaved, backendOwned)
  }

  const lines = bulletList(work.titles, work.count)

  return {
    buttons: backendOwned ? ['Keep Running', 'Quit Anyway'] : ['Cancel', 'Quit'],
    detail: [
      lines.join('\n'),
      lines.length > 0 ? '' : null,
      backendOwned
        ? 'Quitting stops the agent mid-turn. Any work it has not finished writing is lost.'
        : 'The agent keeps running on the gateway. Quitting only closes Hermes on this computer; reconnect later to see the results.'
    ]
      .filter(line => line !== null)
      .join('\n')
      .trim(),
    message: work.count === 1 ? 'Hermes is still working on 1 chat.' : `Hermes is still working on ${work.count} chats.`
  }
}

/**
 * Whether a window's 'close' event should be intercepted to show the
 * active-work confirmation.
 *
 * On Windows and Linux, closing the last window triggers app.quit() →
 * before-quit, but by then the renderer's webContents is already destroyed
 * and its entry has been removed from the active-work map — so the existing
 * before-quit guard finds nothing and the app exits silently. Intercepting
 * 'close' instead catches the guard while the work data is still live.
 *
 * macOS is excluded: closing the primary window there is the standard
 * "window stays in Dock" gesture, not a quit, and prompting on it would
 * block a normal workflow. Chat windows other than the final one are also
 * excluded because closing them does not stop the application or its work.
 * Unsaved Files-pane edits count as work here: closing the last window quits.
 * Every other window close asks from its own will-prevent-unload instead.
 */
export function shouldGuardWindowClose(
  work: ActiveWork,
  quittingForHandoff: boolean,
  isMac: boolean,
  hasOtherChatWindows: boolean
): boolean {
  if (isMac || quittingForHandoff || hasOtherChatWindows || (work.count < 1 && !work.unsavedFiles?.length)) {
    return false
  }

  return true
}
