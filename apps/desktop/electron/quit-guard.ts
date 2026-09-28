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
}

export const NO_ACTIVE_WORK: ActiveWork = { count: 0, titles: [] }

/** Coerce an IPC payload from an untrusted renderer into an ActiveWork. */
export function normalizeActiveWork(payload: unknown): ActiveWork {
  if (!payload || typeof payload !== 'object') {
    return NO_ACTIVE_WORK
  }

  const raw = payload as { count?: unknown; titles?: unknown }

  const titles = Array.isArray(raw.titles)
    ? raw.titles
        .filter((title): title is string => typeof title === 'string')
        .map(title => title.trim())
        .filter(Boolean)
    : []

  const count = typeof raw.count === 'number' && Number.isFinite(raw.count) ? Math.max(0, Math.floor(raw.count)) : 0

  return { count: Math.max(count, titles.length), titles }
}

/** Merge every window's report into one. Windows can show the same session. */
export function mergeActiveWork(reports: Iterable<ActiveWork>): ActiveWork {
  const titles: string[] = []
  let count = 0

  for (const report of reports) {
    count = Math.max(count, report.count)

    for (const title of report.titles) {
      if (!titles.includes(title)) {
        titles.push(title)
      }
    }
  }

  return { count: Math.max(count, titles.length), titles }
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
  if (quittingForHandoff || work.count < 1) {
    return null
  }

  const listed = work.titles.slice(0, MAX_LISTED)
  const remaining = work.count - listed.length
  const lines = listed.map(title => `• ${title}`)

  if (remaining > 0) {
    lines.push(remaining === 1 ? '• 1 more' : `• ${remaining} more`)
  }

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
