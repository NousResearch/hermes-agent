/**
 * Background MCP health checker. On gateway connect — and every 30 minutes
 * after — read the ACTIVE profile's MCP runtime state and nudge the user when
 * a server needs re-authentication (expired OAuth token) or a REMOTE server is
 * failing: on the transition, then at most once a day while it stays broken,
 * with a one-click path to the MCP page and a Disable button for servers the
 * user no longer wants.
 *
 * The sweep never connects to anything. It reads `mcp.servers.status`, the
 * backend's cached view of the connections its agents actually use. A probe of
 * its own (a fresh connect + list_tools) raced the backend's startup connects
 * into false failures (#133155), evicted single-session hosts' live session
 * (#89576) and could open an OAuth consent tab unattended (#142406).
 *
 * Local servers (stdio, or a loopback URL) are never nagged about. They are
 * usually hosted by an app that comes and goes (an editor's built-in MCP
 * server only answers while the editor runs), and the backend re-probes a
 * parked server on its own, so tools come back once the app is open again.
 * Their state stays visible on the MCP page.
 */

import type { McpServersStatusResult } from '@hermes/shared'

import { getHermesConfigRecord, setMcpServerEnabled } from '@/hermes'
import { translateNow } from '@/i18n'
import { isLoopbackPreviewUrl } from '@/lib/local-preview'
import { getServers, type McpServerEntry, serverEnabled } from '@/lib/mcp-servers'
import { persistString, storedString } from '@/lib/storage'
import { requestGatewayForProfile } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $gatewayState } from '@/store/session'

// A constant, not a config knob: the sweep is one cached-state read, and the
// notification is transition-gated below, so there is nothing to tune.
const CHECK_INTERVAL_MS = 30 * 60_000
const STATUS_TIMEOUT_MS = 15_000

/** `offline` = a local server that is not answering; it never notifies. */
export type McpHealthStatus = 'error' | 'needs-auth' | 'offline' | 'ok'

/**
 * The notify decision, as a pure state machine: nudge on a TRANSITION into a
 * bad state — never for ok. An unknown previous state (first sweep of the
 * session) notifies only when its persisted cooldown is absent or elapsed,
 * so an expired token discovered at launch is still surfaced without
 * re-notifying after a renderer restart.
 *
 * A server that STAYS broken is nudged again once the daily snooze lapses:
 * a dead OAuth token is a standing problem the user has to act on (sign in
 * again, or disable the server), and a one-shot toast they closed on Monday
 * is forgotten by Wednesday. `snoozedUntil` is the persisted per-server
 * cooldown (set when a toast is shown); the recheck path re-nudges only past
 * it, so an unchanged bad state costs at most one toast per day.
 */
export function shouldNotify(
  previous: McpHealthStatus | null,
  next: McpHealthStatus,
  snoozedUntil: number,
  now: number
): boolean {
  if (next !== 'error' && next !== 'needs-auth') {
    return false
  }

  // A renderer restart clears transition memory but not the per-server
  // cooldown. Treat that first sweep as a recheck: an active persisted snooze
  // must still suppress the toast, while an unset or elapsed snooze may nudge.
  if (previous === null) {
    return now >= snoozedUntil
  }

  return previous !== next || now >= snoozedUntil
}

/** Local = runs next to the backend: a spawned command or a loopback URL. */
export const isLocalMcpServer = (server: McpServerEntry): boolean =>
  typeof server.url !== 'string' || isLoopbackPreviewUrl(server.url)

type RuntimeRow = McpServersStatusResult['servers'][number]

/** One status row → health. Only a FAILED connect is unhealthy; connecting,
 *  lazy and never-tried servers have nothing to report yet. */
export function healthOf(row: RuntimeRow, server: McpServerEntry): McpHealthStatus {
  if (row.needs_auth) {
    return 'needs-auth'
  }

  if (row.status !== 'failed') {
    return 'ok'
  }

  return isLocalMcpServer(server) ? 'offline' : 'error'
}

// Same time-based snooze the update/skew toasts use (store/updates.ts): a
// shown toast arms a 24h cooldown for that (profile, server), persisted so an
// app restart does not re-nudge before the day is up.
const SNOOZE_KEY_PREFIX = 'hermes:mcp-health-snooze-until:'
const SNOOZE_MS = 24 * 60 * 60 * 1000

function snoozedUntil(key: string): number {
  const until = Number(storedString(SNOOZE_KEY_PREFIX + key) || 0)

  return Number.isFinite(until) ? until : 0
}

function snooze(key: string): void {
  persistString(SNOOZE_KEY_PREFIX + key, String(Date.now() + SNOOZE_MS))
}

// Last-known status per (profile, server) — the transition memory. Keyed by
// profile so one profile's broken server can't mute or trigger another's
// (AGENTS.md scope-in-key).
const lastStatus = new Map<string, McpHealthStatus>()

let started = false
let timer: ReturnType<typeof setInterval> | null = null
// Bumped on profile switch; in-flight sweeps compare and bail so a slow
// profile-A read can't record (or notify) into profile B's state.
let sweepEpoch = 0
// At most one sweep runs and one follow-up is remembered. Reconnect storms
// still request a fresh pass, but cannot append an unbounded backlog.
let sweepInFlight: Promise<void> | null = null
let sweepQueued = false
let offGatewayState: (() => void) | null = null
let offProfile: (() => void) | null = null

function openMcpServerPage(name: string): void {
  window.location.hash = `#/capabilities?tab=connectors&server=${encodeURIComponent(name)}`
}

// "Disable" from the toast: `enabled: false` in config.yaml (the server stays
// listed on the MCP page for a later re-enable). The backend follows the edit
// on its own — the gateway's config reconcile and the serve backend's next
// reload both drop a disabled server — so no reload RPC is issued here.
async function disableServer(profileKey: string, name: string): Promise<void> {
  try {
    await setMcpServerEnabled(name, false)
    lastStatus.delete(`${profileKey}::${name}`)
    notify({
      kind: 'success',
      message: translateNow('notifications.mcp.disabledMessage', name)
    })
  } catch (err) {
    notifyError(err, translateNow('notifications.mcp.disableFailed', name))
  }
}

function recordResult(profileKey: string, name: string, status: McpHealthStatus): void {
  const key = `${profileKey}::${name}`
  const previous = lastStatus.get(key) ?? null
  lastStatus.set(key, status)

  if (!shouldNotify(previous, status, snoozedUntil(key), Date.now())) {
    return
  }

  snooze(key)

  const needsAuth = status === 'needs-auth'

  notify({
    action: {
      label: translateNow(needsAuth ? 'notifications.mcp.signIn' : 'notifications.mcp.view'),
      onClick: () => openMcpServerPage(name)
    },
    id: `mcp-health-${key}`,
    kind: 'warning',
    message: translateNow(needsAuth ? 'notifications.mcp.needsAuthMessage' : 'notifications.mcp.errorMessage', name),
    secondaryAction: {
      label: translateNow('notifications.mcp.disable'),
      onClick: () => void disableServer(profileKey, name)
    },
    title: translateNow(needsAuth ? 'notifications.mcp.needsAuthTitle' : 'notifications.mcp.errorTitle')
  })
}

async function sweep(): Promise<void> {
  const epoch = sweepEpoch
  const profileKey = normalizeProfileKey($activeGatewayProfile.get())

  let config: Record<string, unknown>
  let status: McpServersStatusResult

  try {
    ;[config, status] = await Promise.all([
      getHermesConfigRecord(),
      requestGatewayForProfile<McpServersStatusResult>(profileKey, 'mcp.servers.status', {}, STATUS_TIMEOUT_MS)
    ])
  } catch {
    // Backend unreachable / mid-restart, or one too old to report MCP runtime
    // state (a missing capability, not a failure) — the next tick retries.
    return
  }

  if (epoch !== sweepEpoch || $gatewayState.get() !== 'open') {
    return
  }

  // getServers drops non-object entries (a bare `name:` in config.yaml parses
  // as `null`), so the lookups below never read off a null entry.
  const servers = getServers(config)

  for (const row of status.servers) {
    const server = servers[row.name]

    if (server && serverEnabled(server)) {
      recordResult(profileKey, row.name, healthOf(row, server))
    }
  }
}

function queueSweep(): void {
  if (sweepInFlight) {
    sweepQueued = true

    return
  }

  sweepInFlight = sweep()

  const settled = () => {
    sweepInFlight = null

    if (sweepQueued) {
      sweepQueued = false
      queueSweep()
    }
  }

  sweepInFlight.then(settled, settled)
}

function arm(): void {
  if (timer !== null) {
    return
  }

  queueSweep()
  timer = setInterval(queueSweep, CHECK_INTERVAL_MS)
}

function disarm(): void {
  sweepQueued = false

  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}

/** Wire the background checker to gateway/profile state. Idempotent. */
export function startMcpHealthChecker(): void {
  if (started || typeof window === 'undefined') {
    return
  }

  started = true

  // Never check while the gateway is disconnected: the timer only exists
  // while the active socket is open, and rearms on reconnect.
  offGatewayState = $gatewayState.subscribe((state: string) => {
    if (state === 'open') {
      arm()
    } else {
      disarm()
    }
  })

  // Profile switch: invalidate in-flight sweeps, drop the timer, and re-arm
  // for the new profile (fresh immediate sweep + fresh interval) if its
  // gateway is connected. lastStatus and the snooze keys are profile-keyed,
  // so no wipe is needed — profile A's transition memory stays intact for
  // when the user switches back.
  offProfile = $activeGatewayProfile.listen(() => {
    sweepEpoch += 1
    disarm()

    if ($gatewayState.get() === 'open') {
      arm()
    }
  })
}

export function stopMcpHealthChecker(): void {
  disarm()
  sweepEpoch += 1
  offGatewayState?.()
  offGatewayState = null
  offProfile?.()
  offProfile = null
  started = false
}
