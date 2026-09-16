/**
 * Background MCP health checker. On gateway connect — and every 30 minutes
 * after — probe the ACTIVE profile's enabled HTTP/SSE MCP servers and nudge
 * the user when one is in needs-auth (expired OAuth token) or error: on the
 * transition, then at most once a day while it stays broken, with a
 * one-click path to the MCP page's Authenticate button and a Disable button
 * for servers the user no longer wants.
 *
 * Scope is deliberate: stdio servers are NEVER probed here. Probing a stdio
 * server SPAWNS a local process, so a background timer would silently launch
 * user-configured commands every half hour. Only url-shaped servers (HTTP/SSE
 * — where OAuth expiry actually lives) are swept, sequentially, and through
 * the same probe cache the MCP page uses so neither surface re-probes what
 * the other just learned.
 */

import { getHermesConfigRecord, type McpTestResult, setMcpServerEnabled, testMcpServer } from '@/hermes'
import { translateNow } from '@/i18n'
import { classifyProbe, freshProbe, probeCache, probeKey } from '@/lib/mcp-probe-cache'
import { getServers } from '@/lib/mcp-servers'
import { persistString, storedString } from '@/lib/storage'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $gatewayState } from '@/store/session'

// A constant, not a config knob: the sweep is cheap (a handful of sequential
// HTTP probes at most) and the notification is transition-gated below, so
// there is nothing for a user to meaningfully tune.
const CHECK_INTERVAL_MS = 30 * 60_000

export type McpHealthStatus = 'error' | 'needs-auth' | 'ok'

/**
 * The notify decision, as a pure state machine: nudge on a TRANSITION into a
 * bad state — never for ok. An unknown previous state (first sweep of the
 * session) counts as a transition: an expired token discovered at launch is
 * exactly the case this exists for.
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

  return previous !== next || now >= snoozedUntil
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
// profile-A probe can't record (or notify) into profile B's state.
let sweepEpoch = 0
// At most one sweep runs and one follow-up is remembered. Reconnect storms
// still request a fresh pass, but cannot append an unbounded backlog.
let sweepInFlight: Promise<void> | null = null
let sweepQueued = false
let offGatewayState: (() => void) | null = null
let offProfile: (() => void) | null = null

// Navigation only — never auto-launch an OAuth flow from the background. The
// server query param routes through useDeepLinkHighlight on the MCP tab, which
// scrolls to and focuses the server so its ServerConfig pane (with the
// Authenticate button) is one click away.
function openMcpServerPage(name: string): void {
  window.location.hash = `#/skills?tab=mcp&server=${encodeURIComponent(name)}`
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

const isUrlServer = (server: Record<string, unknown>): boolean =>
  typeof server.url === 'string' && server.enabled !== false

async function sweep(): Promise<void> {
  const epoch = sweepEpoch
  const profileKey = normalizeProfileKey($activeGatewayProfile.get())

  let config: Record<string, unknown>

  try {
    config = await getHermesConfigRecord()
  } catch {
    // Backend unreachable / mid-restart — the next interval tick retries.
    return
  }

  if (epoch !== sweepEpoch) {
    return
  }

  // getServers drops non-object entries (a bare `name:` in config.yaml parses
  // as `null`), so isUrlServer below never reads `.url` off a null entry.
  const servers = getServers(config)

  for (const [name, server] of Object.entries(servers)) {
    if (!isUrlServer(server)) {
      continue
    }

    // A profile switch or gateway drop mid-sweep stops the remaining probes.
    if (epoch !== sweepEpoch || $gatewayState.get() !== 'open') {
      return
    }

    const key = probeKey(name, server, profileKey)
    let result = freshProbe(key)

    if (!result) {
      try {
        result = await testMcpServer(name)
      } catch (err) {
        result = { ok: false, error: err instanceof Error ? err.message : String(err), tools: [] } as McpTestResult
      }

      if (epoch !== sweepEpoch) {
        return
      }

      probeCache.set(key, { at: Date.now(), result })
    }

    recordResult(profileKey, name, classifyProbe(result))
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
