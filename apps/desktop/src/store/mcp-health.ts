/**
 * Background MCP health checker. On gateway connect — and every 30 minutes
 * after — probe the ACTIVE profile's enabled HTTP/SSE MCP servers and nudge
 * the user when one transitions into needs-auth (expired OAuth token) or
 * error, with a one-click path to the MCP page's Authenticate button.
 *
 * Scope is deliberate: stdio servers are NEVER probed here. Probing a stdio
 * server SPAWNS a local process, so a background timer would silently launch
 * user-configured commands every half hour. Only url-shaped servers (HTTP/SSE
 * — where OAuth expiry actually lives) are swept, sequentially, and through
 * the same probe cache the MCP page uses so neither surface re-probes what
 * the other just learned.
 */

import { getHermesConfigRecord, type McpTestResult, testMcpServer } from '@/hermes'
import { translateNow } from '@/i18n'
import { classifyProbe, freshProbe, probeCache, probeKey } from '@/lib/mcp-probe-cache'
import { notify } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $gatewayState } from '@/store/session'

// A constant, not a config knob: the sweep is cheap (a handful of sequential
// HTTP probes at most) and the notification is transition-gated below, so
// there is nothing for a user to meaningfully tune.
const CHECK_INTERVAL_MS = 30 * 60_000

export type McpHealthStatus = 'error' | 'needs-auth' | 'ok'

/**
 * The notify decision, as a pure state machine: nudge only on a TRANSITION
 * into a bad state — never on every recheck of an already-known-bad server,
 * and never for ok. An unknown previous state (first sweep of the session)
 * counts as a transition: an expired token discovered at launch is exactly
 * the case this exists for.
 */
export function shouldNotifyOnTransition(previous: McpHealthStatus | null, next: McpHealthStatus): boolean {
  return (next === 'error' || next === 'needs-auth') && previous !== next
}

// Last-known status per (profile, server) — the transition memory — and the
// per-app-session notification cap. Both keyed by profile so one profile's
// broken server can't mute or trigger another's (AGENTS.md scope-in-key).
const lastStatus = new Map<string, McpHealthStatus>()
const notifiedThisSession = new Set<string>()

let started = false
let timer: ReturnType<typeof setInterval> | null = null
// Bumped on profile switch; in-flight sweeps compare and bail so a slow
// profile-A probe can't record (or notify) into profile B's state.
let sweepEpoch = 0
// Sweeps are chained, never concurrent — sequential probes, no parallel bursts.
let sweepChain: Promise<void> = Promise.resolve()
let offGatewayState: (() => void) | null = null
let offProfile: (() => void) | null = null

// Navigation only — never auto-launch an OAuth flow from the background. The
// server query param routes through useDeepLinkHighlight on the MCP tab, which
// scrolls to and focuses the server so its ServerConfig pane (with the
// Authenticate button) is one click away.
function openMcpServerPage(name: string): void {
  window.location.hash = `#/skills?tab=mcp&server=${encodeURIComponent(name)}`
}

function recordResult(profileKey: string, name: string, status: McpHealthStatus): void {
  const key = `${profileKey}::${name}`
  const previous = lastStatus.get(key) ?? null
  lastStatus.set(key, status)

  if (!shouldNotifyOnTransition(previous, status) || notifiedThisSession.has(key)) {
    return
  }

  notifiedThisSession.add(key)

  const needsAuth = status === 'needs-auth'

  notify({
    action: {
      label: translateNow(needsAuth ? 'notifications.mcp.signIn' : 'notifications.mcp.view'),
      onClick: () => openMcpServerPage(name)
    },
    id: `mcp-health-${key}`,
    kind: 'warning',
    message: translateNow(needsAuth ? 'notifications.mcp.needsAuthMessage' : 'notifications.mcp.errorMessage', name),
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

  const raw = config.mcp_servers
  const servers =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, Record<string, unknown>>) : {}

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
  sweepChain = sweepChain.then(sweep, sweep)
}

function arm(): void {
  if (timer !== null) {
    return
  }

  queueSweep()
  timer = setInterval(queueSweep, CHECK_INTERVAL_MS)
}

function disarm(): void {
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
  // gateway is connected. lastStatus/notifiedThisSession are profile-keyed,
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
