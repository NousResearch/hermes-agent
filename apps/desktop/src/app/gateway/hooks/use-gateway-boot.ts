import { isGatewayReauthRequired, JsonRpcGatewayError, resolveGatewayWsUrl } from '@hermes/shared'
import { useEffect, useRef } from 'react'

import { shouldApplyPostBootProgressError } from '@/components/boot-failure-reauth'
import type { HermesConnection } from '@/global'
import { HermesGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { desktopDefaultCwd } from '@/lib/desktop-fs'
import { reconnectBackoffDelayMs } from '@/lib/reconnect-backoff'
import {
  $desktopBoot,
  applyDesktopBootProgress,
  completeDesktopBoot,
  failDesktopBoot,
  resumeDesktopBootForRetry,
  setDesktopBootStep
} from '@/store/boot'
import {
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  ensureGatewayForProfile,
  gatewayActivationEpoch,
  isActivePrimary,
  liveSecondaryConnectionIds,
  pruneSecondaryGateways,
  reconnectSecondaryGateways,
  reportPrimaryGatewayState,
  setPrimaryGateway,
  touchSecondaryGateways
} from '@/store/gateway'
import { registerGatewayReconnect } from '@/store/gateway-reconnect'
import { $gatewaySwitching, wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeGatewayProfile,
  normalizeProfileKey,
  refreshActiveProfile,
  touchActiveGatewayBackend
} from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $sessions,
  ensureDefaultWorkspaceCwd,
  setConnection,
  setCurrentBranch,
  setCurrentCwd,
  setSessionsLoading
} from '@/store/session'
import {
  $attentionSessionIds,
  $workingSessionIds,
  liveSessionScopes,
  reconcileBusyStatesOnReconnect,
  recordSessionEventScope,
  resetTileRuntimeBindings
} from '@/store/session-states'
import { windowProfileOverride } from '@/store/windows'
import type { RpcEvent } from '@/types/hermes'

import { stashGatewaySurvivor, survivorIsStale, takeGatewaySurvivor } from './gateway-hmr-survivor'

// After the reconnect loop has been failing for this long, raise a NON-blocking
// warning toast. Full-screen BootFailureOverlay used to lock the user out of
// reading/drafting for the whole blip even though the transcript is still on
// screen underneath. Confirmed reauth still escalates to the overlay (Sign in
// is required). Time-based (not attempt-count) because full-jitter backoff
// makes attempt counts a meaningless clock.
//
// 5 minutes (not the historical ~45s) so brief transport weather — ticket mint
// flaps, sleep/wake, Wi‑Fi blips that self-heal in 1–3 minutes — never even
// toast. Chat stays readable/draftable the whole time either way.
const RECONNECT_ESCALATE_AFTER_MS = 300_000

// Bound for the sleep/wake liveness probe (see reconnectNow): long enough to
// ride out a busy-but-healthy backend's scheduling jitter, short enough that a
// half-open socket fails fast instead of hanging the wake path. Independent of
// PROMPT_SUBMIT_REQUEST_TIMEOUT_MS (30 min) — that long timeout is correct for
// an in-flight turn, but must never be what a dead connection burns.
const GATEWAY_LIVENESS_PROBE_TIMEOUT_MS = 5_000

// Bounded self-heal for a failed REMOTE boot (#82679): when the primary boot
// fails on a transient remote fault (dropped SSH/HTTP registered connection,
// mint timeout — main tags those `retryable` on the boot progress), the
// renderer re-attempts the whole boot with the same full-jitter backoff the
// post-boot reconnect loop uses, up to this many attempts. Retries are
// bounded and end in the real recovery affordance (the boot-failure overlay
// with Retry / Settings), never an infinite spinner. Local failures and
// confirmed reauth rejections never enter this loop — a missing capability
// differs from a transient failure.
const BOOT_RETRY_MAX_ATTEMPTS = 5
// Base delay for boot retries. Deliberately slower than the socket reconnect
// loop's 300ms: each attempt may rebuild an SSH master + remote dashboard.
const BOOT_RETRY_BASE_DELAY_MS = 2_000

// desktop.revalidateConnection() / getConnection() / resolveGatewayWsUrl() are
// IPC round-trips into the main process with no timeout of their own (#93454).
// A remote backend that looks alive to a fresh probe but leaves the
// main-process reconnect path stuck (e.g. a wedged revalidation after a
// liveness-probe trip) hangs these awaits forever. While any is pending,
// `reconnecting` never clears, so scheduleReconnect()/attemptReconnect()
// early-return permanently and the backoff loop is latched — the UI stays
// "reconnecting" until the app is restarted even though the gateway is
// reachable again. Bound all three so a stall rejects instead, letting the
// existing catch/finally clear the guard and resume backoff. gateway.connect()
// already has its own connect timeout.
const RECONNECT_ATTEMPT_TIMEOUT_MS = 20_000

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms)

    promise.then(
      value => {
        clearTimeout(timer)
        resolve(value)
      },
      err => {
        clearTimeout(timer)
        reject(err)
      }
    )
  })
}

/** Registry identity whose runtimes died with the primary connection. */
export function primaryRuntimeConnectionId(connection: Pick<HermesConnection, 'connectionId' | 'mode'>): null | string {
  const connectionId = connection.connectionId?.trim()

  if (connectionId) {
    return connectionId
  }

  return connection.mode === 'local' ? 'local' : null
}

interface GatewayBootOptions {
  beforeConnectionSwitch: () => void
  handleGatewayEvent: (event: RpcEvent) => void
  onConnectionReady: (
    connection: Awaited<ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>> | null
  ) => void
  onGatewayReady: (gateway: HermesGateway | null) => void
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
}

export function useGatewayBoot({
  beforeConnectionSwitch,
  handleGatewayEvent,
  onConnectionReady,
  onGatewayReady,
  refreshHermesConfig,
  refreshSessions
}: GatewayBootOptions) {
  const callbacksRef = useRef({
    beforeConnectionSwitch,
    handleGatewayEvent,
    onConnectionReady,
    onGatewayReady,
    refreshHermesConfig,
    refreshSessions
  })

  callbacksRef.current = {
    beforeConnectionSwitch,
    handleGatewayEvent,
    onConnectionReady,
    onGatewayReady,
    refreshHermesConfig,
    refreshSessions
  }

  useEffect(() => {
    let cancelled = false
    const desktop = window.hermesDesktop

    const publish = (next: HermesConnection | null) => {
      callbacksRef.current.onConnectionReady(next)
      setConnection(next)
    }

    if (!desktop) {
      failDesktopBoot('Desktop IPC bridge is unavailable.')
      setSessionsLoading(false)

      return () => void (cancelled = true)
    }

    // --- Reconnect-after-sleep machinery -------------------------------------
    // macOS sleep silently drops the renderer's WebSocket. The backend Python
    // process keeps running, but nothing re-opened the socket on wake, so the
    // composer stayed disabled forever on "Starting Hermes...". Once the
    // initial boot succeeds we treat any non-open state as recoverable and
    // reconnect with backoff, and we nudge a reconnect on the OS/browser
    // signals that fire around wake (power resume, network online, the window
    // becoming visible).
    let bootCompleted = false
    let reconnecting = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let reconnectAttempt = 0
    // Wall-clock start of the current disconnect episode (first failed
    // reconnect attempt); null while healthy. Drives the time-based
    // escalation below. Reset on a clean open or a manual/wake reconnect.
    let reconnectFailingSince: number | null = null
    // Surface "sign in again" once per disconnect episode, not on every backoff
    // tick — a stale OAuth ticket fails every attempt and would otherwise stack
    // identical error toasts (and their haptics). Reset on the next clean open.
    let reauthNotified = false
    // Raised once the reconnect loop has been failing for
    // RECONNECT_ESCALATE_AFTER_MS so we fire a single non-blocking toast.
    // Reset on a clean open or a manual/wake-driven reconnect.
    let escalated = false
    // Bounded automatic boot retry for transient REMOTE failures (#82679).
    let bootRetryAttempt = 0
    let bootRetryTimer: ReturnType<typeof setTimeout> | null = null

    const clearBootRetryTimer = () => {
      if (bootRetryTimer !== null) {
        clearTimeout(bootRetryTimer)
        bootRetryTimer = null
      }
    }

    // Whether the failed boot is a TRANSIENT remote fault main marked as
    // retryable (dropped SSH/HTTP registered connection, mint timeout).
    // Local failures and confirmed reauth rejections come back false and go
    // straight to the recovery overlay.
    const bootFailureIsRetryable = async (): Promise<boolean> => {
      try {
        const snapshot = await desktop.getBootProgress()

        return snapshot?.retryable === true
      } catch {
        return false
      }
    }

    // Wrap the live getter in a call so TS control-flow analysis doesn't narrow
    // `connectionState` to a constant across the early-return guards (the state
    // genuinely changes between reads).
    const gatewayOpen = () => gateway.connectionState === 'open'

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    }

    const attemptReconnect = async () => {
      if (cancelled || reconnecting || gatewayOpen() || $gatewaySwitching.get()) {
        return
      }

      reconnecting = true

      try {
        // Drop a stale REMOTE backend cache before re-dialing. After sleep/wake a
        // remote backend can become unreachable, but it has no child process
        // whose 'exit' would clear the main process's cached descriptor — without
        // this the renderer re-dials the same dead endpoint forever and stays on
        // "Starting Hermes…". The probe is a no-op for a healthy or local backend.
        // Bounded like the two awaits below: a wedged revalidation (#93454) is
        // the specific hang this loop must survive, not just a rejection.
        await withTimeout(
          desktop.revalidateConnection?.() ?? Promise.resolve(),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out revalidating the gateway connection'
        ).catch(() => undefined)

        // Primary sleep/wake reconnect must dial the WINDOW-owned primary backend
        // (same as boot/softSwitch). Passing $activeGatewayProfile would retarget
        // this primary socket at a secondary profile's backend after a live swap.
        // Secondaries reconnect via reconnectSecondaryGateways().
        const conn = await withTimeout(
          desktop.getConnection(),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out reconnecting to Hermes backend'
        )

        if (cancelled) {
          return
        }

        // Only publish the primary descriptor when the primary is active.
        // Otherwise a background-profile view would inherit the primary's
        // mode/baseUrl and break image.attach / fs / media routing (#46651).
        if (isActivePrimary()) {
          publish(conn)
        }

        // Re-mint the WS URL before reconnecting. OAuth tickets are single-use
        // with a short TTL, so the ticket baked into the cached conn.wsUrl is
        // dead on every reconnect after the initial boot — reusing it surfaces
        // as an opaque "Could not connect to Hermes gateway". resolveGatewayWsUrl
        // mints a fresh ticket rather than connecting with a stale one. An
        // explicit auth rejection asks for sign-in; transport failures stay in
        // this reconnect loop. For local/token gateways the URL carries a
        // long-lived token and the re-mint is a cheap no-op.
        const wsUrl = await withTimeout(
          resolveGatewayWsUrl(desktop, conn),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out re-minting the gateway WebSocket URL'
        )

        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        reconnectAttempt = 0
        reconnectFailingSince = null
        // A respawned backend re-mints (recycles) runtime ids, so any tile's
        // bound runtime id is now stale — drop them so each tile re-resumes.
        // A legacy remote primary has no registry identity to scope by; fall
        // back to preserving only Bot runtimes owned by provably-live
        // secondaries so the restarted backend's own tiles still rebind.
        resetTileRuntimeBindings(
          primaryRuntimeConnectionId(conn) ?? { liveConnectionIds: liveSecondaryConnectionIds() }
        )
        // Same staleness, other half: pre-reconnect busy flags are keyed by
        // those dead runtime ids and would never receive their terminal
        // busy:false — clear them or the sidebar running arc lies forever
        // (#53902/#73082). A genuinely live turn re-asserts busy on its next
        // post-reconnect event.
        reconcileBusyStatesOnReconnect()
        // Resync state that may have moved on the backend while we were asleep.
        await callbacksRef.current.refreshHermesConfig().catch(() => undefined)
        await callbacksRef.current.refreshSessions().catch(() => undefined)
      } catch (err) {
        // OAuth session expired mid-reconnect: surface the actionable "sign in
        // again" recovery overlay once instead of silently looping the backoff
        // against a ticket that can never succeed. Transport failures fall
        // through to the backoff in the finally block below — they must NOT
        // take the full-screen "couldn't start" path (locks reading/drafting).
        if (!cancelled && isGatewayReauthRequired(err) && !reauthNotified) {
          reauthNotified = true
          const message = err instanceof Error ? err.message : String(err)
          failDesktopBoot(message)
          notifyError(err, translateNow('boot.errors.gatewaySignInRequired'))
        }
      } finally {
        reconnecting = false

        if (!cancelled && !gatewayOpen() && !$gatewaySwitching.get()) {
          if (reconnectFailingSince === null) {
            reconnectFailingSince = Date.now()
          }

          if (Date.now() - reconnectFailingSince >= RECONNECT_ESCALATE_AFTER_MS && !escalated) {
            escalated = true
            // Non-blocking: chat stays readable/draftable while we keep retrying.
            // Settings / Gateway menu remain reachable without a modal lockout.
            notify({
              kind: 'warning',
              title: translateNow('boot.errors.gatewayConnectionLost'),
              message: translateNow('boot.errors.gatewayConnectionLostDetail'),
              durationMs: 0
            })
          }

          scheduleReconnect()
        }
      }
    }

    function scheduleReconnect() {
      if (cancelled || reconnecting || reconnectTimer !== null || gatewayOpen() || $gatewaySwitching.get()) {
        return
      }

      // Full-jitter exponential backoff (300ms base, 15s cap) so a gateway
      // restart doesn't get redialed by every desktop client in lockstep —
      // an immediate-retry reconnect storm can exhaust the gateway's file
      // descriptors while it's still coming back up.
      const delay = reconnectBackoffDelayMs(reconnectAttempt)
      reconnectAttempt += 1
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        void attemptReconnect()
      }, delay)
    }

    const reconnectNow = async ({ forceOpenSocket = false }: { forceOpenSocket?: boolean } = {}) => {
      if (cancelled || !bootCompleted || $gatewaySwitching.get()) {
        return
      }

      clearReconnectTimer()
      reconnectAttempt = 0
      reconnectFailingSince = null
      escalated = false
      reconnectSecondaryGateways({ forceOpenSockets: forceOpenSocket })

      // Browser WebSocket state can remain OPEN after sleep even though the OS
      // discarded the underlying TCP connection. Strong recovery signals used
      // to blind-close here, but that churned perfectly healthy connections on
      // every wake/online blip — the liveness probe below now decides, closing
      // only a socket that is provably dead.

      if (!gatewayOpen()) {
        await attemptReconnect()
        return
      }

      // The socket reports open, but sleep/wake (or a silent network drop)
      // can leave a half-open TCP connection: no close event fires, so
      // connectionState stays 'open' while every RPC hangs until its per-call
      // timeout — prompt.submit's is 30 minutes, which reads as "enter does
      // nothing until I restart the app". Probe liveness with a short-bounded
      // ping; on failure force the socket down so the onState handler above
      // schedules a reconnect (and resetTileRuntimeBindings re-resumes tiles),
      // instead of letting the user's next submit hang against a dead socket.
      try {
        await gateway.request('ping', {}, GATEWAY_LIVENESS_PROBE_TIMEOUT_MS)
      } catch (probeErr) {
        // A version-skewed backend that predates the ping method answers
        // -32601 (method not found) — a HEALTHY response, not a dead socket.
        // Force-closing on it would spin the reconnect loop forever. Every
        // other failure (timeout on a swallowed ping, transport error) means
        // the socket is not actually alive and must be rebuilt.
        if (probeErr instanceof JsonRpcGatewayError && probeErr.code === -32601) {
          return
        }

        gateway.close()
      }
    }

    // Adopt the profile the primary (window) backend booted as, so same-profile
    // resumes are no-op swaps and reconnects target the right backend.
    // Best-effort: a missing preference means "default". Shared by boot + soft
    // switch.
    //
    // Helper windows (the HUD) can carry an explicit profile override in their
    // URL: the HUD is opened ON a conversation, and when that conversation
    // belongs to a non-primary profile, adopting the primary here resolves the
    // session id against the wrong backend — the HUD then falls back to the
    // default profile's last session (#82285). The override wins over the
    // stored preference; absent, behavior is unchanged.
    async function adoptPrimaryProfile() {
      const override = windowProfileOverride()

      try {
        const profileKey = override ?? (await desktop.profile?.get?.())?.profile ?? ''
        const key = normalizeProfileKey(profileKey)
        $activeGatewayProfile.set(key)
        setPrimaryGateway(gateway, key)
        void ensureGatewayForProfile(key)
      } catch {
        $activeGatewayProfile.set(normalizeProfileKey(override))
      }
    }

    // Seed the working dir from the backend default on a fresh view (nothing
    // open yet). Shared by boot + soft switch.
    async function seedDefaultCwd() {
      await ensureDefaultWorkspaceCwd()
      const remoteDefault = await desktopDefaultCwd().catch(() => null)

      if (remoteDefault?.cwd && !$activeSessionId.get() && !$currentCwd.get()) {
        setCurrentCwd(remoteDefault.cwd)
        setCurrentBranch(remoteDefault.branch || '')
      }
    }

    // Soft gateway-mode apply: main tore down the primary without reloading.
    // Wipe session lists so skeletons retrigger, then re-dial in place.
    const softSwitch = async () => {
      if (cancelled) {
        return
      }

      $gatewaySwitching.set(true)
      clearReconnectTimer()
      clearBootRetryTimer()
      bootRetryAttempt = 0
      reconnectAttempt = 0
      reconnectFailingSince = null
      escalated = false
      reauthNotified = false
      callbacksRef.current.beforeConnectionSwitch()
      wipeSessionListsForGatewaySwitch()

      try {
        gateway.close()
        closeSecondaryGateways()

        // Same override rule as boot(): a profile-pinned helper window stays
        // on its pinned profile's backend across a soft switch.
        const conn = await desktop.getConnection(windowProfileOverride() ?? undefined)

        if (cancelled) {
          return
        }

        publish(conn)
        // Bounded for the same reason as attemptReconnect() (#93454): a wedged
        // ticket mint would otherwise hang the gateway switch forever.
        const wsUrl = await withTimeout(
          resolveGatewayWsUrl(desktop, conn),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out re-minting the gateway WebSocket URL'
        )
        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        // Same shape as boot(): profile first (session scope depends on it),
        // then the independent fetches concurrently. refreshActiveProfile is
        // explicit here: the rail's $profiles still shows the PREVIOUS
        // backend's list after a connection/mode apply, and nothing else
        // re-pulls /api/profiles deterministically post-switch — leaving the
        // rail stale or (if a stale in-flight response landed) collapsed
        // (#85731). Best-effort like the rest: a failure keeps the cached
        // list rather than blanking the rail.
        await adoptPrimaryProfile()
        await Promise.all([
          seedDefaultCwd(),
          refreshActiveProfile().catch(() => undefined),
          callbacksRef.current.refreshHermesConfig().catch(() => undefined),
          callbacksRef.current.refreshSessions().catch(() => undefined)
        ])
        completeDesktopBoot()
        bootCompleted = true
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err)
          failDesktopBoot(message)
          notifyError(err, translateNow('boot.errors.desktopBootFailed'))
          setSessionsLoading(false)
        }
      } finally {
        $gatewaySwitching.set(false)
      }
    }

    const offBootProgress = desktop.onBootProgress(payload => {
      // Soft switch / post-boot startHermes re-emits progress — ignore so the
      // cold-boot CONNECTING overlay stays down. Post-boot errors are gated:
      // only confirmed reauth takes the full-screen recovery surface. Transient
      // ticket-mint / host-unreachable failures must stay in the reconnect loop
      // (otherwise a 1–3 min blip bricks reading/drafting behind "couldn't start").
      if ($gatewaySwitching.get() || bootCompleted) {
        if (payload.error && shouldApplyPostBootProgressError(payload.error)) {
          applyDesktopBootProgress(payload)
        }

        return
      }

      applyDesktopBootProgress(payload)
    })

    void desktop
      .getBootProgress()
      .then(snapshot => applyDesktopBootProgress(snapshot))
      .catch(() => undefined)

    setDesktopBootStep({
      phase: 'renderer.boot',
      message: translateNow('boot.steps.startingDesktopConnection'),
      progress: 6
    })

    // HMR adoption: in a dev hot update, the previous effect instance parked its
    // still-open socket instead of closing it (see the cleanup below). Re-adopt
    // it so an edit doesn't drop the live agent session. A stale (closed) parked
    // socket is discarded and we boot fresh. No-op in production: import.meta.hot
    // is undefined there, so this folds to `null` and the whole survivor module
    // dead-code-eliminates out of the bundle.
    const survivor = import.meta.hot ? takeGatewaySurvivor() : null
    const adoptedFromHmr = Boolean(survivor && !survivorIsStale(survivor))

    if (survivor && !adoptedFromHmr) {
      // Parked socket died between edits (e.g. backend restart) — release it.
      try {
        survivor.gateway.close()
      } catch {
        // ignore
      }
    }

    const gateway = adoptedFromHmr ? survivor!.gateway : new HermesGateway()

    callbacksRef.current.onGatewayReady(gateway)
    setPrimaryGateway(gateway, survivor?.profile ?? normalizeProfileKey($activeGatewayProfile.get()))
    // Secondary (background-profile) sockets funnel into the same handler.
    // Record each event's source scope first: registry-tagged events feed the
    // (connectionId, profile) keep-set so two sources exposing the same
    // profile name (every source has a 'default') can't collide.
    configureGatewayRegistry({
      onActiveConnectionChanged: publish,
      // Keep $activeGatewayProfile in lockstep with the registry's OWN record
      // of which profile the active socket serves. The registry is the only
      // party that sees eviction fallbacks (idle reap, connection removal,
      // profile delete → primary); before this mirror those fallbacks moved
      // the SOCKET back to the primary while the profile atom kept naming the
      // evicted bot. ensureGatewayProfile's "already active" fast path then
      // trusted the stale atom and skipped the re-swap, so every
      // session-scoped RPC for that bot went out on the primary socket — the
      // #89206 "Waking up… → retries gave up" wake failure, while the bot's
      // own backend sat healthy and idle.
      onActiveRouteChanged: profile => {
        const key = normalizeProfileKey(profile)

        if (normalizeProfileKey($activeGatewayProfile.get()) !== key) {
          $activeGatewayProfile.set(key)
        }
      },
      onEvent: event => {
        recordSessionEventScope(event)
        callbacksRef.current.handleGatewayEvent(event)
      },
      onActiveConnectionInvalidated: (fallbackProfile, invalidationEpoch) => {
        $activeGatewayProfile.set(fallbackProfile)
        void desktop
          .getConnection(fallbackProfile)
          .then(connection => {
            if (!cancelled && gatewayActivationEpoch() === invalidationEpoch) {
              publish(connection)
            }
          })
          .catch(() => {
            if (!cancelled && gatewayActivationEpoch() === invalidationEpoch) {
              publish(null)
            }
          })
      }
    })

    const offState = gateway.onState(st => {
      // Mirror to the composer only while the primary is the active profile —
      // a background secondary reconnect mustn't flip the foreground state.
      reportPrimaryGatewayState(st)

      if (st === 'open') {
        reconnectAttempt = 0
        reconnectFailingSince = null
        reauthNotified = false
        escalated = false
        clearReconnectTimer()

        // A revalidate-driven reconnect can rebuild the backend in place when the
        // cached remote was found dead, which re-drives the boot-progress overlay.
        // Unlike the initial boot, nothing calls completeDesktopBoot() afterwards,
        // so dismiss it here once we're open again — otherwise the overlay sticks
        // at ~94%. A no-op on a normal (non-rebuild) reconnect.
        if (bootCompleted) {
          completeDesktopBoot()
        }
      } else if (bootCompleted && !$gatewaySwitching.get() && (st === 'closed' || st === 'error')) {
        // The socket dropped after a healthy boot (typically sleep/wake). Try
        // to bring it back instead of leaving the composer stuck disabled.
        scheduleReconnect()
      }
    })

    const sourceProfile = normalizeProfileKey($activeGatewayProfile.get())

    const offEvent = gateway.onEvent(event =>
      callbacksRef.current.handleGatewayEvent({ ...event, profile: sourceProfile })
    )

    // Wake signals: power resume (macOS/Windows), network coming back, and the
    // window regaining focus/visibility. Each nudges an immediate reconnect.
    const forceReconnectNow = () => reconnectNow({ forceOpenSocket: true })
    const offPowerResume = desktop.onPowerResume?.(() => void forceReconnectNow())
    const offConnectionApplied = desktop.onConnectionApplied?.(() => void softSwitch())
    const offGatewayReconnect = registerGatewayReconnect(forceReconnectNow)

    // Registry lifecycle: a removed connection's secondaries must close NOW
    // (remote/cloud have no local process whose death would drop the socket —
    // they'd keep streaming ghost events); a materially edited one is
    // disposed AND re-dialed so its sockets target the new endpoint.
    const offConnectionsChanged = desktop.connections?.onChanged?.(payload => {
      if (!payload || typeof payload.connectionId !== 'string') {
        return
      }

      disposeSecondariesForConnection(payload.connectionId, { redial: payload.reason === 'updated' })
    })

    const onOnline = () => void forceReconnectNow()

    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        void reconnectNow()
      }
    }

    window.addEventListener('online', onOnline)
    document.addEventListener('visibilitychange', onVisible)

    // Keep live pool backends alive while this window is open (the main process
    // can't observe the direct renderer↔backend WS). No-op for the primary.
    const keepaliveTimer = setInterval(() => {
      touchActiveGatewayBackend()
      touchSecondaryGateways()
    }, 60_000)

    // Bound concurrency cost to live work: keep a background socket only while
    // its profile has a running (working) or blocked (needs-input) session.
    // Once that profile goes idle its socket is dropped and its backend is free
    // to idle-reap. The active profile is always spared.
    const recomputeKeptGateways = () => {
      const live = new Set([...$workingSessionIds.get(), ...$attentionSessionIds.get()])
      // Registry-scoped (connectionId, profile) scopes with live work. Two
      // sources can expose the same profile name (every source has a
      // 'default'), so bare profile names can't represent a non-local
      // source's liveness without keeping the wrong gateway alive.
      const keep = liveSessionScopes()

      for (const session of $sessions.get()) {
        if (live.has(session.id)) {
          keep.add(normalizeProfileKey(session.profile))
        }
      }

      pruneSecondaryGateways(keep)
    }

    const offWorking = $workingSessionIds.subscribe(() => recomputeKeptGateways())
    const offAttention = $attentionSessionIds.subscribe(() => recomputeKeptGateways())
    const offActiveProfile = $activeGatewayProfile.subscribe(() => recomputeKeptGateways())

    const offWindowState = desktop.onWindowStateChanged?.(payload => {
      const current = $connection.get()

      if (current) {
        publish({ ...current, ...payload })
      }
    })

    const offExit = desktop.onBackendExit(() => {
      if ($gatewaySwitching.get()) {
        return
      }

      if ($desktopBoot.get().running || $desktopBoot.get().visible) {
        failDesktopBoot(translateNow('boot.errors.backgroundExitedDuringStartup'))
      }

      notify({
        kind: 'error',
        title: translateNow('boot.errors.backendStopped'),
        message: translateNow('boot.errors.backgroundExited'),
        durationMs: 0
      })
    })

    async function boot() {
      try {
        // A profile-pinned helper window (the HUD) dials its target profile's
        // backend directly — ensureBackend spawns/reuses it from the pool.
        // Everything else keeps dialing the primary.
        const conn = await desktop.getConnection(windowProfileOverride() ?? undefined)

        if (cancelled) {
          return
        }

        setDesktopBootStep({
          phase: 'renderer.gateway.connect',
          message: translateNow('boot.steps.connectingGateway'),
          progress: 95
        })
        publish(conn)

        // Seed the workspace BEFORE the gateway opens: every session-restore
        // path is gated on gatewayState === 'open', so nothing can be active yet
        // and ensureDefaultWorkspaceCwd's live-session guard passes. The
        // post-connect seed could lose that race on a slow start (#71873). A
        // resumed session's own cwd still supersedes this once its runtime
        // arrives. Non-fatal: the remembered cwd is a fine fallback and the
        // post-connect pass retries the sync.
        try {
          await ensureDefaultWorkspaceCwd()
        } catch (err) {
          console.warn('Failed to seed default workspace cwd pre-connect', err)
        }

        // Mint a fresh WS URL right before connecting. For OAuth gateways the
        // ticket is single-use with a short TTL, so the ticket baked into
        // conn.wsUrl is stale; resolveGatewayWsUrl() re-mints it rather than
        // connecting with a dead ticket. Auth rejection asks for sign-in;
        // connectivity failures remain retryable. Bounded like the reconnect
        // path (#93454) so a wedged mint fails into boot retry instead of
        // hanging "Starting Hermes…" forever.
        const wsUrl = await withTimeout(
          resolveGatewayWsUrl(desktop, conn),
          RECONNECT_ATTEMPT_TIMEOUT_MS,
          'Timed out minting the gateway WebSocket URL'
        )
        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        // Profile adoption must land first: refreshSessions scopes its fetch by
        // $profileScope ← $activeGatewayProfile. The remaining three fetches
        // (cwd seed, config, sessions) are independent REST calls — running
        // them serially added their sum to time-to-populated-sidebar when only
        // the max is needed.
        await adoptPrimaryProfile()

        setDesktopBootStep({
          phase: 'renderer.config',
          message: translateNow('boot.steps.loadingSettings'),
          progress: 97
        })

        await Promise.all([
          // The pre-connect seed already applied the configured default; this
          // post-connect pass covers the remote backend default. Non-fatal: a
          // failed sync must not abort boot (the remembered cwd remains).
          seedDefaultCwd().catch(err => console.warn('Failed to sync default workspace cwd post-connect', err)),
          callbacksRef.current.refreshHermesConfig(),
          // Session-list population is never boot-fatal. The gateway WS is
          // already open by this point — a failed sidebar fetch (transient
          // blip, or an endpoint the fallback couldn't cover) must leave the
          // app usable with an empty sidebar (the reconnect/turn refreshes
          // retry it), not brick boot behind the "Hermes couldn't start"
          // overlay. Matches the reconnect + softSwitch call sites.
          callbacksRef.current.refreshSessions().catch(() => {
            setSessionsLoading(false)
          })
        ])

        if (cancelled) {
          return
        }

        completeDesktopBoot()
        bootCompleted = true
        bootRetryAttempt = 0
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err)

          // Transient remote failure (dropped SSH/HTTP registered connection,
          // mint timeout): self-heal with bounded, jittered retries instead of
          // parking on "Desktop boot failed" until the user re-enters the same
          // connection details (#82679). Main already cleared the failed cached
          // descriptor, so the next getConnection() rebuilds the connection —
          // exactly what manual re-entry forced. Exhausted retries, local
          // failures, and confirmed reauth rejections end in the real recovery
          // affordance (the boot-failure overlay), never an infinite spinner.
          if (bootRetryAttempt < BOOT_RETRY_MAX_ATTEMPTS && (await bootFailureIsRetryable()) && !cancelled) {
            const delay = reconnectBackoffDelayMs(bootRetryAttempt, { baseDelayMs: BOOT_RETRY_BASE_DELAY_MS })
            bootRetryAttempt += 1
            resumeDesktopBootForRetry(translateNow('boot.steps.retryingRemoteBackend'))
            clearBootRetryTimer()
            bootRetryTimer = setTimeout(() => {
              bootRetryTimer = null
              void boot()
            }, delay)

            return
          }

          failDesktopBoot(message)
          notifyError(err, translateNow('boot.errors.desktopBootFailed'))
          setSessionsLoading(false)
        }
      }
    }

    // Adopt the parked socket without re-running the full boot handshake: the
    // socket is already open, the backend session is untouched, and we already
    // know the profile. We only re-publish the connection, re-sync config +
    // sessions (cheap, and the backend may have moved on between edits), and
    // dismiss any boot overlay. This is what keeps a live, mid-stream session
    // intact across an HMR update.
    async function adoptBoot() {
      bootCompleted = true
      completeDesktopBoot()

      if (survivor?.connection) {
        publish(survivor.connection)
      }

      const profile = survivor?.profile ?? $activeGatewayProfile.get()
      $activeGatewayProfile.set(profile)
      void ensureGatewayForProfile(profile)

      // Mirror the current (already-open) socket state into the composer so the
      // input doesn't sit disabled after the swap.
      reportPrimaryGatewayState(gateway.connectionState)

      await callbacksRef.current.refreshHermesConfig().catch(() => undefined)

      if (cancelled) {
        return
      }

      await callbacksRef.current.refreshSessions().catch(() => undefined)
    }

    if (adoptedFromHmr) {
      void adoptBoot()
    } else {
      void boot()
    }

    return () => {
      cancelled = true
      $gatewaySwitching.set(false)
      clearReconnectTimer()
      clearBootRetryTimer()
      clearInterval(keepaliveTimer)
      offWorking()
      offAttention()
      offActiveProfile()
      window.removeEventListener('online', onOnline)
      document.removeEventListener('visibilitychange', onVisible)
      offPowerResume?.()
      offConnectionApplied?.()
      offConnectionsChanged?.()
      offGatewayReconnect()
      offState()
      offEvent()
      offExit()
      offWindowState?.()
      offBootProgress()

      // HMR teardown vs. real unmount. On a hot update we must NOT close the
      // socket — that's the whole bug. Detach this instance's listeners (their
      // closures capture the disposed module), park the still-open gateway, and
      // let the freshly loaded effect re-adopt it. Secondaries are owned by the
      // gateway store (HMR-stable module state), so they survive untouched.
      // Production: import.meta.hot is undefined, so this branch never runs and
      // the original destructive teardown below is byte-for-byte preserved.
      if (import.meta.hot && gateway.connectionState === 'open') {
        stashGatewaySurvivor({
          gateway,
          profile: survivor?.profile ?? $activeGatewayProfile.get(),
          connection: $connection.get()
        })

        return
      }

      closeSecondaryGateways()
      gateway.close()
      publish(null)
      callbacksRef.current.onGatewayReady(null)
      setPrimaryGateway(null)
      $gateway.set(null)
    }
  }, [])
}
