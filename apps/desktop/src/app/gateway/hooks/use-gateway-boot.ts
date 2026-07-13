import { isGatewayReauthRequired, resolveGatewayWsUrl } from '@hermes/shared'
import { useEffect, useRef } from 'react'

import type { HermesConnection } from '@/global'
import { HermesGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { desktopDefaultCwd } from '@/lib/desktop-fs'
import {
  $desktopBoot,
  applyDesktopBootProgress,
  completeDesktopBoot,
  failDesktopBoot,
  setDesktopBootStep
} from '@/store/boot'
import {
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForProfile,
  pruneSecondaryGateways,
  reconnectSecondaryGateways,
  reportPrimaryGatewayState,
  setPrimaryGateway,
  touchSecondaryGateways
} from '@/store/gateway'
import { $gatewaySwitching, wipeSessionListsForGatewaySwitch } from '@/store/gateway-switch'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey, touchActiveGatewayBackend } from '@/store/profile'
import {
  $activeSessionId,
  $attentionSessions,
  $connection,
  $currentCwd,
  $workingSessions,
  ensureDefaultWorkspaceCwd,
  setConnection,
  setCurrentBranch,
  setCurrentCwd,
  setSessionsLoading
} from '@/store/session'
import type { SessionIdentity } from '@/store/session-identity'
import type { RpcEvent } from '@/types/hermes'

// After this many consecutive failed reconnects (≈45s with the 1→15s backoff)
// raise a recoverable boot error. Otherwise a dropped remote gateway loops the
// backoff forever behind the fullscreen CONNECTING overlay with no way to reach
// Settings / sign in / switch to local — the "lost connection breaks the app"
// dead end. The next successful reconnect clears it.
const RECONNECT_ESCALATE_AFTER = 6

export function profilesWithLiveSessionState(
  working: readonly SessionIdentity[],
  attention: readonly SessionIdentity[]
): Set<string> {
  return new Set([...working, ...attention].map(identity => normalizeProfileKey(identity.profile)))
}

interface GatewayBootOptions {
  handleGatewayEvent: (event: RpcEvent) => void
  onConnectionReady: (
    connection: Awaited<ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>> | null
  ) => void
  onGatewayReady: (gateway: HermesGateway | null) => void
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
}

export function useGatewayBoot({
  handleGatewayEvent,
  onConnectionReady,
  onGatewayReady,
  refreshHermesConfig,
  refreshSessions
}: GatewayBootOptions) {
  const callbacksRef = useRef({
    handleGatewayEvent,
    onConnectionReady,
    onGatewayReady,
    refreshHermesConfig,
    refreshSessions
  })

  callbacksRef.current = {
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
    // Surface "sign in again" once per disconnect episode, not on every backoff
    // tick — a stale OAuth ticket fails every attempt and would otherwise stack
    // identical error toasts (and their haptics). Reset on the next clean open.
    let reauthNotified = false
    // Raised once the reconnect loop crosses RECONNECT_ESCALATE_AFTER so the
    // recovery overlay replaces the dead-end CONNECTING screen. Reset on a clean
    // open or a manual/wake-driven reconnect.
    let escalated = false

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
        await desktop.revalidateConnection?.().catch(() => undefined)

        const conn = await desktop.getConnection($activeGatewayProfile.get())

        if (cancelled) {
          return
        }

        publish(conn)
        // Re-mint the WS URL before reconnecting. OAuth tickets are single-use
        // with a short TTL, so the ticket baked into the cached conn.wsUrl is
        // dead on every reconnect after the initial boot — reusing it surfaces
        // as an opaque "Could not connect to Hermes gateway". resolveGatewayWsUrl
        // mints a fresh ticket (or throws a reauth error in OAuth mode rather
        // than connecting with a stale one). For local/token gateways the URL
        // carries a long-lived token and the re-mint is a cheap no-op.
        const wsUrl = await resolveGatewayWsUrl(desktop, conn)
        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        reconnectAttempt = 0
        // Resync state that may have moved on the backend while we were asleep.
        await callbacksRef.current.refreshHermesConfig().catch(() => undefined)
        await callbacksRef.current.refreshSessions().catch(() => undefined)
      } catch (err) {
        // OAuth session expired mid-reconnect: surface the actionable "sign in
        // again" message once instead of silently looping the backoff against a
        // ticket that can never succeed. Transport failures fall through to the
        // backoff in the finally block below.
        if (!cancelled && isGatewayReauthRequired(err) && !reauthNotified) {
          reauthNotified = true
          notifyError(err, translateNow('boot.errors.gatewaySignInRequired'))
        }
      } finally {
        reconnecting = false

        if (!cancelled && !gatewayOpen() && !$gatewaySwitching.get()) {
          if (reconnectAttempt >= RECONNECT_ESCALATE_AFTER && !escalated) {
            escalated = true
            failDesktopBoot(translateNow('boot.errors.gatewayConnectionLost'))
          }

          scheduleReconnect()
        }
      }
    }

    function scheduleReconnect() {
      if (cancelled || reconnecting || reconnectTimer !== null || gatewayOpen() || $gatewaySwitching.get()) {
        return
      }

      // 1s, 2s, 4s … capped at 15s.
      const delay = Math.min(15_000, 1_000 * 2 ** Math.min(reconnectAttempt, 4))
      reconnectAttempt += 1
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        void attemptReconnect()
      }, delay)
    }

    const reconnectNow = () => {
      if (cancelled || !bootCompleted || $gatewaySwitching.get()) {
        return
      }

      clearReconnectTimer()
      reconnectAttempt = 0
      escalated = false
      reconnectSecondaryGateways()

      if (!gatewayOpen()) {
        void attemptReconnect()
      }
    }

    // Adopt the profile the primary (window) backend booted as, so same-profile
    // resumes are no-op swaps and reconnects target the right backend.
    // Best-effort: a missing preference means "default". Shared by boot + soft
    // switch.
    async function adoptPrimaryProfile() {
      try {
        const pref = await desktop.profile?.get?.()
        const profileKey = (pref?.profile ?? '').trim() || 'default'
        $activeGatewayProfile.set(profileKey)
        setPrimaryGateway(gateway, profileKey)
        void ensureGatewayForProfile(profileKey)
      } catch {
        $activeGatewayProfile.set('default')
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
      reconnectAttempt = 0
      escalated = false
      reauthNotified = false
      wipeSessionListsForGatewaySwitch()

      try {
        gateway.close()
        closeSecondaryGateways()

        const conn = await desktop.getConnection()

        if (cancelled) {
          return
        }

        publish(conn)
        const wsUrl = await resolveGatewayWsUrl(desktop, conn)
        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        await adoptPrimaryProfile()
        await seedDefaultCwd()
        await callbacksRef.current.refreshHermesConfig().catch(() => undefined)
        await callbacksRef.current.refreshSessions().catch(() => undefined)
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
      // cold-boot CONNECTING overlay stays down. Errors still surface.
      if ($gatewaySwitching.get() || bootCompleted) {
        if (payload.error) {
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

    const gateway = new HermesGateway()
    callbacksRef.current.onGatewayReady(gateway)
    setPrimaryGateway(gateway, normalizeProfileKey($activeGatewayProfile.get()))
    // Secondary (background-profile) sockets funnel into the same handler.
    configureGatewayRegistry({ onEvent: event => callbacksRef.current.handleGatewayEvent(event) })

    const offState = gateway.onState(st => {
      // Mirror to the composer only while the primary is the active profile —
      // a background secondary reconnect mustn't flip the foreground state.
      reportPrimaryGatewayState(st)

      if (st === 'open') {
        reconnectAttempt = 0
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
    const offEvent = gateway.onEvent(event => callbacksRef.current.handleGatewayEvent({ ...event, profile: sourceProfile }))

    // Wake signals: power resume (macOS/Windows), network coming back, and the
    // window regaining focus/visibility. Each nudges an immediate reconnect.
    const offPowerResume = desktop.onPowerResume?.(() => reconnectNow())
    const offConnectionApplied = desktop.onConnectionApplied?.(() => void softSwitch())

    const onOnline = () => reconnectNow()

    const onVisible = () => {
      if (document.visibilityState === 'visible') {
        reconnectNow()
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
    const recomputeKeptGateways = () =>
      pruneSecondaryGateways(profilesWithLiveSessionState($workingSessions.get(), $attentionSessions.get()))

    const offWorking = $workingSessions.subscribe(() => recomputeKeptGateways())
    const offAttention = $attentionSessions.subscribe(() => recomputeKeptGateways())
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
        const conn = await desktop.getConnection()

        if (cancelled) {
          return
        }

        setDesktopBootStep({
          phase: 'renderer.gateway.connect',
          message: translateNow('boot.steps.connectingGateway'),
          progress: 95
        })
        publish(conn)
        // Mint a fresh WS URL right before connecting. For OAuth gateways the
        // ticket is single-use with a short TTL, so the ticket baked into
        // conn.wsUrl is stale; resolveGatewayWsUrl() re-mints it and, on
        // failure, throws a reauth error rather than connecting with a dead
        // ticket (which would surface as an opaque "connection closed").
        const wsUrl = await resolveGatewayWsUrl(desktop, conn)
        await gateway.connect(wsUrl)

        if (cancelled) {
          return
        }

        await adoptPrimaryProfile()

        setDesktopBootStep({
          phase: 'renderer.config',
          message: translateNow('boot.steps.loadingSettings'),
          progress: 97
        })
        await seedDefaultCwd()

        await callbacksRef.current.refreshHermesConfig()

        if (cancelled) {
          return
        }

        setDesktopBootStep({
          phase: 'renderer.sessions',
          message: translateNow('boot.steps.loadingSessions'),
          progress: 99
        })
        await callbacksRef.current.refreshSessions()
        completeDesktopBoot()
        bootCompleted = true
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err)
          failDesktopBoot(message)
          notifyError(err, translateNow('boot.errors.desktopBootFailed'))
          setSessionsLoading(false)
        }
      }
    }

    void boot()

    return () => {
      cancelled = true
      $gatewaySwitching.set(false)
      clearReconnectTimer()
      clearInterval(keepaliveTimer)
      offWorking()
      offAttention()
      offActiveProfile()
      window.removeEventListener('online', onOnline)
      document.removeEventListener('visibilitychange', onVisible)
      offPowerResume?.()
      offConnectionApplied?.()
      offState()
      offEvent()
      offExit()
      offWindowState?.()
      offBootProgress()
      closeSecondaryGateways()
      gateway.close()
      publish(null)
      callbacksRef.current.onGatewayReady(null)
      setPrimaryGateway(null)
      $gateway.set(null)
    }
  }, [])
}
