import { atom } from 'nanostores'

import { $apiRequestScope } from '@/api/client'
import { getActionStatus, getStatus, type ResolvedOwner, resolveOwnerNow, restartGateway, startGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { sharedGatewayProfiles } from '@/lib/shared-gateway-restart'
import { confirm } from '@/store/confirm'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { notify, notifyError } from '@/store/notifications'
import type { ActionResponse } from '@/types/hermes'

const POLL_ATTEMPTS = 18
const POLL_INTERVAL_MS = 1200
const GATEWAY_RESTART_ACTION = 'gateway-restart'

// Resolve once, and keep invalidation sticky across A→B→A. The authority
// subscription's immediate first notification equals the captured origin.
function captureActionOwner(owner = resolveOwnerNow()) {
  const origin = $apiRequestScope.get()
  let current = true

  const dispose = $apiRequestScope.subscribe(scope => {
    if (scope.connectionId !== origin.connectionId || scope.profile !== origin.profile) {
      current = false
    }
  })

  return {
    owner,
    dispose,
    isCurrent: () => current,
    canReconnect: () => {
      const active = resolveOwnerNow()

      return current && active.connectionId === owner.connectionId && active.profile === owner.profile
    }
  }
}

type ActionOwner = ReturnType<typeof captureActionOwner>

// True while a gateway restart is in flight — drives the statusbar gateway
// indicator (glyph spinner) so the restart shows up where users already look,
// instead of a toast that vanishes or a generic "Agents running" counter.
export const $gatewayRestarting = atom(false)

// Poll a backend action to completion (or a bounded window), throwing on a
// non-zero exit so the caller can surface the failure. In no-service installs
// the child becomes the foreground gateway and never exits, so "still running
// when the window closes" counts as success.
//
// `gateway-restart` is polled against the very backend the restart takes down:
// while the old process exits and the new one binds, requests refuse/fail and
// the in-memory action registry dies with the old process (a 404 or a fresh
// process's `running:false, exit_code:null` is a healthy mid-restart state,
// not a failure). Mid-window refusals are therefore tolerated (#123111) — but
// only an ANSWERED poll proves the replacement backend actually came back, so
// a window refused end to end is a failure, not a silent success: resolving it
// would erase the caller's "restart needed" banner while the gateway stays
// down.
async function awaitAction(
  name: string,
  owner: ResolvedOwner,
  failureMessage = translateNow('commandCenter.gatewayRestartFailed')
): Promise<void> {
  let sawAnsweredPoll = false
  let lastPollError: unknown = null

  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS))

    let status: Awaited<ReturnType<typeof getActionStatus>>

    try {
      status = await getActionStatus(name, 200, owner)
    } catch (err) {
      // The backend accepted the restart POST a moment ago and is now
      // refusing — the expected shape of the restart window itself.
      lastPollError = err

      continue
    }

    sawAnsweredPoll = true
    lastPollError = null

    if (!status.running && status.exit_code == null) {
      // A fresh process that never saw this action id answers exactly this
      // way; only a recorded non-zero exit is a failure (an entirely
      // unanswered window is handled after the loop).
      continue
    }

    if (!status.running) {
      if (status.exit_code != null && status.exit_code !== 0) {
        // The action endpoint retains the child process output. Keep its last
        // line with the toast so a failed restart names the actual cause
        // (bad credentials, a port conflict, and so on) instead of making the
        // user open gateway.log to learn why it failed.
        const cause = status.lines
          .map(line => line.trim())
          .filter(Boolean)
          .at(-1)

        throw new Error(cause || failureMessage)
      }

      return
    }
  }

  if (!sawAnsweredPoll) {
    throw lastPollError ?? new Error(failureMessage)
  }
}

// Under `gateway.multiplex_profiles` the profile in view has no gateway of its
// own: "Restart gateway" restarts the ONE shared multiplexer and every bot on
// this device blips. Ask first, naming them; standalone gateways (and older
// backends that do not report `gateway_shared_with`) keep the silent restart.
// Resolves the served list when the user confirmed, `null` when nothing is
// shared, `false` when they cancelled.
export async function confirmSharedGatewayRestart(owner?: ResolvedOwner): Promise<false | null | string[]> {
  let shared: null | string[] = null

  try {
    shared = sharedGatewayProfiles(await getStatus(owner))
  } catch {
    // Status unavailable: fall back to the plain restart rather than blocking it.
    return null
  }

  if (!shared) {
    return null
  }

  const ok = await confirm({
    title: translateNow('commandCenter.sharedGatewayRestartTitle'),
    description: translateNow('commandCenter.sharedGatewayRestartDescription', shared.join(', ')),
    confirmLabel: translateNow('commandCenter.sharedGatewayRestartConfirm'),
    cancelLabel: translateNow('common.cancel'),
    destructive: true
  })

  return ok ? shared : false
}

// Restart the messaging gateway, surfacing progress in the statusbar gateway
// indicator. Self-contained and never rejects, so every trigger — Cmd+K, the
// messaging save/toggle toasts — gets identical feedback from a plain
// `void runGatewayRestart()`, and a failure is the only thing that toasts.
// Resolves `true` when the restart child completed cleanly (callers that keep
// a "restart needed" banner clear it on that signal only).
export async function runGatewayRestart(owner = resolveOwnerNow()): Promise<boolean> {
  const context = captureActionOwner(owner)
  const shared = await confirmSharedGatewayRestart(context.owner)

  if (shared === false || !context.isCurrent()) {
    context.dispose()

    return false
  }

  return runGatewayAction(
    () => restartGateway(context.owner),
    context,
    translateNow('commandCenter.gatewayRestartFailed'),
    shared
  )
}

// Starting loads saved settings without interrupting an existing shared gateway.
export function runGatewayStart(owner = resolveOwnerNow()): Promise<boolean> {
  const context = captureActionOwner(owner)

  return runGatewayAction(() => startGateway(context.owner), context, translateNow('messaging.gatewayStartFailed'))
}

async function runGatewayAction(
  request: () => Promise<ActionResponse>,
  context: ActionOwner,
  failureMessage: string,
  shared: null | string[] = null
): Promise<boolean> {
  $gatewayRestarting.set(true)

  try {
    const started = await request()
    await awaitAction(started.name, context.owner, failureMessage)

    if (shared) {
      notify({ kind: 'success', message: translateNow('commandCenter.sharedGatewayRestarted', shared.length) })
    }

    return true
  } catch (err) {
    notifyError(err, failureMessage)

    return false
  } finally {
    $gatewayRestarting.set(false)

    // The restart took down the process serving this client's own WebSocket
    // (and any action-registry state with it), so leaving recovery to the
    // passive close→backoff machinery lets a Windows close-frame-less drop sit
    // as a zombie until the 45s heartbeat deadline — or until the user
    // relaunches the app. Hand reconnection to the owner that knows how, as a
    // RESTART follow-through: the owner probes the socket first, so a restart
    // that never touched this client's backend doesn't tear a healthy
    // connection down (and one that did gets rebuilt without the manual
    // path's unconditional close). A not-yet-registered handler or a
    // still-down backend rejects and is swallowed (the boot loop keeps
    // retrying regardless).
    if (context.canReconnect()) {
      void reconnectGateway({ source: 'restart-followthrough', isCurrent: context.canReconnect })
        .catch(() => undefined)
        .finally(context.dispose)
    } else {
      context.dispose()
    }
  }
}

// Watch a restart the BACKEND spawned (e.g. after Telegram QR onboarding writes
// credentials) instead of one this app requested. Same indicator, same bounded
// poll, same restart-window tolerance and reconnect follow-through.
export async function watchGatewayRestartOutcome(owner = resolveOwnerNow()): Promise<boolean> {
  const context = captureActionOwner(owner)
  $gatewayRestarting.set(true)

  try {
    await awaitAction(GATEWAY_RESTART_ACTION, context.owner)

    return true
  } catch {
    return false
  } finally {
    $gatewayRestarting.set(false)

    // Same restart follow-through as the requested flow above: probe-first
    // recovery, never the manual path's unconditional teardown.
    if (context.canReconnect()) {
      void reconnectGateway({ source: 'restart-followthrough', isCurrent: context.canReconnect })
        .catch(() => undefined)
        .finally(context.dispose)
    } else {
      context.dispose()
    }
  }
}
