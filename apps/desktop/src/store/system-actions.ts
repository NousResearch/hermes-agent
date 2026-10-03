import { atom } from 'nanostores'

import { getActionStatus, getStatus, restartGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { sharedGatewayProfiles } from '@/lib/shared-gateway-restart'
import { confirm } from '@/store/confirm'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { notify, notifyError } from '@/store/notifications'
import type { ActionResponse } from '@/types/hermes'

const POLL_ATTEMPTS = 18
const POLL_INTERVAL_MS = 1200
const GATEWAY_RESTART_ACTION = 'gateway-restart'

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
async function awaitAction(name: string): Promise<void> {
  let sawAnsweredPoll = false
  let lastPollError: unknown = null

  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS))

    let status: Awaited<ReturnType<typeof getActionStatus>>

    try {
      status = await getActionStatus(name)
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
        throw new Error(translateNow('commandCenter.gatewayRestartFailed'))
      }

      return
    }
  }

  if (!sawAnsweredPoll) {
    throw lastPollError ?? new Error(translateNow('commandCenter.gatewayRestartFailed'))
  }
}

// Under `gateway.multiplex_profiles` the profile in view has no gateway of its
// own: "Restart gateway" restarts the ONE shared multiplexer and every bot on
// this device blips. Ask first, naming them; standalone gateways (and older
// backends that do not report `gateway_shared_with`) keep the silent restart.
// Resolves the served list when the user confirmed, `null` when nothing is
// shared, `false` when they cancelled.
export async function confirmSharedGatewayRestart(): Promise<false | null | string[]> {
  let shared: null | string[] = null

  try {
    shared = sharedGatewayProfiles(await getStatus())
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
export async function runGatewayRestart(): Promise<boolean> {
  const shared = await confirmSharedGatewayRestart()

  if (shared === false) {
    return false
  }

  $gatewayRestarting.set(true)

  try {
    const started: ActionResponse = await restartGateway()
    await awaitAction(started.name)

    if (shared) {
      notify({ kind: 'success', message: translateNow('commandCenter.sharedGatewayRestarted', shared.length) })
    }

    return true
  } catch (err) {
    notifyError(err, translateNow('commandCenter.gatewayRestartFailed'))

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
    void reconnectGateway({ source: 'restart-followthrough' }).catch(() => undefined)
  }
}

// Watch a restart the BACKEND spawned (e.g. after Telegram QR onboarding writes
// credentials) instead of one this app requested. Same indicator, same bounded
// poll, same restart-window tolerance and reconnect follow-through.
export async function watchGatewayRestartOutcome(): Promise<boolean> {
  $gatewayRestarting.set(true)

  try {
    await awaitAction(GATEWAY_RESTART_ACTION)

    return true
  } catch {
    return false
  } finally {
    $gatewayRestarting.set(false)
    // Same restart follow-through as the requested flow above: probe-first
    // recovery, never the manual path's unconditional teardown.
    void reconnectGateway({ source: 'restart-followthrough' }).catch(() => undefined)
  }
}
