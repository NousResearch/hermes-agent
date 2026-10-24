import { atom } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { getActionStatus, getStatus, restartGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { sharedGatewayProfiles } from '@/lib/shared-gateway-restart'
import { confirm } from '@/store/confirm'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { notify, notifyError } from '@/store/notifications'
import type { ActionResponse } from '@/types/hermes'

const POLL_ATTEMPTS = 18
const POLL_INTERVAL_MS = 1200
const POLL_LINES = 200
const GATEWAY_RESTART_ACTION = 'gateway-restart'

// True while a gateway restart is in flight — drives the statusbar gateway
// indicator (glyph spinner) so the restart shows up where users already look,
// instead of a toast that vanishes or a generic "Agents running" counter.
export const $gatewayRestarting = atom(false)
let activeGatewayRestarts = 0

function beginGatewayRestart() {
  activeGatewayRestarts += 1
  $gatewayRestarting.set(true)
}

function endGatewayRestart() {
  activeGatewayRestarts = Math.max(0, activeGatewayRestarts - 1)
  $gatewayRestarting.set(activeGatewayRestarts > 0)
}

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
async function awaitAction(name: string, scope?: ProfileScope, isCurrent?: () => boolean): Promise<boolean> {
  let sawAnsweredPoll = false
  let lastPollError: unknown = null

  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS))

    if (isCurrent?.() === false) {
      return false
    }

    let status: Awaited<ReturnType<typeof getActionStatus>>

    try {
      status = await getActionStatus(name, POLL_LINES, scope)
    } catch (err) {
      lastPollError = err

      continue
    }

    if (isCurrent?.() === false) {
      return false
    }

    sawAnsweredPoll = true
    lastPollError = null

    if (!status.running && status.exit_code == null) {
      // A fresh process that never saw this action id answers exactly this way.
      continue
    }

    if (!status.running) {
      if (status.exit_code != null && status.exit_code !== 0) {
        throw new Error(translateNow('commandCenter.gatewayRestartFailed'))
      }

      return true
    }
  }

  if (!sawAnsweredPoll) {
    throw lastPollError ?? new Error(translateNow('commandCenter.gatewayRestartFailed'))
  }

  return isCurrent?.() !== false
}

// Under `gateway.multiplex_profiles` the profile in view has no gateway of its
// own: "Restart gateway" restarts the ONE shared multiplexer and every bot on
// this device blips. Ask first, naming them; standalone gateways (and older
// backends that do not report `gateway_shared_with`) keep the silent restart.
export async function confirmSharedGatewayRestart(
  scope?: ProfileScope,
  isCurrent?: () => boolean
): Promise<false | null | string[]> {
  let shared: null | string[] = null

  try {
    shared = sharedGatewayProfiles(await getStatus(scope))
  } catch {
    return isCurrent?.() === false ? false : null
  }

  if (isCurrent?.() === false) {
    return false
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

  return ok && isCurrent?.() !== false ? shared : false
}

// Restart the messaging gateway, surfacing progress in the statusbar gateway
// indicator. Self-contained and never rejects, so every trigger — Cmd+K, the
// messaging save/toggle toasts — gets identical feedback from a plain
// `void runGatewayRestart()`, and a failure is the only thing that toasts.
// Resolves `true` when the restart child completed cleanly (callers that keep
// a "restart needed" banner clear it on that signal only). Pass `scope` to pin
// the restart request and status polls to one backend owner; the ambient scope
// serves callers without an owner (#71352).
export async function runGatewayRestart(scope?: ProfileScope, isCurrent?: () => boolean): Promise<boolean> {
  if (isCurrent?.() === false) {
    return false
  }

  const shared = await confirmSharedGatewayRestart(scope, isCurrent)

  if (shared === false || isCurrent?.() === false) {
    return false
  }

  beginGatewayRestart()

  try {
    const started: ActionResponse = await restartGateway(scope)

    if (!(await awaitAction(started.name, scope, isCurrent))) {
      return false
    }

    if (shared && isCurrent?.() !== false) {
      notify({ kind: 'success', message: translateNow('commandCenter.sharedGatewayRestarted', shared.length) })
    }

    return isCurrent?.() !== false
  } catch (err) {
    if (isCurrent?.() !== false) {
      notifyError(err, translateNow('commandCenter.gatewayRestartFailed'))
    }

    return false
  } finally {
    endGatewayRestart()

    if (isCurrent?.() !== false) {
      void reconnectGateway({ source: 'restart-followthrough' }).catch(() => undefined)
    }
  }
}

// Watch a restart the BACKEND spawned instead of one this app requested.
export async function watchGatewayRestartOutcome(
  scope?: ProfileScope,
  isCurrent?: () => boolean
): Promise<boolean> {
  beginGatewayRestart()

  try {
    return await awaitAction(GATEWAY_RESTART_ACTION, scope, isCurrent)
  } catch {
    return false
  } finally {
    endGatewayRestart()

    if (isCurrent?.() !== false) {
      void reconnectGateway({ source: 'restart-followthrough' }).catch(() => undefined)
    }
  }
}
