import { atom } from 'nanostores'

import type { ProfileScope } from '@/api/client'
import { getActionStatus, getStatus, restartGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { sharedGatewayProfiles } from '@/lib/shared-gateway-restart'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import type { ActionResponse } from '@/types/hermes'

const POLL_ATTEMPTS = 18
const POLL_INTERVAL_MS = 1200
const POLL_TIMEOUT_S = 180
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
async function awaitAction(name: string, scope?: ProfileScope, isCurrent?: () => boolean): Promise<boolean> {
  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS))

    if (isCurrent?.() === false) {
      return false
    }

    const status = await getActionStatus(name, POLL_TIMEOUT_S, scope)

    if (isCurrent?.() === false) {
      return false
    }

    if (!status.running) {
      if (status.exit_code != null && status.exit_code !== 0) {
        throw new Error(translateNow('commandCenter.gatewayRestartFailed'))
      }

      return true
    }
  }

  return isCurrent?.() !== false
}

// Under `gateway.multiplex_profiles` the profile in view has no gateway of its
// own: "Restart gateway" restarts the ONE shared multiplexer and every bot on
// this device blips. Ask first, naming them; standalone gateways (and older
// backends that do not report `gateway_shared_with`) keep the silent restart.
// Resolves the served list when the user confirmed, `null` when nothing is
// shared, `false` when they cancelled.
export async function confirmSharedGatewayRestart(
  scope?: ProfileScope,
  isCurrent?: () => boolean
): Promise<false | null | string[]> {
  let shared: null | string[] = null

  try {
    shared = sharedGatewayProfiles(await getStatus(scope))
  } catch {
    // Status unavailable: fall back to the plain restart rather than blocking it.
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
// a "restart needed" banner clear it on that signal only).
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
  }
}

// Watch a restart the BACKEND spawned (e.g. after Telegram QR onboarding writes
// credentials) instead of one this app requested. Same indicator, same bounded
// poll; resolves `false` on a non-zero exit so the caller can re-arm its banner.
export async function watchGatewayRestartOutcome(scope?: ProfileScope): Promise<boolean> {
  beginGatewayRestart()

  try {
    await awaitAction(GATEWAY_RESTART_ACTION, scope)

    return true
  } catch {
    return false
  } finally {
    endGatewayRestart()
  }
}
