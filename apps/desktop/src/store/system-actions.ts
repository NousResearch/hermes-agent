import { atom } from 'nanostores'

import { getActionStatus, restartGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { notifyError } from '@/store/notifications'
import type { ActionResponse } from '@/types/hermes'

const POLL_ATTEMPTS = 18
const POLL_INTERVAL_MS = 1200
const POLL_TIMEOUT_S = 180
const GATEWAY_RESTART_ACTION = 'gateway-restart'

// True while a gateway restart is in flight — drives the statusbar gateway
// indicator (glyph spinner) so the restart shows up where users already look,
// instead of a toast that vanishes or a generic "Agents running" counter.
export const $gatewayRestarting = atom(false)

// Poll a backend action to completion (or a bounded window), throwing on a
// non-zero exit so the caller can surface the failure. In no-service installs
// the child becomes the foreground gateway and never exits, so "still running
// when the window closes" counts as success.
async function awaitAction(name: string): Promise<void> {
  for (let attempt = 0; attempt < POLL_ATTEMPTS; attempt += 1) {
    await new Promise(resolve => window.setTimeout(resolve, POLL_INTERVAL_MS))
    const status = await getActionStatus(name, POLL_TIMEOUT_S)

    if (!status.running) {
      if (status.exit_code != null && status.exit_code !== 0) {
        throw new Error(translateNow('commandCenter.gatewayRestartFailed'))
      }

      return
    }
  }
}

// Restart the messaging gateway, surfacing progress in the statusbar gateway
// indicator. Self-contained and never rejects, so every trigger — Cmd+K, the
// messaging save/toggle toasts — gets identical feedback from a plain
// `void runGatewayRestart()`, and a failure is the only thing that toasts.
// Resolves `true` when the restart child completed cleanly (callers that keep
// a "restart needed" banner clear it on that signal only).
export async function runGatewayRestart(): Promise<boolean> {
  $gatewayRestarting.set(true)

  try {
    const started: ActionResponse = await restartGateway()
    await awaitAction(started.name)

    return true
  } catch (err) {
    notifyError(err, translateNow('commandCenter.gatewayRestartFailed'))

    return false
  } finally {
    $gatewayRestarting.set(false)
  }
}

// Watch a restart the BACKEND spawned (e.g. after Telegram QR onboarding writes
// credentials) instead of one this app requested. Same indicator, same bounded
// poll; resolves `false` on a non-zero exit so the caller can re-arm its banner.
export async function watchGatewayRestartOutcome(): Promise<boolean> {
  $gatewayRestarting.set(true)

  try {
    await awaitAction(GATEWAY_RESTART_ACTION)

    return true
  } catch {
    return false
  } finally {
    $gatewayRestarting.set(false)
  }
}
