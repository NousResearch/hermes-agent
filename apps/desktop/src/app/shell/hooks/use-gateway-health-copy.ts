import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import type { GatewayHealthPillCopy } from '@/lib/gateway-health-pill'
import { requestGatewayForProfile } from '@/store/gateway'
import {
  $gatewayUpdateDrains,
  bindGatewayUpdateDrain,
  clearGatewayUpdateDrain,
  drainOwnerKey,
  gatewayUpdateDrainFor,
  refreshGatewayUpdateDrain,
  UPDATE_WAIT_MS,
  updateDrainTurns
} from '@/store/gateway-update-drain'

/** Re-read while draining, so "finishing N turns" counts down as turns end. */
const DRAIN_POLL_MS = 5_000

type Requester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/** "Updating to v0.23.1 · finishing 1 turn" while *profile*'s gateway drains for an update, else
 *  null. Owns the watch: asks that gateway on every open (a new owner answers null and clears it),
 *  re-reads while draining, and forgets a cause whose gateway stays away past UPDATE_WAIT_MS. The
 *  cause is stored under that gateway alone, so switching profile or source never shows another's. */
export function useGatewayUpdateLabel(gatewayState: string, request: Requester, profile: string): null | string {
  const { t } = useI18n()
  const key = drainOwnerKey(profile)
  const drain = gatewayUpdateDrainFor(useStore($gatewayUpdateDrains), key)
  const open = gatewayState === 'open'
  const draining = drain !== null

  useEffect(() => bindGatewayUpdateDrain(request), [request])

  useEffect(() => {
    if (!open) {
      if (!draining) {
        return
      }

      const timer = window.setTimeout(() => clearGatewayUpdateDrain(key), UPDATE_WAIT_MS)

      return () => window.clearTimeout(timer)
    }

    const describe = () => refreshGatewayUpdateDrain(key, () => requestGatewayForProfile(profile, 'runtime.describe'))
    void describe()

    if (!draining) {
      return
    }

    const timer = window.setInterval(() => void describe(), DRAIN_POLL_MS)

    return () => window.clearInterval(timer)
  }, [draining, key, open, profile])

  // Once the drained owner is gone its turns are done; only the target is still true.
  return drain ? t.gatewayUpdate.updating(drain.target ?? null, open ? updateDrainTurns(drain) : 0) : null
}

/** The status-bar gateway chip's copy; every connection detail names the update while one drains. */
export function useGatewayHealthCopy(
  copy: Translations['shell']['statusbar'],
  gatewayState: string,
  request: Requester,
  profile: string
): GatewayHealthPillCopy {
  const updating = useGatewayUpdateLabel(gatewayState, request, profile)

  return {
    backend: copy.backend,
    checking: updating ?? copy.gatewayChecking,
    connecting: updating ?? copy.gatewayConnecting,
    messagingDegraded: copy.messagingDegraded,
    messagingStopped: copy.messagingStopped,
    needsSetup: updating ?? copy.gatewayNeedsSetup,
    offline: updating ?? copy.gatewayOffline,
    ready: updating ?? copy.gatewayReady,
    restarting: copy.gatewayRestarting,
    unavailable: updating ?? copy.gatewayUnavailable
  }
}
