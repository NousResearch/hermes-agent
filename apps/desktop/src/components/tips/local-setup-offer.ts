/**
 * The local-setup campaign: one bubble on the model pill for machines that
 * could run local models and haven't set them up.
 *
 * Not a rotation tip — a campaign the rotation CONSULTS first at each quiet
 * due moment (use-tip-rotation.ts): conditional (most machines qualify or
 * don't, permanently), actionable (it carries the one button a tip may
 * have), and perishable (setting up local models — or the ✕ — ends it).
 * A live "your GPU can run this, free and private" outranks the walk's
 * "the model name is a button" whenever both are true, and an ignored
 * bubble may return in a week rather than walking on forever.
 *
 * Eligibility is fetched, not assumed: the backend's own fit check (the
 * same catalog `fits` the Local Models pane prices its hero with) decides
 * whether this machine qualifies. Reads are lazy — nothing polls for a
 * bubble. The first quiet due moment kicks one status+catalog read and
 * holds the turn (no walk tip may spend the cooldown ahead of a pending
 * campaign); the cached answer serves every later one. Completing
 * setup flips the next read to ineligible, so the campaign retires itself
 * without bookkeeping — and the cache dies with a connection change,
 * because eligibility is a fact about the backend's machine.
 */

import { getLocalCatalog, getLocalModelsStatus } from '@/hermes'
import type { Translations } from '@/i18n/types'
import { LOCAL_SETUP_TIP_ID, localSetupDue, localSetupEligible } from '@/lib/tips/local-cta'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $connection } from '@/store/session'
import { $retiredTips, $tipShownAt, dismissTip, showTip } from '@/store/tips'

/** The pill the bubble points at — the same handle the rotation's
 *  model-switch tip uses, so the two can never drift to different anchors. */
const MODEL_PILL_TARGETS = ['[data-tour="model-pill"]'] as const

let eligibilityCache: { eligible: boolean } | null = null
let eligibilityInFlight = false
let boundToConnection = false

/** Reset the session cache — tests only. */
export function resetLocalSetupOfferCache(): void {
  eligibilityCache = null
  eligibilityInFlight = false
}

/**
 * Offer the campaign the current quiet moment. True = it put its bubble up
 * and the moment is spent; false = the rotation's walk may have it.
 */
export function offerLocalSetupTip(copy: Translations['tips'], openLocalModels: () => void): boolean {
  // Local models ship behind the --local launch flag; without it there is
  // no Local Models pane for the button to open, so the campaign never runs
  // (and never spends a status/catalog read).
  if (!$localModelsEnabled.get()) {
    return false
  }

  if ($retiredTips.get().includes(LOCAL_SETUP_TIP_ID)) {
    return false
  }

  if (!localSetupDue(Date.now(), $tipShownAt.get()[LOCAL_SETUP_TIP_ID])) {
    return false
  }

  // Local backends only: on a remote connection (cloud resolves to remote)
  // the models would run on the far machine, and "stays on your computer"
  // would be promising someone else's computer. Checked before the cache so
  // a re-home mid-session can't serve a stale yes.
  if (($connection.get()?.mode ?? null) !== 'local') {
    return false
  }

  if (!boundToConnection) {
    boundToConnection = true
    $connection.listen(() => resetLocalSetupOfferCache())
  }

  if (!eligibilityCache) {
    if (!eligibilityInFlight) {
      eligibilityInFlight = true

      void Promise.all([getLocalModelsStatus(), getLocalCatalog()])
        .then(([status, catalog]) => {
          eligibilityCache = {
            eligible: localSetupEligible($connection.get()?.mode ?? null, status, catalog.models)
          }
        })
        .catch(() => {
          // No backend answer, no campaign this session. The next launch —
          // or the next connection — asks again.
          eligibilityCache = { eligible: false }
        })
        .finally(() => {
          eligibilityInFlight = false
        })
    }

    // Hold the moment while the read flies: nothing shows and no cooldown
    // arms, so the next tick answers from the cache. Handing this moment to
    // the rotation instead would put a walk tip up first and park the
    // campaign behind the six-hour cooldown — the exact inversion of the
    // priority. Costs an ineligible machine one 30s tick, once per session.
    return true
  }

  if (!eligibilityCache.eligible) {
    return false
  }

  showTip({
    action: {
      label: copy.items['local-setup'].action,
      onSelect: () => {
        dismissTip()
        openLocalModels()
      }
    },
    side: 'top',
    targets: MODEL_PILL_TARGETS,
    text: copy.items['local-setup'].text,
    tipId: LOCAL_SETUP_TIP_ID,
    title: copy.items['local-setup'].title
  })

  return true
}
