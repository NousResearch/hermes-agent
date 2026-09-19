import { getLocalModelsJobs, getLocalModelsStatus } from '@/hermes'
import type { Translations } from '@/i18n/types'
import { localSetupDue } from '@/lib/tips/local-cta'
import { $activeGatewayRoute } from '@/store/gateway'
import { $localModelsEnabled } from '@/store/local-models-flag'
import {
  $localRuntimeInstallStarting,
  $localRuntimeJobs,
  localRuntimeInstallBusy,
  startLocalRuntimeInstall
} from '@/store/local-runtime-jobs'
import { $connection } from '@/store/session'
import { $activeTip, $retiredTips, $tipsEnabled, $tipShownAt, dismissTip, retireActiveTip, showTip } from '@/store/tips'
import type { LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

let snapshot: { readAt: number; status: LocalModelsStatus | null; jobs: LocalRuntimeJob[] } | null = null
let pending = false
let generation = 0

const closeUpdateTip = () => {
  const active = $activeTip.get()
  if (active?.tipId?.startsWith('local-runtime-update:')) {
    retireActiveTip()
  }
}

const reset = () => {
  generation++
  snapshot = null
  pending = false
  closeUpdateTip()
}

$connection.listen(reset)
$activeGatewayRoute.listen(reset)
$tipsEnabled.listen(enabled => {
  if (!enabled) {
    reset()
  }
})
$localRuntimeJobs.listen(() => {
  if (localRuntimeInstallBusy()) {
    reset()
  }
})
$localRuntimeInstallStarting.listen(starting => {
  if (starting) {
    reset()
  }
})

export function offerLocalRuntimeUpdateTip(copy: Translations['tips'], openLocalModels: () => void): boolean {
  if (
    !$localModelsEnabled.get() ||
    !$tipsEnabled.get() ||
    $connection.get()?.mode !== 'local' ||
    localRuntimeInstallBusy()
  ) {
    return false
  }

  const maxAge = snapshot?.status ? 60_000 : 5_000

  if (snapshot && Date.now() - snapshot.readAt > maxAge) {
    snapshot = null
  }

  if (!snapshot) {
    if (!pending) {
      pending = true
      const request = generation
      void Promise.all([getLocalModelsStatus(), getLocalModelsJobs()])
        .then(([status, { jobs }]) => {
          if (request === generation) {
            snapshot = { status, jobs, readAt: Date.now() }
          }
        })
        .catch(() => {
          if (request === generation) {
            snapshot = { status: null, jobs: [], readAt: Date.now() }
          }
        })
        .finally(() => {
          if (request === generation) {
            pending = false
          }
        })
    }

    return true
  }

  // Reuse a fresh answer while the UI checks for a quiet moment each second.
  const { status, jobs } = snapshot

  if (jobs.some(job => job.status === 'running' && (job.kind === 'runtime-install' || job.kind === 'quickstart'))) {
    return false
  }

  if (!status || !status.enabled || !status.runtime_installed || !status.update_available) {
    return false
  }

  const tipId = `local-runtime-update:${status.configured_tag}`

  if ($retiredTips.get().includes(tipId) || !localSetupDue(Date.now(), $tipShownAt.get()[tipId])) {
    return false
  }

  const text = copy.items['local-runtime-update']
  const offeredIn = generation
  showTip({
    tipId,
    title: text.title,
    text: text.text,
    targets: ['[data-tour="model-pill"]'],
    side: 'top',
    action: {
      label: text.action,
      onSelect: () => {
        if (
          offeredIn !== generation ||
          !$tipsEnabled.get() ||
          !$localModelsEnabled.get() ||
          $connection.get()?.mode !== 'local' ||
          localRuntimeInstallBusy() ||
          $activeTip.get()?.tipId !== tipId
        ) {
          closeUpdateTip()

          return
        }

        // Mark the tip as retired for this engine version so it doesn't
        // reappear on every launch when status.update_available is sticky.
        // (The dismissTip() call below handles the current-session dismiss;
        // retireActiveTip() persists the suppression in $retiredTips so
        // offerLocalRuntimeUpdateTip returns false on next startup.)
        retireActiveTip()
        dismissTip()
        openLocalModels()
        void startLocalRuntimeInstall()
      }
    }
  })

  return true
}
