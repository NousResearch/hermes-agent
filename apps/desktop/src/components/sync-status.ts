/**
 * Derive display state from a pm receipt — the pure logic behind the
 * sync-status surface. Receipts are optional-typed (older/newer receipts
 * may lack sections), so every derive degrades gracefully to 'nothing
 * to show' rather than throwing at the UI boundary.
 */

import type { DesktopSyncReceipt } from '@/global'

/** One actionable line per surface: what needs attention + why. */
export interface SyncStatusSummary {
  /** null = nothing to show (healthy / no receipt / no sections). */
  headline: string | null
  level: 'ok' | 'info' | 'warn' | 'error'
  /** Bisect decisions — always worth listing when present. */
  disabledPlugins: Array<{ plugin: string; reason: string }>
  /** check-updates needs-fixing entries (update_url mismatches). */
  needsFixing: Array<{ plugin: string; reason: string }>
  /** Updates the last check saw (name -> current -> latest). */
  updatesAvailable: Array<{ name: string; current: string | null; latest: string | null }>
}

export function deriveSyncStatusSummary(receipt: DesktopSyncReceipt | null): SyncStatusSummary {
  const empty: SyncStatusSummary = {
    headline: null,
    level: 'ok',
    disabledPlugins: [],
    needsFixing: [],
    updatesAvailable: []
  }

  if (!receipt) {
    return empty
  }

  const disabledPlugins = (receipt.pm_plugin_bisect ?? receipt.plugin_bisect ?? [])
    .filter(d => d.action === 'disabled')
    .map(d => ({ plugin: d.plugin, reason: d.reason }))

  const checks = receipt.plugin_checks ?? []

  const needsFixing = checks
    .filter(c => c.needs_fixing)
    .map(c => ({ plugin: c.name, reason: c.needs_fixing as string }))

  const updatesAvailable = checks
    .filter(c => c.update_available === true)
    .map(c => ({ name: c.name, current: c.current ?? null, latest: c.latest ?? null }))

  const outcome = receipt.pm_sync_outcome ?? receipt.outcome
  const rebuild = receipt.pm_venv_rebuild ?? receipt.venv_rebuild
  const steps = receipt.pm_steps ?? receipt.steps ?? []
  const failureDetail = steps.find(step => step.ok === false)?.detail ?? rebuild?.reason

  const rebuildFailed =
    outcome === 'failed' ||
    outcome === 'refused' ||
    (outcome !== 'ok' && outcome !== 'success' && rebuild?.ok === false)

  let headline: string | null = null
  let level: SyncStatusSummary['level'] = 'ok'

  if (rebuildFailed) {
    headline = failureDetail
      ? `Dependency rebuild failed — ${failureDetail}`
      : 'Dependency rebuild failed — inspect the update receipt'
    level = 'error'
  } else if (needsFixing.length > 0) {
    headline = `${needsFixing.length} plugin${needsFixing.length === 1 ? '' : 's'} need update-url review`
    level = 'warn'
  } else if (disabledPlugins.length > 0) {
    headline = `${disabledPlugins.length} plugin${
      disabledPlugins.length === 1 ? '' : 's'
    } disabled by dependency conflicts`
    level = 'warn'
  } else if (updatesAvailable.length > 0) {
    headline = `Plugin updates available (${updatesAvailable.length})`
    level = 'info'
  }

  return { headline, level, disabledPlugins, needsFixing, updatesAvailable }
}
