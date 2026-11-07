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
  /** #57295: what the last update did with the user's local source edits. */
  stash: StashOutcome | null
}

/**
 * What the last `hermes update` did with uncommitted local source changes
 * (#57295): the receipt's `local_changes_stash` step records the disposition
 * as `<parked|restored|discarded>: <stash sha> (<reason>)`. `parked` means the
 * changes are NOT in the working tree — they only exist in the named stash
 * entry — which is exactly the outcome the desktop used to swallow silently.
 */
export interface StashOutcome {
  disposition: 'parked' | 'restored' | 'discarded'
  /** The stash commit ref; restore with `git stash apply <ref>`. */
  ref: string
  /** Producer's parenthesized reason ('' when none) — keep raw for display. */
  reason: string
  headline: string
  detail: string
  level: 'ok' | 'warn'
}

const STASH_STEP_NAME = 'local_changes_stash'
const STASH_DETAIL_RE = /^(parked|restored|discarded):\s*([0-9a-f]{7,40})(?:\s+\((.*)\))?$/

/** Derive the stash disposition from a receipt, or null when the last update
 *  never touched local changes (no stash step). Copy lives here so the
 *  updates overlay card and the post-update toast cannot drift (#57295). */
export function deriveStashOutcome(receipt: DesktopSyncReceipt | null): StashOutcome | null {
  if (!receipt) {
    return null
  }

  const steps = receipt.pm_steps ?? receipt.steps ?? []
  const step = [...steps].reverse().find(entry => entry?.name === STASH_STEP_NAME)

  if (!step?.detail) {
    return null
  }

  const match = STASH_DETAIL_RE.exec(step.detail)

  if (!match) {
    return null
  }

  const disposition = match[1] as StashOutcome['disposition']
  const ref = match[2]
  const reason = match[3] ?? ''

  if (disposition === 'restored') {
    return {
      disposition,
      ref,
      reason,
      headline: 'Your local changes were restored on top of the update',
      detail: 'Review `git diff` if anything looks off.',
      level: 'ok'
    }
  }

  if (disposition === 'discarded') {
    return {
      disposition,
      ref,
      reason,
      headline: 'Your local changes were discarded by the update',
      detail: 'updates.non_interactive_local_changes is set to discard — the stash was dropped after the update.',
      level: 'warn'
    }
  }

  if (reason.includes('conflict')) {
    return {
      disposition,
      ref,
      reason,
      headline: 'Your local changes conflicted with the update and were not re-applied',
      detail: `They're preserved in git stash — run \`git stash apply ${ref}\` to resolve manually.`,
      level: 'warn'
    }
  }

  if (reason === '--keep-stash') {
    return {
      disposition,
      ref,
      reason,
      headline: 'Your local changes were stashed and not re-applied after the update',
      detail: `They're safe in git stash — restore them with \`git stash apply ${ref}\`.`,
      level: 'warn'
    }
  }

  return {
    disposition,
    ref,
    reason,
    headline: 'Your local changes were stashed during the update and left in git stash',
    detail: `Restore them with \`git stash apply ${ref}\`${reason ? ` (${reason})` : ''}.`,
    level: 'warn'
  }
}

export function deriveSyncStatusSummary(receipt: DesktopSyncReceipt | null): SyncStatusSummary {
  const empty: SyncStatusSummary = {
    headline: null,
    level: 'ok',
    disabledPlugins: [],
    needsFixing: [],
    updatesAvailable: [],
    stash: null
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

  // #57295: a parked stash outranks plugin status — the user's own edits are
  // sitting outside the working tree — but never outranks a failed rebuild
  // (the install itself may be broken).
  const stash = deriveStashOutcome(receipt)

  let headline: string | null = null
  let level: SyncStatusSummary['level'] = 'ok'

  if (rebuildFailed) {
    headline = failureDetail
      ? `Dependency rebuild failed — ${failureDetail}`
      : 'Dependency rebuild failed — inspect the update receipt'
    level = 'error'
  } else if (stash) {
    headline = stash.headline
    level = stash.level
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

  return { headline, level, disabledPlugins, needsFixing, updatesAvailable, stash }
}
