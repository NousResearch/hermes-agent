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

/** Tail progress published by a CLI-driven update (`update_stage.json`). */
export interface UpdateTailStage {
  message: string
  /** ISO timestamp of the heartbeat; the reader derives staleness. */
  at?: string | null
}

/** True for CLI update receipts (`hermes update`); pm receipts never carry these. */
export function isUpdateReceipt(receipt: DesktopSyncReceipt): boolean {
  return (
    receipt.pre_update != null || receipt.post_update != null || receipt.stop_reason != null
  )
}

function relativeTime(iso: string | null | undefined): string {
  if (!iso) {
    return ''
  }

  const then = Date.parse(iso)

  if (Number.isNaN(then)) {
    return ''
  }

  const seconds = Math.max(0, Math.floor((Date.now() - then) / 1000))

  if (seconds < 60) {
    return 'just now'
  }

  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ago`
  }

  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}h ago`
  }

  return `${Math.floor(seconds / 86400)}d ago`
}

function truncateDetail(detail: string, max = 160): string {
  const flat = detail.replace(/\s+/g, ' ').trim()

  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

/** Mirrors the CLI's `TAIL_STAGE_STALE_SECONDS`: a tail silent longer than
 *  this reads as wedged in the overlay. */
const TAIL_STAGE_STALE_MS = 5 * 60 * 1000

function summarizeUpdateReceipt(
  receipt: DesktopSyncReceipt,
  tail: UpdateTailStage | null | undefined
): { headline: string; level: SyncStatusSummary['level'] } {
  const when = relativeTime(receipt.finished_at ?? receipt.started_at)
  const whenSuffix = when ? ` · ${when}` : ''

  if (receipt.outcome === 'running') {
    const lastStep = [...(receipt.steps ?? [])].reverse().find(step => step.name)?.name

    if (tail?.message) {
      // Fresh heartbeats read as progress; a heartbeat older than minutes
      // reads as wedged — the CLI side publishes every tail phase, so silence
      // is signal, not absence of data. (`--status` owns the death verdict via
      // pid liveness; the card only reports what it can see.)
      const tailMs = tail.at ? Date.parse(tail.at) : Number.NaN
      const tailAgeMs = Number.isNaN(tailMs) ? null : Math.max(0, Date.now() - tailMs)
      const stale = tailAgeMs === null || tailAgeMs > TAIL_STAGE_STALE_MS

      if (stale) {
        const age = tailAgeMs === null ? 'a while' : relativeTime(tail.at ?? undefined) || 'a while'

        return {
          headline: `Update in progress — no progress for ${age} (last: ${truncateDetail(tail.message)})`,
          level: 'warn'
        }
      }

      const age = relativeTime(tail.at ?? undefined)

      return {
        headline: `Update in progress — ${truncateDetail(tail.message)}${age && age !== 'just now' ? ` (${age})` : ''}`,
        level: 'info'
      }
    }

    return {
      headline: `Update in progress${lastStep ? ` — ${truncateDetail(lastStep)}` : ''}${whenSuffix}`,
      level: 'info'
    }
  }

  if (receipt.outcome === 'interrupted') {
    return {
      headline: `Last update was interrupted${whenSuffix} — rerun \`hermes update\``,
      level: 'warn'
    }
  }

  if (receipt.outcome === 'partial') {
    return {
      headline: `Last update finished${whenSuffix} with follow-ups owed — see the receipt`,
      level: 'info'
    }
  }

  if (receipt.outcome === 'failed' || receipt.outcome === 'refused') {
    const failedStep = [...(receipt.steps ?? [])].reverse().find(step => step.ok === false)

    const reason = truncateDetail(
      receipt.stop_reason ?? failedStep?.detail ?? failedStep?.name ?? 'see the update receipt'
    )

    return { headline: `Last update failed${whenSuffix} — ${reason}`, level: 'error' }
  }

  if (receipt.outcome === 'success') {
    return { headline: `Last update succeeded${whenSuffix}`, level: 'ok' }
  }

  return { headline: `Last update ended (${receipt.outcome ?? 'unknown outcome'})${whenSuffix}`, level: 'info' }
}

export function deriveSyncStatusSummary(
  receipt: DesktopSyncReceipt | null,
  tail?: UpdateTailStage | null
): SyncStatusSummary {
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

  // CLI update receipts speak a different outcome vocabulary (`success` /
  // `failed` / `running` + `stop_reason`) than pm receipts, and their failed
  // steps all read ok — the pm ladder below would mislabel them as a
  // dependency-rebuild failure. Summarize them on their own terms first.
  if (isUpdateReceipt(receipt)) {
    const { headline, level } = summarizeUpdateReceipt(receipt, tail)

    return { headline, level, disabledPlugins, needsFixing, updatesAvailable }
  }

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
