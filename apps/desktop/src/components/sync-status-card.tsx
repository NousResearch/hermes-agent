/**
 * SyncStatusCard — the desktop warning surface for the latest pm/plugin sync
 * receipt (SPEC-05). The renderer calls `getSyncStatus` here and paints the
 * derived summary: failed venv rebuilds, dependency-bisect disables,
 * needs-fixing update_url mismatches, and available plugin updates. The
 * surface degrades to nothing when there is no receipt (older backends, no
 * sync yet) rather than rendering an empty box.
 *
 * The copy lives in `deriveSyncStatusSummary` (the extracted, unit-tested
 * owner) — this component is presentation only, so the wording cannot drift
 * between the derivation and the paint.
 */

import { useEffect, useState } from 'react'

import { AlertCircle, AlertTriangle, Check } from '@/lib/icons'

import { deriveSyncStatusSummary, type SyncStatusSummary } from './sync-status'

const LEVEL_STYLES: Record<SyncStatusSummary['level'], string> = {
  ok: 'text-muted-foreground',
  info: 'text-muted-foreground',
  warn: 'text-amber-600 dark:text-amber-400',
  error: 'text-destructive'
}

export function SyncStatusCard() {
  const [summary, setSummary] = useState<SyncStatusSummary | null>(null)

  useEffect(() => {
    let alive = true

    void window.hermesDesktop
      ?.getSyncStatus?.()
      .then(receipt => {
        if (alive) {
          setSummary(deriveSyncStatusSummary(receipt))
        }
      })
      .catch(() => {
        // A missing/broken receipt surface must never break the updates
        // overlay — degrade to nothing (the receipt is advisory).
        if (alive) {
          setSummary(null)
        }
      })

    return () => {
      alive = false
    }
  }, [])

  if (!summary?.headline) {
    return null
  }

  const tone = LEVEL_STYLES[summary.level]

  return (
    <div
      className="grid gap-2 rounded-lg border border-border bg-muted/40 p-3 text-left"
      data-testid="sync-status-card"
    >
      <p className={`flex items-center gap-2 text-sm font-medium ${tone}`}>
        {summary.level === 'error' || summary.level === 'warn' ? (
          <AlertCircle aria-hidden className="size-4 shrink-0" />
        ) : (
          <Check aria-hidden className="size-4 shrink-0" />
        )}
        {summary.headline}
      </p>

      {summary.needsFixing.length > 0 && (
        <ul className="grid gap-1 text-xs text-foreground">
          {summary.needsFixing.map(item => (
            <li className="flex items-start gap-2" key={`needs-fixing-${item.plugin}`}>
              <AlertTriangle aria-hidden className="mt-0.5 size-3 shrink-0 text-amber-600 dark:text-amber-400" />
              <span className="leading-snug">
                {item.plugin}: {item.reason}
              </span>
            </li>
          ))}
        </ul>
      )}

      {summary.disabledPlugins.length > 0 && (
        <ul className="grid gap-1 text-xs text-foreground">
          {summary.disabledPlugins.map(item => (
            <li className="flex items-start gap-2" key={`disabled-${item.plugin}`}>
              <AlertTriangle aria-hidden className="mt-0.5 size-3 shrink-0 text-amber-600 dark:text-amber-400" />
              <span className="leading-snug">
                {item.plugin}: {item.reason}
              </span>
            </li>
          ))}
        </ul>
      )}

      {summary.updatesAvailable.length > 0 && (
        <ul className="grid gap-1 text-xs text-foreground">
          {summary.updatesAvailable.map(item => (
            <li className="leading-snug" key={`update-${item.name}`}>
              {item.name}
              {item.current && item.latest ? `: ${item.current} → ${item.latest}` : ''}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
