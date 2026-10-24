import { useStore } from '@nanostores/react'
import { type ReactElement, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { previewSessionPrune, type SessionPrunePreview } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, Trash2 } from '@/lib/icons'
import { confirm } from '@/store/confirm'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { CONTROL_TEXT } from './constants'

// Inspired by Claude Cowork's Settings > Storage, which frees space by deleting inactive
// conversations (claude.com/docs/cowork/changelog, v3.1623.0). Hermes keeps history in the
// profile's state.db, so "inactive" means what `hermes sessions prune --older-than` means:
// ended sessions with no activity in N days.
const DAY_CHOICES = [30, 90, 180, 365] as const
const DEFAULT_DAYS = 90
// The owner deletes in one shielded transaction; a cron-heavy store can take minutes.
const PRUNE_TIMEOUT_MS = 600_000

function formatDay(epochSeconds: null | number): string {
  return epochSeconds ? new Date(epochSeconds * 1000).toLocaleDateString() : ''
}

/** "Free up space" card under the Storage breakdown: preview, confirm, then the owner prunes. */
export function StorageCleanup({ onFreed }: { onFreed: () => void }): ReactElement {
  const { t } = useI18n()
  const s = t.settings.storageSection
  const profile = useStore($settingsScopeProfile)
  const connectionId = useStore($activeConnectionId)
  const [days, setDays] = useState<number>(DEFAULT_DAYS)
  const [preview, setPreview] = useState<null | SessionPrunePreview>(null)
  const [previewFailed, setPreviewFailed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [generation, setGeneration] = useState(0)

  useEffect(() => {
    let alive = true

    setPreview(null)
    setPreviewFailed(false)
    previewSessionPrune(days, { connectionId: connectionId ?? undefined, profile })
      .then(result => alive && setPreview(result))
      .catch(() => alive && setPreviewFailed(true))

    return () => void (alive = false)
  }, [connectionId, days, generation, profile])

  const remove = useCallback(async () => {
    if (!preview?.matched) {
      return
    }

    const ok = await confirm({
      confirmLabel: s.cleanupConfirmAction(preview.matched),
      description: s.cleanupConfirmBody(days),
      destructive: true,
      details: [
        { label: s.cleanupProfile, value: profile },
        {
          label: s.cleanupLastUsed,
          value: [formatDay(preview.oldest_last_active), formatDay(preview.newest_last_active)]
            .filter(Boolean)
            .join(' – ')
        }
      ].filter(row => row.value),
      title: s.cleanupConfirmTitle(preview.matched)
    })

    if (!ok) {
      return
    }

    setBusy(true)

    try {
      const result = await requestGatewayForAgent<{ deleted: number }>(
        connectionId,
        profile,
        'session.prune',
        { filters: { older_than_days: days, archived: false } },
        PRUNE_TIMEOUT_MS
      )

      notify({ durationMs: 6_000, kind: 'success', message: s.cleanupDone(result?.deleted ?? 0) })
      onFreed()
    } catch (err) {
      notifyError(err, s.cleanupFailed)
    } finally {
      setBusy(false)
      setGeneration(n => n + 1)
    }
  }, [connectionId, days, onFreed, preview, profile, s])

  const matched = preview?.matched ?? 0

  return (
    <div className="mt-6 flex flex-col gap-2 rounded-xl border border-border/60 bg-background/40 px-4 py-3">
      <span className="text-sm font-medium">{s.cleanupTitle}</span>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0 text-sm">
          <div>{s.cleanupSessions}</div>
          <div className="text-xs text-muted-foreground">
            {previewFailed
              ? s.cleanupPreviewFailed
              : preview === null
                ? s.cleanupCounting
                : matched === 0
                  ? s.cleanupNothing(days)
                  : s.cleanupMatched(matched, days)}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Select onValueChange={value => setDays(Number(value))} value={String(days)}>
            <SelectTrigger aria-label={s.cleanupAge} className={CONTROL_TEXT}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DAY_CHOICES.map(choice => (
                <SelectItem key={choice} value={String(choice)}>
                  {s.cleanupDays(choice)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button disabled={busy || matched === 0} onClick={() => void remove()} size="sm" variant="outline">
            {busy ? <Loader2 className="animate-spin" /> : <Trash2 />}
            {s.cleanupAction}
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{s.cleanupFootnote}</p>
    </div>
  )
}
