import { useEffect, useState } from 'react'

import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { CheckCircle2, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { getUnattendedSchedule, setUnattendedSchedule } from '@/store/updates'

const HOURS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'))
const MINUTES = Array.from({ length: 60 }, (_, i) => String(i).padStart(2, '0'))

/**
 * Scheduled (unattended) Windows self-update — LOCAL opt-in only.
 *
 * This surface exists solely so the user of THIS machine can opt in: the
 * schedule is read/written exclusively through native IPC to the main process,
 * persisted only in the local userData `updates.json`, and defaults to OFF.
 * Remote/backend agents see nothing here and cannot toggle it.
 */
export function UnattendedUpdateSettings() {
  const { t } = useI18n()
  const a = t.settings.about

  const [enabled, setEnabled] = useState(false)
  const [hour, setHour] = useState(3)
  const [minute, setMinute] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    let cancelled = false

    void getUnattendedSchedule()
      .then(schedule => {
        if (cancelled) {
          return
        }

        setEnabled(schedule.enabled)
        setHour(schedule.hour)
        setMinute(schedule.minute)
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) {
          setLoaded(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const persist = async (next: { enabled: boolean; hour: number; minute: number }) => {
    setSaving(true)

    try {
      const confirmed = await setUnattendedSchedule(next)
      setEnabled(confirmed.enabled)
      setHour(confirmed.hour)
      setMinute(confirmed.minute)
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2500)
    } catch {
      // Main-process IPC failure: keep the last-known values, no toast spam.
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-xl border border-border/70 bg-muted/20 px-4 py-3 text-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="font-medium">{a.unattendedTitle}</p>
          <p className="mt-1 text-xs text-muted-foreground">{a.unattendedDesc}</p>
          <p className="mt-1 text-xs text-muted-foreground">{a.unattendedWindowsOnly}</p>
        </div>
        <Switch
          aria-label={a.unattendedEnabled}
          checked={enabled}
          disabled={!loaded || saving}
          onCheckedChange={value => void persist({ enabled: value, hour, minute })}
        />
      </div>

      {enabled && (
        <div className="mt-3 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {a.unattendedHourLabel}
            <select
              aria-label={a.unattendedHourLabel}
              className={cn(
                'rounded-md border border-border/80 bg-background px-2 py-1 text-sm text-foreground',
                'focus-visible:border-ring focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50 focus-visible:outline-none'
              )}
              disabled={saving}
              onChange={event => void persist({ enabled: true, hour: Number(event.target.value), minute })}
              value={hour}
            >
              {HOURS.map(h => (
                <option key={h} value={Number(h)}>
                  {h}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {a.unattendedMinuteLabel}
            <select
              aria-label={a.unattendedMinuteLabel}
              className={cn(
                'rounded-md border border-border/80 bg-background px-2 py-1 text-sm text-foreground',
                'focus-visible:border-ring focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50 focus-visible:outline-none'
              )}
              disabled={saving}
              onChange={event => void persist({ enabled: true, hour, minute: Number(event.target.value) })}
              value={minute}
            >
              {MINUTES.map(m => (
                <option key={m} value={Number(m)}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          {(saving || saved) && (
            <div className="ml-auto flex items-center gap-2">
              {saving ? (
                <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
              ) : (
                <span className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="size-3.5" />
                  {a.unattendedSaved}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}