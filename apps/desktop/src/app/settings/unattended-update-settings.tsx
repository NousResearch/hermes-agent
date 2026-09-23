import { useEffect, useState } from 'react'

import { Switch } from '@/components/ui/switch'
import type { DesktopUnattendedTaskState } from '@/global'
import { useI18n } from '@/i18n'
import { CheckCircle2, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { getUnattendedSchedule, setUnattendedSchedule } from '@/store/updates'

const HOURS = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'))
const MINUTES = Array.from({ length: 60 }, (_, i) => String(i).padStart(2, '0'))

function formatClock(hour: number, minute: number): string {
  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`
}

/**
 * Scheduled (unattended) Windows self-update — LOCAL opt-in only.
 *
 * This surface exists solely so the user of THIS machine can opt in: the
 * schedule is read/written exclusively through native IPC to the main process,
 * persisted only in the local userData `updates.json`, and defaults to OFF.
 * Remote/backend agents see nothing here and cannot toggle it.
 *
 * On the Windows desktop build the SAME toggle also create/removes ONE
 * per-user, least-privilege Windows Task Scheduler task (electron/main.ts
 * reconcile → electron/scheduled-task.ts): enabling the schedule registers the
 * exact named task so updates can also run at the chosen local time while the
 * app is CLOSED; disabling uninstalls it. Its live state is surfaced below.
 */
export function UnattendedUpdateSettings() {
  const { t } = useI18n()
  const a = t.settings.about

  const [enabled, setEnabled] = useState(false)
  const [hour, setHour] = useState(2)
  const [minute, setMinute] = useState(0)
  const [task, setTask] = useState<DesktopUnattendedTaskState | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    let cancelled = false

    void getUnattendedSchedule()
      .then(state => {
        if (cancelled) {
          return
        }

        setEnabled(state.schedule.enabled)
        setHour(state.schedule.hour)
        setMinute(state.schedule.minute)
        setTask(state.task)
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
      const state = await setUnattendedSchedule(next)
      setEnabled(state.schedule.enabled)
      setHour(state.schedule.hour)
      setMinute(state.schedule.minute)
      setTask(state.task)
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2500)
    } catch {
      // Main-process IPC failure: keep the last-known values, no toast spam.
    } finally {
      setSaving(false)
    }
  }

  const taskLine = (() => {
    if (!task) {
      return null
    }

    switch (task.kind) {
      case 'installed':

      case 'stale':
        return {
          tone: 'ok',
          text: a.unattendedTaskInstalled(formatClock(hour, minute))
        }

      case 'absent':
        return { tone: 'muted', text: a.unattendedTaskAbsent }

      case 'foreign':
        return { tone: 'warn', text: a.unattendedTaskForeign }

      case 'error':
        return { tone: 'warn', text: task.message || a.unattendedTaskError }

      case 'unsupported':
        return null
    }
  })()

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
          {taskLine && (
            <p
              className={cn(
                'mt-1 w-full text-xs',
                taskLine.tone === 'ok' && 'text-emerald-600 dark:text-emerald-400',
                taskLine.tone === 'muted' && 'text-muted-foreground',
                taskLine.tone === 'warn' && 'text-amber-600 dark:text-amber-400'
              )}
            >
              {taskLine.text}
            </p>
          )}
        </div>
      )}
    </div>
  )
}