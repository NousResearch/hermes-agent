import { useStore } from '@nanostores/react'
import { type FC, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Clock } from '@/lib/icons'
import { fmtClock, HOUR } from '@/lib/time'
import {
  $now,
  type ScheduledRetry,
  sessionScheduledRetry,
  setScheduledRetry
} from '@/store/scheduled-retry'

// ── "Retry in X hours" for the failed-turn error card ──────────────────────
// A 429 "usage limit reached" frees the quota hours later, usually while the
// user is away. Scheduling a retry parks the same reload the Retry button
// fires until then, so the turn finishes by itself. One schedule per session:
// the record rides the error card of the failed message until it fires (the
// scheduled-retry store's scheduler clears it) or is cancelled here.

/** Quick delays offered next to the custom time field. */
const PRESETS_MS = [1 * HOUR, 3 * HOUR, 6 * HOUR]

/** Parse "HH:mm" (or "HHmm") into today's or tomorrow's local time, or null. */
export function parseClockTime(input: string, now: number): null | number {
  const match = input.trim().match(/^(\d{1,2}):?(\d{2})$/)

  if (!match) {
    return null
  }

  const hours = Number(match[1])
  const minutes = Number(match[2])

  if (hours > 23 || minutes > 59) {
    return null
  }

  const target = new Date(now)
  target.setHours(hours, minutes, 0, 0)

  // A time already past today means the small hours of tomorrow — the usage
  // window the user is waiting out has not arrived yet at, say, "00:30".
  if (target.getTime() <= now) {
    target.setDate(target.getDate() + 1)
  }

  return target.getTime()
}

interface ScheduleRetryControls {
  onSchedule: (atMs: number) => void
}

/** The "Retry in…" dropdown: preset delays plus a free-form "at HH:mm". */
const ScheduleRetryMenu: FC<ScheduleRetryControls> = ({ onSchedule }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const [open, setOpen] = useState(false)
  const [customTime, setCustomTime] = useState('')
  const [invalid, setInvalid] = useState(false)
  const nowRef = useRef(Date.now())

  const submitCustom = () => {
    const at = parseClockTime(customTime, nowRef.current)

    if (at === null) {
      setInvalid(true)

      return
    }

    onSchedule(at)
    setOpen(false)
  }

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger
        className="aui-error-action"
        onClick={() => triggerHaptic('selection')}
        title={copy.errorRetryLaterMenu}
        type="button"
      >
        <Clock className="size-3" />
        {copy.errorRetryLater}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start">
        <DropdownMenuLabel>{copy.errorRetryLaterMenu}</DropdownMenuLabel>
        {PRESETS_MS.map((delay, index) => (
          <DropdownMenuItem
            key={delay}
            onSelect={() => onSchedule(nowRef.current + delay)}
          >
            {[copy.errorRetryIn1h, copy.errorRetryIn3h, copy.errorRetryIn6h][index]}
          </DropdownMenuItem>
        ))}
        <DropdownMenuItem
          onSelect={event => {
            // Keep the menu open for the time field — closing here would take
            // the input with it before the value could be submitted.
            event.preventDefault()
            setCustomTime('')
            setInvalid(false)
          }}
        >
          <form
            className="flex w-full items-center gap-1.5"
            data-testid="retry-at-form"
            onSubmit={event => {
              event.preventDefault()
              submitCustom()
            }}
          >
            <span className="shrink-0">{copy.errorRetryAt}</span>
            <input
              aria-invalid={invalid || undefined}
              className="w-14 rounded-sm border border-(--ui-stroke-secondary) bg-transparent px-1 py-0.5 text-xs focus:outline-none focus-visible:border-(--ui-stroke-strong)"
              data-testid="retry-at-input"
              onChange={event => {
                setCustomTime(event.target.value)
                setInvalid(false)
              }}
              placeholder="14:30"
              value={customTime}
            />
            <button
              className="ml-auto shrink-0 rounded-sm px-1.5 py-0.5 text-xs hover:bg-(--ui-control-active-background)"
              data-testid="retry-at-submit"
              type="submit"
            >
              ↵
            </button>
          </form>
          {invalid && <span className="text-[0.625rem] text-destructive">{copy.errorRetryAtInvalid}</span>}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** "Auto-retry at HH:mm · Cancel" once a retry is scheduled for this session. */
const ScheduledRetryNotice: FC<{ retry: ScheduledRetry }> = ({ retry }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const now = useStore($now)

  // Ticks so the label drops "auto-retry at 14:30" the moment it fires.
  useEffect(() => {
    if (now >= retry.at) {
      return
    }

    const id = window.setTimeout(() => $now.set(Date.now()), Math.max(0, retry.at - now))

    return () => window.clearTimeout(id)
  }, [now, retry.at])

  return (
    <span className="inline-flex items-center gap-1" data-testid="scheduled-retry-notice">
      <Clock className="size-3" />
      {copy.errorRetryScheduled(fmtClock.format(retry.at))}
      <button
        className="aui-error-action ml-1"
        data-testid="cancel-scheduled-retry"
        onClick={() => setScheduledRetry(retry.sessionId, null)}
        type="button"
      >
        {copy.errorRetryCancelScheduled}
      </button>
    </span>
  )
}

export interface ScheduledRetryActionProps {
  /** Session the failed message belongs to. */
  sessionId: null | string
  /** The failed assistant message the scheduled reload will target. */
  messageId: string
}

/**
 * The scheduling half of "Retry in X hours": renders the "Retry in…" menu
 * before anything is scheduled, and the countdown notice + cancel once it is.
 * Firing the reload at `at` is the scheduler's job (see
 * store/scheduled-retry) — this leaf only owns the record.
 */
export const ScheduledRetryAction: FC<ScheduledRetryActionProps> = ({ messageId, sessionId }) => {
  const retry = useStore(useMemo(() => sessionScheduledRetry(sessionId), [sessionId]))

  const schedule = useCallback(
    (atMs: number) => {
      if (!sessionId) {
        return
      }

      setScheduledRetry(sessionId, { at: atMs, messageId, sessionId })
    },
    [messageId, sessionId]
  )

  // Stale schedule pointing at a different (older) failed message still
  // shows — it belongs to this session's error card, and cancelling it here
  // is the only way to reach it once its message scrolled away.

  if (retry) {
    return <ScheduledRetryNotice retry={retry} />
  }

  return <ScheduleRetryMenu onSchedule={schedule} />
}
