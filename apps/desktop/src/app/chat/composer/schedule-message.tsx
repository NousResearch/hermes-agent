import { useEffect, useMemo, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { StatusRow } from '@/components/chat/status-row'
import { StatusSection } from '@/components/chat/status-section'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { iconSize, Trash2 } from '@/lib/icons'
import { useSessionSlice } from '@/lib/use-session-slice'
import { notifyError } from '@/store/notifications'
import {
  $scheduledMessagesBySession,
  defaultDueInput,
  formatLocalDue,
  fromWire,
  isFutureDueInput,
  setScheduledMessages
} from '@/store/scheduled-messages'

import { GHOST_ICON_BTN } from './control-classes'

/**
 * "Schedule" beside Send, and the pending list above the composer (#111873).
 *
 * Deliberately the messenger gesture and nothing more: write a draft, pick a
 * date and time, and the exact text arrives later as a normal user turn in this
 * chat. No provider, model, delivery target or recurrence field exists here —
 * anything advanced belongs to the scheduled-jobs UI, not the composer.
 *
 * The backend is authoritative for the list (it persists it and fires it); this
 * is a cache that refreshes on mount, after every mutation, and on a slow poll
 * so an item that fires while the panel is on screen disappears on its own.
 */

const REFRESH_MS = 15_000

interface ScheduleControlProps {
  /** Text of the draft being scheduled, read at click time (DOM-truth, not a lagging render). */
  getText: () => string
  /** Fired after the backend accepted the item, so the draft can be cleared. */
  onScheduled: () => void
  sessionId: string
}

export function ScheduleMessageControl({ getText, onScheduled, sessionId }: ScheduleControlProps) {
  const { t } = useI18n()
  const c = t.composer
  const { requestGateway } = useGatewayRequest()
  const [open, setOpen] = useState(false)
  const [dueValue, setDueValue] = useState(() => defaultDueInput())
  const [submitting, setSubmitting] = useState(false)
  const valid = useMemo(() => isFutureDueInput(dueValue), [dueValue])

  const schedule = async () => {
    const text = getText().trim()

    if (!text || !valid || submitting) {
      return
    }

    setSubmitting(true)

    try {
      const result = await requestGateway<{ messages?: unknown }>('schedule.message.create', {
        session_id: sessionId,
        text,
        // The picker is a local wall clock, so hand the backend the epoch ms of
        // that instant; it stores seconds and the panel renders local time back.
        due_at: new Date(dueValue).getTime() / 1000
      })

      setScheduledMessages(sessionId, fromWire(result?.messages))
      setOpen(false)
      onScheduled()
    } catch (error) {
      notifyError(error, c.scheduledSendFailed)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Popover
      onOpenChange={next => {
        setOpen(next)

        if (next) {
          setDueValue(defaultDueInput())
        }
      }}
      open={open}
    >
      <Tip label={c.schedule}>
        <PopoverTrigger asChild>
          <Button aria-label={c.schedule} className={GHOST_ICON_BTN} size="icon" type="button" variant="ghost">
            <Codicon name="calendar" size="0.875rem" />
          </Button>
        </PopoverTrigger>
      </Tip>
      <PopoverContent align="end" className="w-72 space-y-2 p-3" side="top">
        <div className="text-xs font-medium text-foreground/90">{c.scheduleTitle}</div>
        <label className="block text-xs text-muted-foreground" htmlFor="composer-schedule-due">
          {c.scheduleDue}
        </label>
        <Input
          id="composer-schedule-due"
          onChange={event => setDueValue(event.target.value)}
          type="datetime-local"
          value={dueValue}
        />
        {valid ? null : <div className="text-xs text-(--ui-text-tertiary)">{c.scheduleInvalid}</div>}
        <Button
          disabled={!valid || submitting}
          onClick={() => void schedule()}
          size="sm"
          type="button"
          variant="default"
        >
          {c.scheduleConfirm}
        </Button>
      </PopoverContent>
    </Popover>
  )
}

/** One pending item: its local scheduled time, a preview, and Cancel. */
export function ScheduledMessagesPanel({ sessionId }: { sessionId: null | string }) {
  const { t } = useI18n()
  const c = t.composer
  const { requestGateway } = useGatewayRequest()
  const messages = useSessionSlice($scheduledMessagesBySession, sessionId ?? null)

  useEffect(() => {
    if (!sessionId) {
      return
    }

    let cancelled = false

    const refresh = async () => {
      try {
        const result = await requestGateway<{ messages?: unknown }>('schedule.message.list', {
          session_id: sessionId
        })

        if (!cancelled) {
          setScheduledMessages(sessionId, fromWire(result?.messages))
        }
      } catch {
        // A refresh is not a user action: a dropped poll must not toast. The
        // next poll (or the next mutation) repaints from the backend.
      }
    }

    void refresh()

    const timer = window.setInterval(() => void refresh(), REFRESH_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [requestGateway, sessionId])

  if (!sessionId || messages.length === 0) {
    return null
  }

  const cancel = async (messageId: string) => {
    try {
      const result = await requestGateway<{ messages?: unknown }>('schedule.message.cancel', {
        message_id: messageId,
        session_id: sessionId
      })

      setScheduledMessages(sessionId, fromWire(result?.messages))
    } catch (error) {
      notifyError(error, c.scheduledCancelFailed)
    }
  }

  return (
    <StatusSection
      icon={<Codicon className="text-muted-foreground/70" name="calendar" size="0.8rem" />}
      label={c.scheduled(messages.length)}
    >
      {messages.map(message => (
        <StatusRow
          key={message.id}
          leading={
            <span className="shrink-0 text-[0.7rem] tabular-nums text-muted-foreground/80">
              {formatLocalDue(message.dueAt)}
            </span>
          }
          trailing={
            <Tip label={c.scheduledCancel}>
              <Button
                aria-label={c.scheduledCancel}
                className="size-5 rounded-md"
                onClick={() => void cancel(message.id)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Trash2 className={iconSize.xs} />
              </Button>
            </Tip>
          }
        >
          <span className="truncate">{(message.displayText || message.text).trim()}</span>
        </StatusRow>
      ))}
    </StatusSection>
  )
}
