import { createWisdomMuteController, initialWisdomMuteState, type WisdomMuteDuration } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { capabilityScoped, profileScopeKey } from '@/api/client'
import { Button } from '@/components/ui/button'
import { chooseWisdomMute, getWisdomMute, prepareWisdomMute, type ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, RefreshCw, Volume2, VolumeX } from '@/lib/icons'

export function WisdomNotificationSettings({ profile }: { profile?: ProfileScope }) {
  const scope = capabilityScoped(profile)

  return <Settings key={profileScopeKey(scope)} profile={scope} />
}

function Settings({ profile }: { profile?: ProfileScope }) {
  const { t, locale } = useI18n()
  const copy = t.skills.collective.notificationPreferences
  const [open, setOpen] = useState(false)
  const [state, setState] = useState(initialWisdomMuteState)
  const [scope] = useState(profile)
  const controller = useRef<ReturnType<typeof createWisdomMuteController> | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- Effect-owned imperative controller, not a mirrored reactive value.
  useEffect(() => {
    if (!open) {
      return
    }

    const next = createWisdomMuteController({
      prepare: () => prepareWisdomMute(scope),
      read: () => getWisdomMute(scope),
      choose: (id, duration) => chooseWisdomMute(id, duration, scope),
      changed: setState
    })

    controller.current = next
    void next.refresh()

    const poll = () => {
      if (document.visibilityState !== 'hidden') {
        void next.poll()
      }
    }

    const timer = window.setInterval(poll, 15_000)
    window.addEventListener('focus', poll)

    return () => {
      next.dispose()
      controller.current = null
      window.clearInterval(timer)
      window.removeEventListener('focus', poll)
    }
  }, [open, scope])

  const muted = state.snapshot?.mute?.muted
  const until = state.snapshot?.mute?.muted_until
  const sync = state.snapshot?.sync?.preference_sync

  const notice =
    sync === 'pending' || sync === 'syncing'
      ? copy.pending
      : sync === 'failed'
        ? copy.failed
        : sync === 'conflict'
          ? copy.conflict
          : sync === 'expired'
            ? copy.expired
            : null

  const Icon = muted ? VolumeX : Volume2
  const choice = state.choice === undefined ? '' : (state.choice ?? 'off')

  return (
    <section className="border-b border-(--ui-stroke-tertiary) min-w-0 py-2">
      <button
        aria-expanded={open}
        className="flex max-w-full items-center gap-2 text-sm"
        onClick={() => setOpen(value => !value)}
        type="button"
      >
        <Icon aria-hidden="true" className="size-4 shrink-0" />
        <span className="break-words text-start">{copy.title}</span>
      </button>
      {open && (
        <div className="mt-3 min-w-0 space-y-3">
          <p className="text-muted-foreground text-xs">{copy.scope}</p>
          <p className="break-words text-sm" role="status">
            {state.snapshot?.mute
              ? muted
                ? copy.muted
                : copy.on
              : state.busy
                ? t.common.loading
                : t.skills.collective.unavailable}
            {muted && until && (
              <>
                {' '}
                · <time dateTime={until}>{new Date(until).toLocaleString(locale)}</time>
              </>
            )}
            {muted && state.snapshot?.mute?.forever && <> · {copy.forever}</>}
          </p>
          {notice && (
            <p className="text-xs" role="status">
              {notice}
            </p>
          )}
          {state.error && (
            <p className="text-destructive text-xs" role="alert">
              {t.skills.collective.unavailable}
            </p>
          )}
          <form
            className="flex min-w-0 flex-wrap items-center gap-2"
            onSubmit={event => {
              event.preventDefault()
              void controller.current?.apply()
            }}
          >
            <select
              aria-label={copy.title}
              className="border-(--ui-stroke-tertiary) bg-background h-9 min-w-0 max-w-full flex-1 rounded border px-2 text-sm"
              disabled={state.busy || !state.control || state.retryingChoice}
              onChange={event =>
                controller.current?.select(
                  event.target.value === 'off' ? null : (event.target.value as WisdomMuteDuration)
                )
              }
              value={choice}
            >
              <option disabled value="">
                {copy.title}
              </option>
              <option value="off">{copy.on}</option>
              <option value="1_day">{copy.day}</option>
              <option value="1_week">{copy.week}</option>
              <option value="30_days">{copy.month}</option>
              <option value="forever">{copy.forever}</option>
            </select>
            <Button
              aria-label={t.common.refresh}
              className="h-9 shrink-0"
              disabled={state.busy}
              onClick={() => void controller.current?.refresh()}
              size="sm"
              title={t.common.refresh}
              type="button"
              variant="outline"
            >
              <RefreshCw aria-hidden="true" className="size-4" />
            </Button>
            <Button
              className="h-9 shrink-0"
              disabled={state.busy || (!state.retryingChoice && (state.choice === undefined || !state.control))}
              size="sm"
              type="submit"
            >
              {state.busy && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
              {state.retryingChoice ? t.common.retry : t.common.save}
            </Button>
          </form>
        </div>
      )}
    </section>
  )
}
