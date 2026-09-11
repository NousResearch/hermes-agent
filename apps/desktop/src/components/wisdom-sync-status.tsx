import { createWisdomSyncController, initialWisdomSyncView, wisdomSyncCopy, type WisdomSyncState } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { capabilityScoped, profileScopeKey } from '@/api/client'
import { Button } from '@/components/ui/button'
import { getWisdomSync, type ProfileScope, retryWisdomSync } from '@/hermes'
import { useI18n } from '@/i18n'
import { Loader2, RefreshCw } from '@/lib/icons'

export function WisdomSyncStatus({ profile }: { profile?: ProfileScope }) {
  const scope = capabilityScoped(profile)

  return <SyncStatus key={profileScopeKey(scope)} profile={scope} />
}

function SyncStatus({ profile }: { profile?: ProfileScope }) {
  const { t } = useI18n()
  const copy = t.skills.collective.syncRecovery ?? wisdomSyncCopy
  const [open, setOpen] = useState(false)
  const [scope] = useState(profile)
  const [view, setView] = useState(initialWisdomSyncView)
  const controller = useRef<ReturnType<typeof createWisdomSyncController> | null>(null)
  // eslint-disable-next-line no-restricted-syntax -- Effect-owned controller, disposed on close/profile change.
  useEffect(() => {
    if (!open) {
      return
    }

    const next = createWisdomSyncController({
      read: () => getWisdomSync(scope),
      retry: () => retryWisdomSync(scope),
      changed: setView
    })

    controller.current = next
    void next.refresh()

    const poll = () => {
      if (document.visibilityState !== 'hidden') {
        void next.refresh()
      }
    }

    const timer = window.setInterval(poll, 15_000)

    return () => {
      next.dispose()
      controller.current = null
      window.clearInterval(timer)
    }
  }, [open, scope])

  return (
    <section className="min-w-0 border-b border-(--ui-stroke-tertiary) py-2">
      <button
        aria-expanded={open}
        className="flex max-w-full items-center gap-2 text-sm"
        onClick={() => setOpen(value => !value)}
        type="button"
      >
        <RefreshCw aria-hidden="true" className="size-4 shrink-0" />
        <span className="break-words text-start">{copy.title}</span>
      </button>
      {open && (
        <div className="mt-3 min-w-0 space-y-3">
          <p className="text-xs text-muted-foreground">{copy.scope}</p>
          {view.error && (
            <p className="text-xs text-destructive" role="alert">
              {copy.unavailable}
            </p>
          )}
          {!view.snapshot && view.busy && (
            <p className="text-xs" role="status">
              {t.common.loading}
            </p>
          )}
          {view.snapshot && !view.error && (
            <dl aria-live="polite" className="space-y-2 text-xs">
              {(['delivery', 'operation'] as const).map(kind => {
                const counts = view.snapshot![kind]
                const states = (Object.keys(counts) as WisdomSyncState[]).filter(state => counts[state] > 0)

                return (
                  <div key={kind}>
                    <dt className="font-medium">{copy[kind]}</dt>
                    <dd className="break-words text-muted-foreground">
                      {states.length
                        ? states.map(state => (
                            <div key={state}>
                              {copy.states[state]}: {counts[state]}
                            </div>
                          ))
                        : copy.current}
                    </dd>
                  </div>
                )
              })}
            </dl>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              aria-label={t.common.refresh}
              disabled={view.busy}
              onClick={() => void controller.current?.refresh()}
              size="sm"
              title={t.common.refresh}
              type="button"
              variant="outline"
            >
              <RefreshCw aria-hidden="true" className="size-4" />
            </Button>
            <Button
              disabled={view.busy || view.error || !view.snapshot?.can_retry}
              onClick={() => void controller.current?.retry()}
              size="sm"
              type="button"
            >
              {view.busy && <Loader2 aria-hidden="true" className="size-4 animate-spin" />}
              {copy.retry}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
