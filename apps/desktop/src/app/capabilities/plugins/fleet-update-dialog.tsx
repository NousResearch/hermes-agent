import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Check, Loader2, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import {
  $fleetUpdateRequest,
  $pluginFleet,
  $pluginFleetByPlugin,
  closeFleetUpdate,
  type FleetInstall,
  fleetInstallKey,
  type FleetSkippedSource,
  type FleetUpdateOutcome,
  openFleetUpdate,
  settleFleetInstalls,
  updateFleetInstall
} from '@/store/plugin-fleet-updates'

type RowState = 'pending' | 'running' | 'done' | 'failed' | 'declined'

const UPDATE_CONCURRENCY = 3

const pinLabel = (install: FleetInstall) => install.row.catalog_version ?? install.row.catalog_sha?.slice(0, 8) ?? ''

/** Shared row geometry: status glyph · who · what · result. */
const ROW = 'flex items-center gap-2.5 px-3 py-1.5 text-[length:var(--conversation-caption-font-size)]'
/** Gateway heading inside the list — same treatment as the page's catalog section label. */
const GROUP = 'px-3 pt-2 pb-0.5 text-[0.62rem] font-medium tracking-wide uppercase text-(--ui-text-quaternary)'

function StateGlyph({ state }: { state: RowState }) {
  if (state === 'running') {
    return <Loader2 aria-hidden className="size-3.5 shrink-0 animate-spin text-(--ui-text-tertiary)" />
  }

  if (state === 'done') {
    return <Check aria-hidden className="size-3.5 shrink-0 text-(--ui-success,#4ade80)" />
  }

  if (state === 'failed') {
    return <X aria-hidden className="size-3.5 shrink-0 text-(--ui-danger,#f87171)" />
  }

  return (
    <span aria-hidden className="flex size-3.5 shrink-0 items-center justify-center">
      <span
        className={cn(
          'size-1.5 rounded-full',
          state === 'declined' ? 'bg-(--ui-text-quaternary)' : 'bg-(--ui-text-tertiary)'
        )}
      />
    </span>
  )
}

/**
 * Review-then-apply dialog for updating agent plugins on every gateway and
 * profile that runs an outdated catalog pin. One line per install; the same
 * lines become the live progress list once the user confirms. A pin that
 * widens a plugin asks ONCE for the whole batch rather than per profile.
 */
export function FleetUpdateDialog({
  installs,
  onClose,
  onFinished,
  open,
  skipped,
  title
}: {
  installs: FleetInstall[]
  onClose: () => void
  /** Called once the batch settles with how many installs changed. */
  onFinished?: (applied: number) => void
  open: boolean
  skipped: FleetSkippedSource[]
  title: string
}) {
  const { t } = useI18n()
  const p = t.skills.plugins.fleet
  const [states, setStates] = useState<Record<string, RowState>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [phase, setPhase] = useState<'review' | 'running' | 'done'>('review')

  // A fresh open reviews a fresh list.
  useEffect(() => {
    if (open) {
      setStates({})
      setErrors({})
      setPhase('review')
    }
  }, [open])

  const gateways = useMemo(
    () => new Set(installs.map(install => install.target.connectionId ?? 'local')).size,
    [installs]
  )

  const multiPlugin = useMemo(() => new Set(installs.map(install => install.row.name)).size > 1, [installs])
  const showGateway = gateways > 1 || skipped.length > 0

  // One group per gateway, in scan order; a gateway heading replaces a
  // per-row label so the profile column stays straight.
  const groups = useMemo(() => {
    const byGateway = new Map<string, { label: string; installs: FleetInstall[] }>()

    for (const install of installs) {
      const id = install.target.connectionId ?? 'local'
      const group = byGateway.get(id) ?? { label: install.target.connectionLabel, installs: [] }
      group.installs.push(install)
      byGateway.set(id, group)
    }

    return [...byGateway.entries()]
  }, [installs])

  const setState = (install: FleetInstall, state: RowState, error?: string) => {
    const key = fleetInstallKey(install)
    setStates(current => ({ ...current, [key]: state }))

    if (error) {
      setErrors(current => ({ ...current, [key]: error }))
    }
  }

  const run = async () => {
    setPhase('running')
    const queue = [...installs]
    const settled = new Set<string>()
    const needsConsent: { install: FleetInstall; outcome: Extract<FleetUpdateOutcome, { kind: 'consent' }> }[] = []
    let applied = 0

    const apply = async (install: FleetInstall, accept: boolean) => {
      setState(install, 'running')
      const outcome = await updateFleetInstall(install, accept)

      if (outcome.kind === 'consent') {
        needsConsent.push({ install, outcome })
        setState(install, 'pending')

        return
      }

      if (outcome.kind === 'failed') {
        setState(install, 'failed', outcome.error)

        return
      }

      applied += outcome.kind === 'applied' ? 1 : 0
      settled.add(fleetInstallKey(install))
      setState(install, 'done')
    }

    const drain = async (items: FleetInstall[], accept: boolean) => {
      const work = [...items]

      await Promise.all(
        Array.from({ length: Math.min(UPDATE_CONCURRENCY, work.length) }, async () => {
          for (let next = work.shift(); next; next = work.shift()) {
            await apply(next, accept)
          }
        })
      )
    }

    await drain(queue, false)

    if (needsConsent.length > 0) {
      const lines = [...new Set(needsConsent.flatMap(entry => entry.outcome.deltaLines))]
      const names = [...new Set(needsConsent.map(entry => entry.install.row.name))].join(', ')

      const ok = await confirm({
        confirmLabel: t.skills.plugins.updateConsentConfirm,
        description: [p.consentBody(names, needsConsent.length), ...lines].join('\n'),
        title: t.skills.plugins.updateConsentTitle(names)
      })

      const pending = needsConsent.splice(0).map(entry => entry.install)

      if (ok) {
        await drain(pending, true)
      } else {
        pending.forEach(install => setState(install, 'declined'))
      }
    }

    settleFleetInstalls(settled)
    setPhase('done')
    onFinished?.(applied)
  }

  const running = phase === 'running'
  const settledStates = Object.values(states)
  const updatedCount = settledStates.filter(state => state === 'done').length
  const failed = settledStates.filter(state => state === 'failed').length
  const declined = settledStates.filter(state => state === 'declined').length

  return (
    <Dialog onOpenChange={value => !value && !running && onClose()} open={open}>
      <DialogContent className="max-w-lg gap-3">
        <DialogHeader>
          <DialogTitle className="truncate pr-6" title={title}>
            {title}
          </DialogTitle>
          <DialogDescription>
            {phase === 'done'
              ? failed > 0
                ? p.summaryFailed(updatedCount, failed)
                : declined > 0
                  ? p.summaryDeclined(updatedCount, declined)
                  : p.summaryDone(updatedCount)
              : p.summary(installs.length, gateways)}
          </DialogDescription>
        </DialogHeader>

        <div
          className="max-h-80 overflow-y-auto rounded-lg border border-(--ui-stroke-tertiary) py-1"
          data-testid="plugin-fleet-update-list"
          role="list"
        >
          {groups.map(([gatewayId, group]) => (
            <div key={gatewayId} role="group">
              {showGateway && group.label && <div className={GROUP}>{group.label}</div>}
              {group.installs.map(install => {
                const key = fleetInstallKey(install)
                const state = states[key] ?? 'pending'
                const error = errors[key]
                const { target, row } = install

                return (
                  <div className={ROW} data-state={state} key={key} role="listitem">
                    <StateGlyph state={state} />
                    <div className="flex min-w-0 flex-1 items-baseline gap-1.5">
                      <span className="truncate font-medium text-foreground">{target.profile}</span>
                      {target.current && <span className="shrink-0 text-(--ui-text-quaternary)">{p.thisProfile}</span>}
                    </div>
                    {multiPlugin && <span className="max-w-40 truncate text-(--ui-text-secondary)">{row.name}</span>}
                    <span className="shrink-0 font-mono text-[0.68rem] text-(--ui-text-tertiary)">
                      {state === 'failed' ? (
                        <Tip label={error ?? ''}>
                          <span className="font-sans text-(--ui-danger,#f87171)">{p.failed}</span>
                        </Tip>
                      ) : state === 'declined' ? (
                        <span className="font-sans">{p.declined}</span>
                      ) : state === 'done' ? (
                        <span className="text-(--ui-text-secondary)">{pinLabel(install)}</span>
                      ) : (
                        <>
                          {row.version ? `${row.version} → ` : ''}
                          <span className="text-(--ui-text-secondary)">{pinLabel(install)}</span>
                        </>
                      )}
                    </span>
                  </div>
                )
              })}
            </div>
          ))}
          {skipped.map(source => (
            <div
              className={cn(GROUP, 'flex items-center justify-between')}
              key={`skipped-${source.connectionId}-${source.label}`}
            >
              <span>{source.label}</span>
              <span className="normal-case tracking-normal">
                {
                  {
                    idle: p.skippedIdle,
                    offline: p.skippedOffline,
                    'sign-in': p.skippedSignIn,
                    unreadable: p.skippedUnreadable
                  }[source.reason]
                }
              </span>
            </div>
          ))}
        </div>

        <DialogFooter>
          {phase === 'done' ? (
            <Button onClick={onClose} size="sm">
              {p.done}
            </Button>
          ) : (
            <>
              <Button disabled={running} onClick={onClose} size="sm" variant="ghost">
                {t.common.cancel}
              </Button>
              <Button disabled={running || installs.length === 0} onClick={() => void run()} size="sm">
                {running ? p.updating : p.confirm(installs.length)}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** Plugins-tab header button: present only while some profile on some gateway is behind its pin. */
export function FleetUpdateAllButton() {
  const { t } = useI18n()
  const f = t.skills.plugins.fleet
  const outdated = useStore($pluginFleet)?.outdated ?? []
  const plugins = useStore($pluginFleetByPlugin).size

  if (outdated.length === 0) {
    return null
  }

  return (
    <Tip label={f.updateAllTip(outdated.length, plugins)}>
      <Button
        data-testid="plugin-fleet-update-all"
        onClick={() => openFleetUpdate(outdated, f.titleAll)}
        size="sm"
        type="button"
        variant="outline"
      >
        {f.updateAll(outdated.length)}
      </Button>
    </Tip>
  )
}

/** Mounts the review dialog for whichever fleet update was requested (header button or row caret). */
export function FleetUpdateHost({ onApplied }: { onApplied: (applied: number) => void }) {
  const request = useStore($fleetUpdateRequest)
  const skipped = useStore($pluginFleet)?.skipped ?? []

  return (
    <FleetUpdateDialog
      installs={request?.installs ?? []}
      onClose={closeFleetUpdate}
      onFinished={applied => applied > 0 && onApplied(applied)}
      open={request !== null}
      skipped={skipped}
      title={request?.title ?? ''}
    />
  )
}
