import type { BotRoutineListItem } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export interface RoutinesCopy {
  title: string
  locked: string
  schedule: string
  timezone: string
  destination: string
  activate: string
  pause: string
  saving: string
}

interface RoutineRowProps {
  copy: RoutinesCopy
  disabled: boolean
  onActivate: (values: { schedule: string; timezone: string; destination: string }) => Promise<void>
  onPause: () => Promise<void>
  routine: BotRoutineListItem
}

function RoutineRow({ copy, disabled, onActivate, onPause, routine }: RoutineRowProps) {
  const [schedule, setSchedule] = useState(routine.schedule ?? '')
  const [timezone, setTimezone] = useState(routine.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone ?? '')
  const [destination, setDestination] = useState(routine.destination ?? '')
  const [busy, setBusy] = useState(false)
  const dirty = useRef(false)
  const routineId = useRef(routine.id)

  // Dirty state coordinates incoming server snapshots and is intentionally not rendered.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (routineId.current !== routine.id) {
      routineId.current = routine.id
      dirty.current = false
    }

    if (!dirty.current) {
      setSchedule(routine.schedule ?? '')
      setTimezone(routine.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone ?? '')
      setDestination(routine.destination ?? '')
    }
  }, [routine])

  const complete = Boolean(schedule.trim() && timezone.trim() && destination.trim())
  const active = routine.state === 'active'

  const run = async (action: () => Promise<void>) => {
    setBusy(true)

    try {
      await action()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-3 py-3">
      <div>
        <div className="text-xs font-medium text-(--ui-text-primary)">{routine.name}</div>
        <div className="mt-1 text-xs text-(--ui-text-secondary)">{routine.prompt}</div>
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        <label className="space-y-1 text-xs text-(--ui-text-secondary)">
          <span>{copy.schedule}</span>
          <Input aria-label={`${routine.name} ${copy.schedule}`} disabled={active || busy} onChange={event => { dirty.current = true; setSchedule(event.target.value) }} value={schedule} />
        </label>
        <label className="space-y-1 text-xs text-(--ui-text-secondary)">
          <span>{copy.timezone}</span>
          <Input aria-label={`${routine.name} ${copy.timezone}`} disabled={active || busy} onChange={event => { dirty.current = true; setTimezone(event.target.value) }} value={timezone} />
        </label>
        <label className="space-y-1 text-xs text-(--ui-text-secondary)">
          <span>{copy.destination}</span>
          <Input aria-label={`${routine.name} ${copy.destination}`} disabled={active || busy} onChange={event => { dirty.current = true; setDestination(event.target.value) }} value={destination} />
        </label>
      </div>
      {active ? (
        <Button aria-label={`${copy.pause} ${routine.name}`} disabled={busy} onClick={() => void run(onPause)} size="xs" variant="secondary">
          {busy ? copy.saving : copy.pause}
        </Button>
      ) : (
        <Button
          aria-label={`${copy.activate} ${routine.name}`}
          disabled={disabled || busy || !complete}
          onClick={() => void run(async () => {
            await onActivate({ schedule: schedule.trim(), timezone: timezone.trim(), destination: destination.trim() })
            dirty.current = false
          })}
          size="xs"
          variant="secondary"
        >
          {busy ? copy.saving : copy.activate}
        </Button>
      )}
    </div>
  )
}

interface RoutinesPanelProps {
  canActivate: boolean
  copy: RoutinesCopy
  onActivate: (routine: BotRoutineListItem, values: { schedule: string; timezone: string; destination: string }) => Promise<void>
  onPause: (routine: BotRoutineListItem) => Promise<void>
  routines: BotRoutineListItem[]
}

export function RoutinesPanel({ canActivate, copy, onActivate, onPause, routines }: RoutinesPanelProps) {
  if (!routines.length) {
    return null
  }

  return (
    <section aria-label={copy.title} className="border-t border-(--ui-stroke-tertiary) pt-3">
      <div className="text-xs font-semibold text-(--ui-text-primary)">{copy.title}</div>
      {!canActivate && <div className="mt-1 text-xs text-(--ui-text-tertiary)">{copy.locked}</div>}
      <div className="divide-y divide-(--ui-stroke-tertiary)">
        {routines.map(routine => (
          <RoutineRow
            copy={copy}
            disabled={!canActivate}
            key={routine.id}
            onActivate={values => onActivate(routine, values)}
            onPause={() => onPause(routine)}
            routine={routine}
          />
        ))}
      </div>
    </section>
  )
}
