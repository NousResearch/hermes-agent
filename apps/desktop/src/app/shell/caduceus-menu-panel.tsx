import { useStore } from '@nanostores/react'

import { Switch } from '@/components/ui/switch'
import { Cpu, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $caduceus,
  openTheater,
  setCaduceusEnabled,
  setCaduceusLocal,
  setCaduceusRouter
} from '@/store/caduceus'
import { $workflowRun, workflowIsLive } from '@/store/workflow'

interface CaduceusMenuPanelProps {
  sessionId: null | string
}

interface ToggleRowProps {
  checked: boolean
  description: string
  disabled?: boolean
  hint?: string
  label: string
  onToggle: (next: boolean) => void
}

function ToggleRow({ checked, description, disabled = false, hint, label, onToggle }: ToggleRowProps) {
  return (
    <div className={cn('flex items-start justify-between gap-3 px-3 py-2', disabled && 'opacity-55')}>
      <div className="min-w-0">
        <div className="text-xs font-medium">{label}</div>
        <div className="text-[0.66rem] leading-snug text-muted-foreground">{hint || description}</div>
      </div>
      <Switch
        aria-label={label}
        checked={checked}
        className="mt-0.5"
        disabled={disabled}
        onCheckedChange={next => onToggle(Boolean(next))}
      />
    </div>
  )
}

/** The popover behind the status-bar Caduceus chip: mode + Auto Router + Local. */
export function CaduceusMenuPanel({ sessionId }: CaduceusMenuPanelProps) {
  const caduceus = useStore($caduceus)
  const run = useStore($workflowRun)
  const live = workflowIsLive(run)
  const local = caduceus.local
  const noModels = local.models.length === 0

  return (
    <div className="text-sm">
      <div className="flex items-center gap-2 px-3 py-2.5">
        <Sparkles className={cn('size-3.5', caduceus.enabled ? 'text-amber-300' : 'text-muted-foreground')} />
        <span className="font-medium">Caduceus</span>
        {caduceus.enabled && (
          <span className="ml-auto text-[0.62rem] font-semibold uppercase tracking-[0.12em] text-amber-300">
            {caduceus.split ? 'split' : 'on'}
          </span>
        )}
      </div>

      <div className="border-t border-border/50">
        <ToggleRow
          checked={caduceus.enabled}
          description="Deep planning — a live to-do list, driven methodically. Say “workflow” to fan out."
          label="Caduceus mode"
          onToggle={next => void setCaduceusEnabled(sessionId, next)}
        />
        <ToggleRow
          checked={caduceus.routerEnabled}
          description="Route each worker to the cheapest capable model; orchestrator keeps your model."
          disabled={!caduceus.enabled}
          hint={!caduceus.enabled ? 'Turn on Caduceus to use the Auto Router.' : undefined}
          label="Auto Router"
          onToggle={next => void setCaduceusRouter(sessionId, next)}
        />
        <ToggleRow
          checked={local.enabled}
          description="Run workflow workers on local GPU models (hot-swap + capacity-aware parallelism)."
          disabled={!caduceus.enabled || noModels}
          hint={
            noModels
              ? 'Add models under caduceus.local in config.yaml, then restart.'
              : !caduceus.enabled
                ? 'Turn on Caduceus to use local workers.'
                : undefined
          }
          label="Local workers"
          onToggle={next => void setCaduceusLocal(sessionId, next)}
        />
      </div>

      {local.enabled && !noModels && (
        <div className="border-t border-border/50 px-3 py-2">
          <div className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80">
            Local models
          </div>
          <ul className="mt-1.5 space-y-1">
            {local.models.map(m => (
              <li className="flex items-center justify-between gap-2 text-[0.7rem]" key={m.id}>
                <span className="flex min-w-0 items-center gap-1.5">
                  <Cpu className="size-3 shrink-0 text-muted-foreground" />
                  <span className="truncate" title={m.card || m.id}>
                    {m.id}
                    {m.default ? ' ·default' : ''}
                  </span>
                </span>
                <span className="shrink-0 text-[0.62rem] text-muted-foreground">
                  {m.maxContext ? `${Math.round(m.maxContext / 1000)}k·` : ''}
                  {m.maxSlots}×
                </span>
              </li>
            ))}
          </ul>
          <div className="mt-1.5 text-[0.62rem] text-muted-foreground">
            {local.loaded
              ? `Loaded: ${local.loaded.model} [${local.loaded.profile}], ${local.loaded.slots} slot(s).`
              : 'Loaded: nothing (loads on demand).'}
          </div>
        </div>
      )}

      {live && (
        <div className="border-t border-border/50 px-3 py-2">
          <button
            className="text-[0.7rem] font-medium text-sky-300 hover:text-sky-200"
            onClick={openTheater}
            type="button"
          >
            Open the Orchestration Theater →
          </button>
        </div>
      )}
    </div>
  )
}
