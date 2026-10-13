import { useState } from 'react'

import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelPillLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'

import { useWorkhorseConfig, workhorseRouteReset } from './workhorse-config'

// Mirrors the orchestrator pill's chrome (model-pill.tsx): same size, same
// hover, same truncation contract. The WORKHORSE pill reads and writes the
// `delegation` config section (`delegation.model` / `delegation.provider`),
// not the live session model — it picks which model `delegate_task` subagents
// run on, so a composer can carry both the orchestrator (main agent) and the
// workhorse (subagent) pickers side by side.
const PILL = cn(
  'h-(--composer-control-size) min-w-0 max-w-40 shrink gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export interface WorkhorseModelPillProps {
  compact?: boolean
  disabled: boolean
}

export function WorkhorseModelPill({ compact = false, disabled }: WorkhorseModelPillProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.workhorse
  const { profile, connectionId, delegation, ready, commit, request } = useWorkhorseConfig()
  const [open, setOpen] = useState(false)
  const workhorseModel = String(delegation.model ?? '').trim()
  const workhorseProvider = String(delegation.provider ?? '').trim()

  const commitWorkhorse = (provider: string, model: string) =>
    commit('model', { model, provider, ...workhorseRouteReset })

  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : workhorseModel ? (
    <span className="truncate">{formatModelPillLabel(workhorseModel)}</span>
  ) : (
    <span className="truncate italic opacity-70">{copy.inherit}</span>
  )

  const title = workhorseProvider ? copy.modelTitle(workhorseProvider, workhorseModel || copy.noModel) : copy.openPicker

  const pillClass = compact
    ? cn(
        'h-(--composer-control-size) shrink-0 justify-center gap-1 rounded-md px-2 text-xs font-normal',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  return (
    <>
      <Tip label={title} side="top">
        <Button
          aria-label={copy.openPicker}
          className={pillClass}
          data-testid="workhorse-model-pill"
          disabled={disabled || !ready}
          onClick={() => setOpen(true)}
          type="button"
          variant="ghost"
        >
          <span>{copy.label}</span>
          {label}
        </Button>
      </Tip>
      <ModelPickerDialog
        contentClassName="z-[1000]"
        currentModel={workhorseModel}
        currentProvider={workhorseProvider}
        onOpenChange={setOpen}
        onReset={() => commitWorkhorse('', '')}
        onSelect={selection => void commitWorkhorse(selection.provider, selection.model)}
        open={open}
        ownerConnectionId={connectionId}
        profile={profile}
        request={request}
        resetLabel={copy.resetModel}
        sessionId={null}
        title={copy.openPicker}
      />
    </>
  )
}
