import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { ModelPickerDialog } from '@/components/model-picker'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { formatModelPillLabel } from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

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
  const profile = useStore($activeGatewayProfile)
  const [open, setOpen] = useState(false)

  // The delegation config section is profile-scoped config.yaml, shared with
  // Settings → Advanced (the same record both read and write through
  // HERMES_CONFIG_KEY). An absent provider means "inherit the parent agent"
  // (the delegation runtime resolves it as auto), so the pill renders an
  // "inherit" label when nothing is pinned.
  const { data: config } = useHermesConfigRecord(profile)

  const delegation = useMemo(() => {
    const raw = config?.delegation

    return raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  }, [config])

  const workhorseModel = String(delegation.model ?? '').trim()
  const workhorseProvider = String(delegation.provider ?? '').trim()

  const commitWorkhorse = async (provider: string, model: string) => {
    try {
      await saveHermesConfig({ delegation: { model, provider } }, profile)
      // Mirror the saved section into the shared record cache so this pill (and
      // Settings → Advanced) paints the pick without waiting for a refetch.
      hermesConfigCacheWriter(profile)(prev => ({
        ...(prev ?? {}),
        delegation: {
          ...(prev?.delegation && typeof prev.delegation === 'object' ? prev.delegation : {}),
          model,
          provider
        }
      }))
    } catch (err) {
      notifyError(err, t.shell.modelOptions.updateFailed)
    }
  }

  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : workhorseModel ? (
    <span className="truncate">{formatModelPillLabel(workhorseModel)}</span>
  ) : (
    <span className="truncate italic opacity-70">{copy.inherit}</span>
  )

  const title = workhorseProvider
    ? copy.modelTitle(workhorseProvider, workhorseModel || copy.noModel)
    : copy.openPicker

  const pillClass = compact
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
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
          disabled={disabled}
          onClick={() => setOpen(true)}
          type="button"
          variant="ghost"
        >
          {label}
        </Button>
      </Tip>
      <ModelPickerDialog
        contentClassName="z-[1000]"
        currentModel={workhorseModel}
        currentProvider={workhorseProvider}
        onOpenChange={setOpen}
        onSelect={selection => void commitWorkhorse(selection.provider, selection.model)}
        open={open}
        profile={profile}
        sessionId={null}
      />
    </>
  )
}
