import { DEFAULT_REASONING_EFFORT } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { currentModelCapabilities, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { reasoningEffortLabel } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { ModelOptionsContent } from '../../shell/model-edit-submenu'

// Mirror of reasoning-pill.tsx for the WORKHORSE (delegation) slot: shows the
// `delegation.reasoning_effort` level and edits it through the same
// Thinking / Effort rows the orchestrator's effort pill uses. Hidden entirely
// while no workhorse model is pinned — an inherited (auto) subagent model has
// no reasoning level of its own to set.
const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export interface WorkhorseReasoningPillProps {
  disabled: boolean
}

export function WorkhorseReasoningPill({ disabled }: WorkhorseReasoningPillProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.workhorse
  const profile = useStore($activeGatewayProfile)
  const [open, setOpen] = useState(false)

  const { data: config } = useHermesConfigRecord(profile)

  const delegation = useMemo(() => {
    const raw = config?.delegation

    return raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  }, [config])

  const workhorseModel = String(delegation.model ?? '').trim()
  const workhorseProvider = String(delegation.provider ?? '').trim()
  const rawEffort = String(delegation.reasoning_effort ?? '')
    .trim()
    .toLowerCase()
  const effort = rawEffort === 'false' || rawEffort === 'disabled' ? 'none' : rawEffort

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile),
    queryFn: () => requestModelOptions({ profile }),
    enabled: Boolean(workhorseModel && workhorseProvider)
  })

  const capabilities = currentModelCapabilities(modelOptions.data, workhorseProvider, workhorseModel)

  // No pinned workhorse model → no reasoning slot to edit. Same contract as
  // the orchestrator pill hiding its effort control for a model with no
  // reasoning control.
  if (!workhorseModel || !workhorseProvider) {
    return null
  }

  if (capabilities?.reasoning === false) {
    return null
  }

  const label = effort ? reasoningEffortLabel(effort) : copy.effortInherit
  const title = `${t.shell.modelOptions.effort}: ${label}`

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  const commitEffort = async (next: string) => {
    try {
      await saveHermesConfig({ delegation: { reasoning_effort: next } }, profile)
      hermesConfigCacheWriter(profile)(prev => ({
        ...(prev ?? {}),
        delegation: {
          ...((prev?.delegation && typeof prev.delegation === 'object'
            ? prev.delegation
            : delegation) as Record<string, unknown>),
          reasoning_effort: next
        }
      }))
    } catch (err) {
      notifyError(err, t.shell.modelOptions.updateFailed)
    }
  }

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={title}
            className={PILL}
            data-testid="workhorse-reasoning-pill"
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            <span>{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-52 p-0" side="top" sideOffset={8}>
        {/* Same Thinking / Effort rows as the main reasoning pill. No Fast
            toggle: `delegation` has no service-tier/fast field — the config
            section only carries reasoning_effort. */}
        <ModelOptionsContent
          canDisableReasoning={capabilities?.can_disable_reasoning ?? undefined}
          defaultEffort={effort || DEFAULT_REASONING_EFFORT}
          effort={effort}
          fastControl={{ kind: 'none' }}
          isActive
          model={workhorseModel}
          onSelectModel={() => undefined}
          onSetOptions={patch => {
            if (patch.effort !== undefined) {
              void commitEffort(patch.effort)
            }
          }}
          provider={workhorseProvider}
          reasoning
        />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
