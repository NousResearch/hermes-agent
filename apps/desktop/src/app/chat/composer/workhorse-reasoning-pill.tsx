import { DEFAULT_REASONING_EFFORT } from '@hermes/shared'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { currentModelCapabilities, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { reasoningEffortLabel } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'

import { ModelOptionsContent } from '../../shell/model-edit-submenu'

import { useWorkhorseConfig } from './workhorse-config'

// Mirror of reasoning-pill.tsx for the WORKHORSE (delegation) slot: shows the
// `delegation.reasoning_effort` level and edits it through the same
// Thinking / Effort rows the orchestrator's effort pill uses. Effort can be
// overridden or reset independently even when the model inherits.
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
  const { profile, connectionId, delegation, ready, commit, request } = useWorkhorseConfig()
  const [open, setOpen] = useState(false)

  const workhorseModel = String(delegation.model ?? '').trim()
  const workhorseProvider = String(delegation.provider ?? '').trim()

  const rawEffort = String(delegation.reasoning_effort ?? '')
    .trim()
    .toLowerCase()

  const effort = rawEffort === 'false' || rawEffort === 'disabled' ? 'none' : rawEffort

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, null, connectionId),
    queryFn: () => requestModelOptions({ profile, request }),
    enabled: Boolean(workhorseModel && workhorseProvider)
  })

  const capabilities = currentModelCapabilities(modelOptions.data, workhorseProvider, workhorseModel)

  // Keep an existing override reachable even on a non-reasoning model.
  if (capabilities?.reasoning === false && !effort) {
    return null
  }

  const label = effort ? reasoningEffortLabel(effort) : copy.effortInherit
  const title = `${copy.label} · ${t.shell.modelOptions.effort}: ${label}`

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  const commitEffort = (next: string) => commit('effort', { reasoning_effort: next })

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={title}
            className={PILL}
            data-testid="workhorse-reasoning-pill"
            disabled={disabled || !ready}
            type="button"
            variant="ghost"
          >
            <span>{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-52 p-0" side="top" sideOffset={8}>
        <DropdownMenuItem onSelect={() => commitEffort('')}>{copy.resetEffort}</DropdownMenuItem>
        {/* Same Thinking / Effort rows as the main reasoning pill. No Fast
            toggle: `delegation` has no service-tier/fast field — the config
            section only carries reasoning_effort. */}
        <ModelOptionsContent
          canDisableReasoning={capabilities?.can_disable_reasoning ?? undefined}
          defaultEffort={effort && effort !== 'none' ? effort : DEFAULT_REASONING_EFFORT}
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
          reasoning={capabilities?.reasoning !== false}
        />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
