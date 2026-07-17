import { useStore } from '@nanostores/react'
import { useRef } from 'react'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  dropdownMenuRow,
  dropdownMenuSectionLabel
} from '@/components/ui/dropdown-menu'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { applyReasoningPatch } from '@/lib/patch-reasoning'
import {
  DEFAULT_REASONING_EFFORT,
  formatReasoningPillLabel,
  isThinkingEnabled,
  normalizeReasoningEffort,
  REASONING_EFFORT_OPTIONS
} from '@/lib/model-status-label'
import { cn } from '@/lib/utils'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort
} from '@/store/session'

import type { HermesGateway } from '@/hermes'

const PILL = cn(
  'h-(--composer-control-size) max-w-24 shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

// Adapts a HermesGateway instance to the JSON-RPC function shape
// `applyReasoningPatch` expects (shared with the per-row model picker submenu).
const gatewayAsRequest =
  (gateway: HermesGateway) =>
  <T,>(method: string, params?: Record<string, unknown>): Promise<T> =>
    gateway.request<T>(method, params)

/**
 * Composer reasoning-effort selector — the relocated reasoning suffix from
 * the old combined model pill. Now its own button so long model names no
 * longer truncate the effort ("Med" / "High" / "Off").
 *
 * Hidden when the active model does not support reasoning effort
 * (`reasoningCapable === false`); shown optimistically (Hermes default
 * `Med`) while capability is loading.
 */
export function ReasoningPill({
  compact = false,
  disabled,
  gateway,
  reasoningCapable
}: {
  compact?: boolean
  disabled: boolean
  gateway: HermesGateway | null
  /** `null` while model options load — show the pill, default to "Med". */
  reasoningCapable: boolean | null
}) {
  const { t } = useI18n()
  const copy = t.shell.modelOptions
  const effort = useStore($currentReasoningEffort)
  const model = useStore($currentModel)
  const provider = useStore($currentProvider)
  const activeSessionId = useStore($activeSessionId)

  // Monotonic generation token — bumped on every click. Captured per-call and
  // re-checked on RPC failure via `applyReasoningPatch`'s `latestGeneration`
  // accessor so a stale revert (A→B race where A's RPC fails after B already
  // committed) does not clobber B. Mirrors `usageRequestRef` in
  // `app/command-center/index.tsx` and `resumeRequestRef` in
  // `app/session/hooks/use-session-actions/index.ts`.
  const generationRef = useRef(0)

  // No model yet, or the active model doesn't support reasoning — the pill
  // is irrelevant. Show nothing rather than a disabled "Med" that would lie
  // about capability. We require BOTH a model and a provider to render, not
  // just one — `patchReasoning` writes a preset keyed on both, and an empty
  // key would pollute the preset store (see the guard inside `patchReasoning`
  // below for the belt-and-suspenders form).
  if (reasoningCapable === false || !model.trim() || !provider.trim()) {
    return null
  }

  const effortValue = normalizeReasoningEffort(effort)
  const thinkingOn = isThinkingEnabled(effort)

  // Writes the preset, the live session atom, and pushes `config.set` to the
  // gateway. See `lib/patch-reasoning.ts` for the shared write path used by
  // both this pill and the per-row model picker submenu, the revert
  // semantics on RPC failure, and the generation-guard semantics for the
  // A→B click race.
  const patchReasoning = async (next: string) => {
    const generation = ++generationRef.current

    return applyReasoningPatch({
      failMessage: copy.updateFailed,
      generation,
      isActive: true,
      latestGeneration: () => generationRef.current,
      model,
      next,
      prev: effort,
      provider,
      request: gateway ? gatewayAsRequest(gateway) : null,
      sessionId: activeSessionId
    })
  }

  const pillClass = compact
    ? cn(
        'size-(--composer-control-size) shrink-0 justify-center gap-0 rounded-md p-0',
        'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )
    : PILL

  const label = compact ? (
    <ChevronDown className="size-3.5 shrink-0 opacity-70" />
  ) : (
    <>
      <span className="truncate">{formatReasoningPillLabel(effort)}</span>
      <ChevronDown className="size-2.5 shrink-0 opacity-50" />
    </>
  )

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.effort}
          className={pillClass}
          disabled={disabled}
          title={copy.effort}
          type="button"
          variant="ghost"
        >
          {label}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44 p-0" side="top" sideOffset={8}>
        <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.thinking}</DropdownMenuLabel>
        <DropdownMenuItem className={dropdownMenuRow} onSelect={event => event.preventDefault()}>
          <span>{copy.thinking}</span>
          <Switch
            checked={thinkingOn}
            className="ml-auto"
            onCheckedChange={checked =>
              void patchReasoning(checked ? effortValue || DEFAULT_REASONING_EFFORT : 'none')
            }
            size="xs"
          />
        </DropdownMenuItem>
        <DropdownMenuSeparator className="mx-0" />
        <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.effort}</DropdownMenuLabel>
        <DropdownMenuRadioGroup onValueChange={value => void patchReasoning(value)} value={effortValue}>
          {REASONING_EFFORT_OPTIONS.map(option => (
            <DropdownMenuRadioItem
              className={dropdownMenuRow}
              key={option.value}
              onSelect={event => event.preventDefault()}
              value={option.value}
            >
              {copy[option.labelKey]}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
