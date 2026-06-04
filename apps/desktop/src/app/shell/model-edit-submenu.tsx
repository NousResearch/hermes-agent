import { useStore } from '@nanostores/react'

import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator,
  DropdownMenuSubContent
} from '@/components/ui/dropdown-menu'
import { Switch } from '@/components/ui/switch'
import { notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $currentReasoningEffort,
  setCurrentFastMode,
  setCurrentReasoningEffort
} from '@/store/session'

// Hermes' real reasoning levels (see VALID_REASONING_EFFORTS); `none` is owned
// by the Thinking toggle, not the radio.
const EFFORT_OPTIONS = [
  { value: 'minimal', label: 'Minimal' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'xhigh', label: 'Max' }
] as const

/** How "fast" is achieved for a given model — two different mechanisms:
 *  - `param`: the Anthropic/OpenAI `speed=fast` request parameter.
 *  - `variant`: a separate `…-fast` sibling model selected via the model field.
 */
export type FastControl =
  | { kind: 'none' }
  | { kind: 'param'; on: boolean }
  | { kind: 'variant'; baseId: string; fastId: string; on: boolean }

/** Resolve the fast mechanism for a model: prefer the speed=fast parameter
 *  when the backend supports it, else fall back to a `…-fast` sibling model. */
export function resolveFastControl(
  model: string,
  providerModels: readonly string[],
  paramSupported: boolean,
  currentFastMode: boolean
): FastControl {
  if (paramSupported) {
    return { kind: 'param', on: currentFastMode }
  }

  if (/-fast$/i.test(model)) {
    const baseId = model.replace(/-fast$/i, '')

    // Only a toggle if there's a base to switch back to; otherwise it's a
    // standalone fast model with no "off" state.
    return providerModels.includes(baseId)
      ? { kind: 'variant', baseId, fastId: model, on: true }
      : { kind: 'none' }
  }

  const fastId = `${model}-fast`

  if (providerModels.includes(fastId)) {
    return { kind: 'variant', baseId: model, fastId, on: false }
  }

  // Fast isn't natively offered here, but if the session still has the speed
  // param on (carried over from a previous model), expose the toggle so it can
  // be turned off rather than stranded.
  if (currentFastMode) {
    return { kind: 'param', on: true }
  }

  return { kind: 'none' }
}

interface ModelEditSubmenuProps {
  /** How fast mode is offered for this model (param toggle vs. variant swap). */
  fastControl: FastControl
  /** Whether this row's model is the active one. */
  isActive: boolean
  /** Switch to this model (resolves false on failure). Awaited before applying
   *  edits when not active so a failed switch doesn't write to the old model. */
  onActivate: () => Promise<boolean> | void
  /** Switch to a specific model id (used to swap base ⇄ -fast variant). */
  onSelectModel: (model: string) => Promise<boolean> | void
  /** Whether this model supports reasoning effort. */
  reasoning: boolean
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function ModelEditSubmenu({
  fastControl,
  isActive,
  onActivate,
  onSelectModel,
  reasoning,
  requestGateway
}: ModelEditSubmenuProps) {
  // Reactive session state comes straight from the stores rather than being
  // drilled through the panel, so editing it re-renders only this submenu.
  const activeSessionId = useStore($activeSessionId)
  const currentReasoningEffort = useStore($currentReasoningEffort)

  const effort = normalizeEffort(currentReasoningEffort)
  const thinkingOn = isThinkingEnabled(currentReasoningEffort)

  // Reasoning/fast are session-scoped (they apply to the active model), so
  // editing a non-active model first switches to it. Returns false if the
  // switch failed, so callers skip applying to the wrong (previous) model.
  const ensureActive = async (): Promise<boolean> => {
    if (isActive) {
      return true
    }

    return (await onActivate()) !== false
  }

  const patchReasoning = async (next: string, rollback: string) => {
    setCurrentReasoningEffort(next)

    try {
      if (!(await ensureActive())) {
        setCurrentReasoningEffort(rollback)

        return
      }

      await requestGateway('config.set', {
        key: 'reasoning',
        session_id: activeSessionId ?? '',
        value: next
      })
    } catch (err) {
      setCurrentReasoningEffort(rollback)
      notifyError(err, 'Model option update failed')
    }
  }

  const toggleFast = (enabled: boolean) => {
    if (fastControl.kind === 'variant') {
      // Fast is a separate model id — swap to it (or back to the base).
      void onSelectModel(enabled ? fastControl.fastId : fastControl.baseId)

      return
    }

    if (fastControl.kind === 'param') {
      setCurrentFastMode(enabled)

      void (async () => {
        try {
          if (!(await ensureActive())) {
            setCurrentFastMode(!enabled)

            return
          }

          await requestGateway('config.set', {
            key: 'fast',
            session_id: activeSessionId ?? '',
            value: enabled ? 'fast' : 'normal'
          })
        } catch (err) {
          setCurrentFastMode(!enabled)
          notifyError(err, 'Fast mode update failed')
        }
      })()
    }
  }

  const hasFast = fastControl.kind !== 'none'
  const fastOn = fastControl.kind === 'none' ? false : fastControl.on

  return (
    <DropdownMenuSubContent className="w-52 p-0" sideOffset={4}>
      {!hasFast && !reasoning ? (
        <div className="px-2.5 py-3 text-xs text-(--ui-text-tertiary)">No options for this model</div>
      ) : (
        <>
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>Options</DropdownMenuLabel>
          {reasoning ? (
            <DropdownMenuItem
              className={dropdownMenuRow}
              onSelect={event => event.preventDefault()}
            >
              Thinking
              <Switch
                checked={thinkingOn}
                className="ml-auto"
                onCheckedChange={checked => void patchReasoning(checked ? effort || 'medium' : 'none', currentReasoningEffort)}
                size="xs"
              />
            </DropdownMenuItem>
          ) : null}
          {hasFast ? (
            <DropdownMenuItem
              className={dropdownMenuRow}
              onSelect={event => event.preventDefault()}
            >
              Fast
              <Switch checked={fastOn} className="ml-auto" onCheckedChange={toggleFast} size="xs" />
            </DropdownMenuItem>
          ) : null}
          {reasoning ? (
            <>
              <DropdownMenuSeparator className="mx-0" />
              <DropdownMenuLabel className={dropdownMenuSectionLabel}>Effort</DropdownMenuLabel>
              <DropdownMenuRadioGroup
                onValueChange={value => void patchReasoning(value, currentReasoningEffort)}
                value={effort}
              >
                {EFFORT_OPTIONS.map(option => (
                  <DropdownMenuRadioItem
                    className={dropdownMenuRow}
                    key={option.value}
                    onSelect={event => event.preventDefault()}
                    value={option.value}
                  >
                    {option.label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </>
          ) : null}
        </>
      )}
    </DropdownMenuSubContent>
  )
}

function isThinkingEnabled(effort: string): boolean {
  // Empty = Hermes default (medium) = on; only an explicit "none" is off.
  return (effort || 'medium').trim().toLowerCase() !== 'none'
}

function normalizeEffort(effort: string): string {
  const value = (effort || 'medium').trim().toLowerCase()

  // Thinking off → no effort selected in the radio group.
  if (value === 'none') {
    return ''
  }

  return EFFORT_OPTIONS.some(option => option.value === value) ? value : 'medium'
}
