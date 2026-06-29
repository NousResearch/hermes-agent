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
import { useI18n } from '@/i18n'
import { setModelPreset } from '@/store/model-presets'
import { notifyError } from '@/store/notifications'
import { $activeSessionId, setCurrentFastMode, setCurrentReasoningEffort } from '@/store/session'

// Hermes' real reasoning levels (see VALID_REASONING_EFFORTS); `none` is owned
// by the Thinking toggle, not the radio.
const EFFORT_OPTIONS = [
  { value: 'minimal', labelKey: 'minimal' },
  { value: 'low', labelKey: 'low' },
  { value: 'medium', labelKey: 'medium' },
  { value: 'high', labelKey: 'high' },
  { value: 'xhigh', labelKey: 'max' }
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
    return providerModels.includes(baseId) ? { kind: 'variant', baseId, fastId: model, on: true } : { kind: 'none' }
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
  /** This row's effective reasoning effort (live for the active model, else its
   *  preset) — the submenu shows and edits from this, never the raw session. */
  effort: string
  /** How fast mode is offered for this model (param toggle vs. variant swap). */
  fastControl: FastControl
  /** Whether this row's model is the active one. */
  isActive: boolean
  /** This row's model id — edits persist as its global preset. */
  model: string
  /** Switch to a specific model id (used to swap base ⇄ -fast variant). */
  onSelectModel: (model: string) => Promise<boolean> | void
  /** This row's provider slug — edits persist as its global preset. */
  provider: string
  /** Whether this model supports reasoning effort. */
  reasoning: boolean
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function ModelEditSubmenu({
  effort,
  fastControl,
  isActive,
  model,
  onSelectModel,
  provider,
  reasoning,
  requestGateway
}: ModelEditSubmenuProps) {
  const { t } = useI18n()
  const copy = t.shell.modelOptions
  const activeSessionId = useStore($activeSessionId)

  const effortValue = normalizeEffort(effort)
  const thinkingOn = isThinkingEnabled(effort)

  // Editing always records the model's global preset; the active model also gets
  // it pushed onto the live session. Non-active edits stay preset-only — they do
  // not switch you to that model.
  const patchReasoning = async (next: string) => {
    setModelPreset(provider, model, { effort: next })

    if (!isActive) {
      return
    }

    setCurrentReasoningEffort(next)

    // Preset-only without a session: `isActive` holds for the global/default
    // row pre-session, and the gateway's `config.set` falls back to global
    // config when none matches — so don't reach it (preset + optimistic store
    // are the whole effect). Same guard in applyModelPreset / toggleFast.
    if (!activeSessionId) {
      return
    }

    try {
      await requestGateway('config.set', { key: 'reasoning', session_id: activeSessionId, value: next })
    } catch (err) {
      setCurrentReasoningEffort(effort)
      setModelPreset(provider, model, { effort })
      notifyError(err, copy.updateFailed)
    }
  }

  const toggleFast = (enabled: boolean) => {
    if (fastControl.kind === 'variant') {
      // Fast is a separate model id. Record the choice on the base model's
      // preset (selectFamily picks the `-fast` sibling later when set), and
      // only swap models now if this is the active row — inactive edits must
      // stay preset-only, same as the param path below.
      setModelPreset(provider, fastControl.baseId, { fast: enabled })

      if (isActive) {
        void onSelectModel(enabled ? fastControl.fastId : fastControl.baseId)
      }

      return
    }

    if (fastControl.kind === 'param') {
      setModelPreset(provider, model, { fast: enabled })

      if (!isActive) {
        return
      }

      setCurrentFastMode(enabled)

      // Preset-only without a session (see patchReasoning).
      if (!activeSessionId) {
        return
      }
      void (async () => {
        try {
          await requestGateway('config.set', {
            key: 'fast',
            session_id: activeSessionId,
            value: enabled ? 'fast' : 'normal'
          })
        } catch (err) {
          setCurrentFastMode(!enabled)
          setModelPreset(provider, model, { fast: !enabled })
          notifyError(err, copy.fastFailed)
        }
      })()
    }
  }

  const hasFast = fastControl.kind !== 'none'
  const fastOn = fastControl.kind === 'none' ? false : fastControl.on

  return (
    <DropdownMenuSubContent className="w-52 p-0" sideOffset={4}>
      {!hasFast && !reasoning ? (
        <div className="px-2.5 py-3 text-xs text-(--ui-text-tertiary)">{copy.noOptions}</div>
      ) : (
        <>
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.options}</DropdownMenuLabel>
          {reasoning ? (
            <DropdownMenuItem className={dropdownMenuRow} onSelect={event => event.preventDefault()}>
              {copy.thinking}
              <Switch
                checked={thinkingOn}
                className="ml-auto"
                onCheckedChange={checked => void patchReasoning(checked ? effortValue || 'medium' : 'none')}
                size="xs"
              />
            </DropdownMenuItem>
          ) : null}
          {hasFast ? (
            <DropdownMenuItem className={dropdownMenuRow} onSelect={event => event.preventDefault()}>
              {copy.fast}
              <Switch checked={fastOn} className="ml-auto" onCheckedChange={toggleFast} size="xs" />
            </DropdownMenuItem>
          ) : null}
          {reasoning ? (
            <>
              <DropdownMenuSeparator className="mx-0" />
              <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.effort}</DropdownMenuLabel>
              <DropdownMenuRadioGroup onValueChange={value => void patchReasoning(value)} value={effortValue}>
                {EFFORT_OPTIONS.map(option => (
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
