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
import { isThinkingEnabled, REASONING_EFFORTS, resolveReasoningEffort } from '@/lib/reasoning-effort'

// Hermes' real reasoning levels live in lib/reasoning-effort; `none` is owned
// by the Thinking toggle, not the radio.

/** How "fast" is achieved for a given model — two different mechanisms:
 *  - `param`: the Anthropic/OpenAI `speed=fast` request parameter.
 *  - `variant`: a separate `…-fast` sibling model selected via the model field.
 */
export type FastControl =
  { kind: 'none' } | { kind: 'param'; on: boolean } | { kind: 'variant'; baseId: string; fastId: string; on: boolean }

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
  /** Whether this model can turn thinking off. False on reasoning-mandatory
   *  routes, whose upstream rejects a disable — the toggle is hidden rather
   *  than offered as a control that silently does nothing. */
  canDisableReasoning?: boolean
  /** The profile's configured default effort — what an unset row inherits.
   *  Passed in (not read from a store) so this submenu stays pure. */
  defaultEffort: string
  /** This row's effective reasoning effort (live for the active model, else its
   *  preset) — the submenu shows and edits from this, never the raw session. */
  effort: string
  /** How fast mode is offered for this model (param toggle vs. variant swap). */
  fastControl: FastControl
  /** Whether this row's model is the active one. */
  isActive: boolean
  /** This row's model id. */
  model: string
  /** Switch to a specific model id (used to swap base ⇄ -fast variant). */
  onSelectModel: (model: string) => Promise<boolean | void> | void
  /** Report an option change. This submenu is PURE: it never writes to a
   *  session, a preset store, or the gateway itself — the owning surface's
   *  controller decides what an edit means. That's what lets the same submenu
   *  drive a live chat session and a detached per-task override. */
  onSetOptions: (patch: { effort?: string; fast?: boolean }) => void
  /** This row's provider slug. */
  provider: string
  /** Whether this model supports reasoning effort. */
  reasoning: boolean
}

export function ModelEditSubmenu(props: ModelEditSubmenuProps) {
  // The panel mounts one of these per model row; only the hovered row's
  // submenu is ever open. Keep this wrapper hook-free and render the body as
  // a CHILD of SubContent so Radix's Presence gate leaves it unrendered until
  // the sub actually opens — eagerly running the body's hooks/JSX for every
  // row made opening the menu itself lag on large catalogs.
  return (
    <DropdownMenuSubContent className="w-52 p-0" sideOffset={4}>
      <ModelEditSubmenuBody {...props} />
    </DropdownMenuSubContent>
  )
}

function ModelEditSubmenuBody({
  canDisableReasoning,
  defaultEffort,
  effort,
  fastControl,
  isActive,
  model,
  onSelectModel,
  onSetOptions,
  reasoning
}: ModelEditSubmenuProps) {
  const { t } = useI18n()
  const copy = t.shell.modelOptions

  const effortValue = resolveReasoningEffort(effort, defaultEffort)
  const thinkingOn = isThinkingEnabled(effort, defaultEffort)
  const showThinkingToggle = reasoning && canDisableReasoning !== false

  const setFast = (enabled: boolean) => {
    if (fastControl.kind === 'variant') {
      // Fast is a separate model id. Report the choice so the controller can
      // record it against the base model, and only swap models now if this is
      // the active row — inactive edits stay preference-only.
      onSetOptions({ fast: enabled })

      if (isActive) {
        void onSelectModel(enabled ? fastControl.fastId : fastControl.baseId)
      }

      return
    }

    if (fastControl.kind === 'param') {
      onSetOptions({ fast: enabled })
    }
  }

  const hasFast = fastControl.kind !== 'none'
  const fastOn = fastControl.kind === 'none' ? false : fastControl.on

  return (
    <>
      <div className="truncate px-2.5 pb-1 pt-2 text-xs text-foreground" title={model}>
        {model}
      </div>
      {!hasFast && !reasoning ? (
        <div className="px-2.5 py-3 text-xs text-(--ui-text-tertiary)">{copy.noOptions}</div>
      ) : (
        <>
          <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.options}</DropdownMenuLabel>
          {showThinkingToggle ? (
            <DropdownMenuItem className={dropdownMenuRow} onSelect={event => event.preventDefault()}>
              {copy.thinking}
              <Switch
                checked={thinkingOn}
                className="ml-auto"
                onCheckedChange={checked => onSetOptions({ effort: checked ? effortValue || defaultEffort : 'none' })}
                size="xs"
              />
            </DropdownMenuItem>
          ) : null}
          {hasFast ? (
            <DropdownMenuItem className={dropdownMenuRow} onSelect={event => event.preventDefault()}>
              {copy.fast}
              <Switch checked={fastOn} className="ml-auto" onCheckedChange={setFast} size="xs" />
            </DropdownMenuItem>
          ) : null}
          {reasoning ? (
            <>
              <DropdownMenuSeparator className="mx-0" />
              <DropdownMenuLabel className={dropdownMenuSectionLabel}>{copy.effort}</DropdownMenuLabel>
              <DropdownMenuRadioGroup onValueChange={value => onSetOptions({ effort: value })} value={effortValue}>
                {REASONING_EFFORTS.map(value => (
                  <DropdownMenuRadioItem
                    className={dropdownMenuRow}
                    key={value}
                    onSelect={event => event.preventDefault()}
                    value={value}
                  >
                    {copy[value]}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </>
          ) : null}
        </>
      )}
    </>
  )
}
