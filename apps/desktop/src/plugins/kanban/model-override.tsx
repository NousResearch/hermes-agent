/**
 * Per-task model override — the board's half of the composer's model picker.
 *
 * The MENU is not ours: `ModelCatalogMenu` from the SDK is the same component
 * the chat composer renders, so search, provider grouping, `-fast` families,
 * and the thinking/effort submenu behave identically here. What differs is
 * what a selection MEANS — the controller below holds a detached value instead
 * of writing to a live session, which is exactly the seam the SDK exposes.
 *
 * An unset value means "inherit the assigned profile's own model + effort",
 * which is what every kanban task did before this existed.
 */

import {
  Button,
  cn,
  Codicon,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  ModelCatalogMenu,
  ModelMenuCloseContext,
  type ModelMenuController,
  reasoningEffortLabel
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { useKanban } from './ui'

/** A task's model override. Empty strings mean "inherit the profile". */
export interface TaskModelOverride {
  /** '' = profile default, 'none' = thinking off, else a reasoning level. */
  effort: string
  model: string
  provider: string
}

export const EMPTY_OVERRIDE: TaskModelOverride = { effort: '', model: '', provider: '' }

/** True when nothing is pinned and the worker profile decides everything. */
export const isInherited = (value: TaskModelOverride): boolean =>
  !value.model.trim() && !value.provider.trim() && !value.effort.trim()

/** The trigger's label: `provider: model · High`, or the inherit copy. */
export function overrideLabel(value: TaskModelOverride, inheritCopy: string): string {
  if (isInherited(value)) {
    return inheritCopy
  }

  const model = value.model.trim()
  const base = model ? (value.provider.trim() ? `${value.provider}: ${model}` : model) : inheritCopy
  const effort = value.effort.trim() ? reasoningEffortLabel(value.effort) : ''

  return effort ? `${base} · ${effort}` : base
}

/**
 * The picker itself. Controlled: the caller owns the value, so the New Task
 * dialog can hold it as form state and the drawer can PATCH on change.
 */
export function ModelOverrideField({
  onChange,
  value
}: {
  onChange: (next: TaskModelOverride) => void
  value: TaskModelOverride
}) {
  const k = useKanban()
  const [open, setOpen] = useState(false)

  const controller: ModelMenuController = {
    // Picking a model seeds the depth from what the user last used for it, so
    // the board behaves like the composer. We only READ presets — a per-task
    // choice must never rewrite what the composer opens at.
    applyPreset: (preset, row) =>
      onChange({
        effort: preset.effort ?? '',
        model: row.model,
        provider: row.provider
      }),

    current: { effort: value.effort, fast: false, model: value.model, provider: value.provider },

    // Read-only against Hermes' global presets is deliberate: see applyPreset.
    presetFor: () => ({}),

    select: (model, provider) => {
      onChange({ ...value, model, provider })
    },

    // Fast mode is a live-session request parameter, not something the worker
    // spawn can carry (there is no --fast flag), so the menu is told this
    // catalog has no fast support and only effort edits arrive here.
    setOptions: (patch, row) => {
      if (patch.effort === undefined) {
        return
      }

      onChange({ effort: patch.effort, model: row.model, provider: row.provider })
    }
  }

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>
        <Button
          className={cn(
            'h-8 w-full justify-between gap-2 px-2.5 text-[0.75rem] font-normal',
            isInherited(value) && 'text-(--ui-text-tertiary)'
          )}
          type="button"
          variant="outline"
        >
          <span className="min-w-0 truncate">{overrideLabel(value, k.modelInherit)}</span>
          <span className="flex shrink-0 items-center gap-1">
            {!isInherited(value) && (
              <span
                aria-label={k.modelClear}
                className="grid size-4 place-items-center rounded text-(--ui-text-tertiary) hover:text-foreground"
                onClick={event => {
                  // Clear without opening the menu.
                  event.preventDefault()
                  event.stopPropagation()
                  onChange(EMPTY_OVERRIDE)
                }}
                role="button"
              >
                <Codicon name="close" size="0.7rem" />
              </span>
            )}
            <Codicon className="opacity-50" name="chevron-down" size="0.7rem" />
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72 p-0">
        <ModelMenuCloseContext.Provider value={() => setOpen(false)}>
          <ModelCatalogMenu controller={controller} />
        </ModelMenuCloseContext.Provider>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Translate an override into the REST fields the kanban API expects.
 *  Used by both create (POST) and edit (PATCH); the explicit clear flags exist
 *  because `null` in a PATCH body means "field not sent", not "set to NULL". */
export function overridePatch(value: TaskModelOverride): Record<string, unknown> {
  const model = value.model.trim()
  const effort = value.effort.trim()

  return {
    ...(model
      ? { model_override: model, provider_override: value.provider.trim() || undefined }
      : { clear_model_override: true }),
    ...(effort ? { reasoning_effort: effort } : { clear_reasoning_effort: true })
  }
}

/** The create-time shape: omit rather than clear, so the backend's own
 *  defaults apply to fields the user never touched. */
export function overrideCreateFields(value: TaskModelOverride): Record<string, unknown> {
  const model = value.model.trim()
  const effort = value.effort.trim()

  return {
    ...(model ? { model_override: model } : {}),
    ...(model && value.provider.trim() ? { provider_override: value.provider.trim() } : {}),
    ...(effort ? { reasoning_effort: effort } : {})
  }
}
