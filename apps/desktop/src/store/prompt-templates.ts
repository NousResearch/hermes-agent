/**
 * User-customisable prompt templates — the "Prompt templates" entry in the
 * composer's "+" menu.  The three built-in starters (code review,
 * implementation plan, explain this) seed the store the first time the user
 * opens the dialog; from then on the list is entirely user-owned.  Add, edit,
 * delete, and re-order all persist to localStorage through the shared storage
 * choke point so cross-window sync / telemetry hooks see the writes.
 *
 * Storage shape: `PromptTemplate[]` (ordered array, not a record — order is
 * user-meaningful and the list is small enough that linear scans are fine).
 *
 * i18n: the built-in defaults are read lazily via `translateNow` so they
 * reflect the user's active locale at first-launch time.  The store itself
 * starts empty and is seeded by `ensureSeeded()` when the dialog opens — this
 * avoids the circular import (runtime.ts ↔ store) AND the "English seed on a
 * Chinese UI" problem that a module-level constant would cause.
 */

import { persistentAtom, Codecs } from '@/lib/persisted'
import { translateNow } from '@/i18n/runtime'

const STORAGE_KEY = 'hermes.desktop.prompt-templates'

export interface PromptTemplate {
  id: string
  label: string
  description: string
  text: string
}

/** Stable ids for the three built-in starters.  The user may edit or delete
 *  these — the ids only matter for the initial seed and for reset. */
export const BUILTIN_TEMPLATE_IDS = ['codeReview', 'implementationPlan', 'explainThis'] as const

/**
 * Build the three built-in starters using the active locale's translations.
 * Called lazily (not at module load) so the i18n runtime is initialised.
 * Returns fresh copies so callers can mutate without side effects.
 */
export function getBuiltInTemplates(): PromptTemplate[] {
  return BUILTIN_TEMPLATE_IDS.map(id => ({
    id,
    label: translateNow(`composer.templates.${id}.label`),
    description: translateNow(`composer.templates.${id}.description`),
    text: translateNow(`composer.templates.${id}.text`)
  }))
}

function isTemplate(value: unknown): value is PromptTemplate {
  if (!value || typeof value !== 'object') return false
  const s = value as Record<string, unknown>
  return (
    typeof s.id === 'string' &&
    typeof s.label === 'string' &&
    typeof s.description === 'string' &&
    typeof s.text === 'string'
  )
}

/** Sanitize untrusted persisted JSON: accept only a flat array of valid
 *  template objects; anything else (corrupt, missing) returns an empty array.
 *  The built-in seed is injected lazily by `ensureSeeded()` once the i18n
 *  runtime is ready, so we never seed English text on a non-English UI. */
function sanitizeTemplates(raw: unknown): PromptTemplate[] {
  if (Array.isArray(raw) && raw.every(isTemplate)) {
    return raw
  }

  return []
}

export const $promptTemplates = persistentAtom<PromptTemplate[]>(
  STORAGE_KEY,
  [],
  Codecs.json(sanitizeTemplates)
)

/** Seed the store with locale-appropriate built-in templates the first time
 *  the dialog is opened (or after a corrupted-payload reset).  If the user
 *  has already added/edited templates, this is a no-op. */
export function ensureSeeded(): void {
  if ($promptTemplates.get().length > 0) {
    return
  }

  $promptTemplates.set(getBuiltInTemplates())
}

/** Add a new template at the end of the list.  Returns the created template
 *  (with a generated id) so the UI can immediately drop into edit mode. */
export function addTemplate(label = '', description = '', text = ''): PromptTemplate {
  const template: PromptTemplate = {
    id: `tpl-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    label,
    description,
    text
  }

  $promptTemplates.set([...$promptTemplates.get(), template])

  return template
}

/** Patch a single template by id.  Unknown ids are ignored. */
export function updateTemplate(id: string, patch: Partial<Omit<PromptTemplate, 'id'>>): void {
  $promptTemplates.set(
    $promptTemplates.get().map(s => (s.id === id ? { ...s, ...patch } : s))
  )
}

/** Remove a template by id.  Unknown ids are ignored. */
export function deleteTemplate(id: string): void {
  $promptTemplates.set($promptTemplates.get().filter(s => s.id !== id))
}

/** Move a template one slot up (toward index 0).  No-op at the top. */
export function moveTemplateUp(id: string): void {
  const list = [...$promptTemplates.get()]
  const index = list.findIndex(s => s.id === id)

  if (index <= 0) return

  ;[list[index - 1], list[index]] = [list[index], list[index - 1]]
  $promptTemplates.set(list)
}

/** Move a template one slot down (toward the end).  No-op at the bottom. */
export function moveTemplateDown(id: string): void {
  const list = [...$promptTemplates.get()]
  const index = list.findIndex(s => s.id === id)

  if (index < 0 || index >= list.length - 1) return

  ;[list[index], list[index + 1]] = [list[index + 1], list[index]]
  $promptTemplates.set(list)
}

/** Restore the three built-in starters, discarding everything the user added.
 *  Existing built-in ids are replaced; any user-added templates are removed. */
export function resetToBuiltins(): void {
  $promptTemplates.set(getBuiltInTemplates())
}
