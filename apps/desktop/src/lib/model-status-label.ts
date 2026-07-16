import { normalize } from '@/lib/text'

const REASONING_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
  ultra: 'Ultra'
}

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  if (!key) {
    return ''
  }

  return REASONING_LABELS[key] ?? effort
}

// Hermes' real reasoning levels (see VALID_REASONING_EFFORTS + the
// `/reasoning` allow-list in `apps/desktop/src/app/settings/constants.ts`);
// `none` is owned by the Thinking toggle, not the radio. Shared by the
// composer reasoning pill and the per-row model picker submenu so the two
// menus cannot drift on which levels exist.
export const REASONING_EFFORT_OPTIONS = [
  { value: 'minimal', labelKey: 'minimal' },
  { value: 'low', labelKey: 'low' },
  { value: 'medium', labelKey: 'medium' },
  { value: 'high', labelKey: 'high' },
  { value: 'xhigh', labelKey: 'xhigh' },
  { value: 'max', labelKey: 'max' },
  { value: 'ultra', labelKey: 'ultra' }
] as const

// Hermes' default effort when none is set (or the value is unknown). Used by
// the Thinking-off → restore-on toggle, normalizeReasoningEffort, and the
// pill label fallback.
export const DEFAULT_REASONING_EFFORT = 'medium'

/** Empty = Hermes default (medium) = on; only an explicit "none" is off. */
export function isThinkingEnabled(effort: string): boolean {
  return (effort || DEFAULT_REASONING_EFFORT).trim().toLowerCase() !== 'none'
}

/** Normalize an effort string for the radio group: 'none' → '' (off), unknown
 *  → 'medium' (Hermes default). */
export function normalizeReasoningEffort(effort: string): string {
  const value = (effort || DEFAULT_REASONING_EFFORT).trim().toLowerCase()

  if (value === 'none') {
    return ''
  }

  return REASONING_EFFORT_OPTIONS.some(option => option.value === value) ? value : DEFAULT_REASONING_EFFORT
}

/** Which model/provider a picker should mark "current". With a live session the
 *  gateway's `model.options` is authoritative; pre-session there is no server
 *  "current", so the sticky composer pick wins over the profile default the
 *  global options query returns — else the checkmark snaps back to the default
 *  and the pick looks ignored. */
export function currentPickerSelection(
  hasSession: boolean,
  store: { model: string; provider: string },
  options?: { model?: string; provider?: string }
): { model: string; provider: string } {
  return {
    model: String((hasSession && options?.model) || store.model || options?.model || ''),
    provider: String((hasSession && options?.provider) || store.provider || options?.provider || '')
  }
}

/** Strip provider prefix and normalize for display. */
export function modelBaseId(model: string): string {
  const trimmed = model.trim()
  const slash = trimmed.lastIndexOf('/')

  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed
}

// Trailing model-id variants that should render as a grayed tag beside the
// name (e.g. "Opus 4.8" + "Fast") rather than collapsing two distinct ids to
// the same display name.
const VARIANT_TAGS: ReadonlyArray<readonly [RegExp, string]> = [
  [/-fast$/i, 'Fast'],
  [/-thinking$/i, 'Thinking'],
  [/-preview$/i, 'Preview'],
  [/-latest$/i, 'Latest']
]

const titleCase = (text: string): string => text.replace(/\b\w/g, char => char.toUpperCase()).trim()

function prettifyBase(base: string): string {
  if (/^claude-/i.test(base)) {
    return titleCase(base.replace(/^claude-/i, '').replace(/-/g, ' '))
  }

  if (/^gpt-/i.test(base)) {
    return base.replace(/^gpt-/i, 'GPT-')
  }

  if (/^gemini-/i.test(base)) {
    return base.replace(/^gemini-/i, 'Gemini ').replace(/-/g, ' ')
  }

  return titleCase(base.replace(/-/g, ' '))
}

/** Split a model id into a clean display name plus an optional grayed variant
 *  tag, so distinct ids (e.g. `…-4.8` vs `…-4.8-fast`) don't collapse. */
export function modelDisplayParts(model: string): { name: string; tag: string } {
  let base = modelBaseId(model)
  let tag = ''

  for (const [pattern, label] of VARIANT_TAGS) {
    if (pattern.test(base)) {
      tag = label
      base = base.replace(pattern, '')

      break
    }
  }

  // Drop a trailing date-pin (`…-20251101`) — snapshot noise, not a name.
  base = base.replace(/-\d{8}$/, '')

  return { name: prettifyBase(base) || model.trim() || 'No model', tag }
}

/** Friendly one-line model name for menus and the status bar. */
export function displayModelName(model: string): string {
  return modelDisplayParts(model).name
}

/** Composer model pill label — the model display name plus a `Fast` suffix
 *  when the active variant / speed param is in fast mode. Effort is rendered
 *  by its own pill (see ReasoningPill) so a 10rem pill no longer has to fit
 *  both at once and the effort isn't truncated on long model names.
 *
 *  Returns just the placeholder name when the model is empty. */
export function formatModelPillLabel(model: string, options?: { fastMode?: boolean }): string {
  const name = displayModelName(model)

  if (!model.trim()) {
    return name
  }

  if (options?.fastMode || /-fast$/i.test(modelBaseId(model))) {
    return `${name} · Fast`
  }

  return name
}

/** Composer reasoning-effort pill label. Empty effort (Hermes default) renders
 *  as `Med` so the active level is visible at a glance. `none` renders as `Off`
 *  to make the disabled-thinking state explicit. */
export function formatReasoningPillLabel(effort: string): string {
  return reasoningEffortLabel(effort) || 'Med'
}
