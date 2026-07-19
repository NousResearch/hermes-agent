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

interface ModelDisplayOptions {
  preserveProviderPrefix?: boolean
}

/** Split a model id into a clean display name plus an optional grayed variant
 *  tag, so distinct ids (e.g. `…-4.8` vs `…-4.8-fast`) don't collapse. */
export function modelDisplayParts(model: string, options?: ModelDisplayOptions): { name: string; tag: string } {
  const trimmed = model.trim()
  let base = options?.preserveProviderPrefix ? trimmed : modelBaseId(model)
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

  if (options?.preserveProviderPrefix) {
    return { name: base || trimmed || 'No model', tag }
  }

  return { name: prettifyBase(base) || trimmed || 'No model', tag }
}

/** Friendly labels intentionally hide provider prefixes. Return only labels
 *  shared by distinct ids so picker rows can preserve the prefix selectively. */
export function ambiguousModelDisplayNames(models: readonly string[]): Set<string> {
  const idsByName = new Map<string, Set<string>>()

  for (const model of models) {
    const id = model.trim()
    const name = modelDisplayParts(id).name
    const ids = idsByName.get(name) ?? new Set<string>()
    ids.add(id)
    idsByName.set(name, ids)
  }

  return new Set([...idsByName].filter(([, ids]) => ids.size > 1).map(([name]) => name))
}

/** Friendly one-line model name for menus and the status bar. */
export function displayModelName(model: string, options?: ModelDisplayOptions): string {
  return modelDisplayParts(model, options).name
}

/** Status bar trigger label — model name plus the live session state (effort/fast). */
export function formatModelStatusLabel(
  model: string,
  options?: { fastMode?: boolean; reasoningEffort?: string }
): string {
  const name = displayModelName(model)

  if (!model.trim()) {
    return name
  }

  const parts: string[] = []

  // Fast is shown when the speed=fast param is on (options.fastMode) OR the
  // active model is a `…-fast` variant (fast via a separate model id).
  if (options?.fastMode || /-fast$/i.test(modelBaseId(model))) {
    parts.push('Fast')
  }

  // Always surface the effort (empty = Hermes default of medium) so the
  // current reasoning level is visible at a glance, not just when non-default.
  parts.push(reasoningEffortLabel(options?.reasoningEffort ?? '') || 'Med')

  return `${name} · ${parts.join(' ')}`
}
