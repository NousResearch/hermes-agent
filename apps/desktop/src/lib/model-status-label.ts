import { normalize } from '@/lib/text'

const REASONING_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'Max'
}

interface PickerModelProvider {
  models?: string[]
  name?: string
  slug: string
}

interface PickerModelOptions {
  model?: string
  provider?: string
  providers?: PickerModelProvider[]
}

export function reasoningEffortLabel(effort: string): string {
  const key = normalize(effort)

  if (!key) {
    return ''
  }

  return REASONING_LABELS[key] ?? effort
}

function providerHasModel(options: PickerModelOptions | undefined, provider: string, model: string): boolean {
  const row = options?.providers?.find(candidate => candidate.slug === provider)

  return Boolean(row?.models?.includes(model))
}

function providerOwningModel(options: PickerModelOptions | undefined, model: string): string {
  if (!model) {
    return ''
  }

  return options?.providers?.find(provider => provider.models?.includes(model))?.slug ?? ''
}

export function reconcilePickerSelection(
  selection: { model: string; provider: string },
  options?: PickerModelOptions
): { model: string; provider: string } {
  const model = String(selection.model || '')
  const provider = String(selection.provider || '')

  if (!model || !provider || !options?.providers?.length || providerHasModel(options, provider, model)) {
    return { model, provider }
  }

  const owner = providerOwningModel(options, model)

  return owner ? { model, provider: owner } : { model, provider }
}

/** Which model/provider a picker should mark "current". With a live session the
 *  gateway's `model.options` is authoritative; pre-session there is no server
 *  "current", so the sticky composer pick wins over the profile default the
 *  global options query returns — else the checkmark snaps back to the default
 *  and the pick looks ignored. */
export function currentPickerSelection(
  hasSession: boolean,
  store: { model: string; provider: string },
  options?: PickerModelOptions
): { model: string; provider: string } {
  return reconcilePickerSelection(
    {
      model: String((hasSession && options?.model) || store.model || options?.model || ''),
      provider: String((hasSession && options?.provider) || store.provider || options?.provider || '')
    },
    options
  )
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
