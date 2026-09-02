import { DEFAULT_REASONING_EFFORT, reasoningEffortLabel } from './reasoning-effort.ts'

/** Which model/provider pair a picker should mark "current". SessionView state
 *  also drives the composer label, so a complete pair there wins over an older
 *  `model.options` response. During initial hydration (or pre-session startup),
 *  options remain the fallback. Pick one complete pair before mixing fields so
 *  a model is never shown under a different provider. */
export function currentPickerSelection(
  store: { model: string; provider: string },
  options?: { model?: string; provider?: string }
): { model: string; provider: string } {
  const storeSelection = {
    model: String(store.model || ''),
    provider: String(store.provider || '')
  }

  const optionsSelection = {
    model: String(options?.model || ''),
    provider: String(options?.provider || '')
  }

  if (storeSelection.model && storeSelection.provider) {
    return storeSelection
  }

  if (optionsSelection.model && optionsSelection.provider) {
    return optionsSelection
  }

  return {
    model: storeSelection.model || optionsSelection.model,
    provider: storeSelection.provider || optionsSelection.provider
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

  // Local GGUF ids carry a quant suffix (`…-UD-Q4_K_XL`, `…-Q8_0`). Render it
  // as a quiet tag — "Qwen3.6 27B · Q4" — never as part of the name. Without
  // this the composer pill reads raw quant soup ("Qwen3.6 27B UD Q4 K XL").
  const quant = base.match(/-(?:UD-)?(Q\d(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)*|F16|BF16)$/i)

  if (quant) {
    tag = quant[1].split('_')[0].toUpperCase()
    base = base.slice(0, -quant[0].length)
    // Instruct/chat markers are noise once the quant confirmed a local build.
    base = base.replace(/-(?:Instruct|Chat)(?:-\d{4})?$/i, '')
  }

  if (!tag) {
    for (const [pattern, label] of VARIANT_TAGS) {
      if (pattern.test(base)) {
        tag = label
        base = base.replace(pattern, '')

        break
      }
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

/** Provider prefix of a model id (`openai` from `openai/gpt-5.5`), '' when the
 *  id is bare or the slash is malformed. */
function providerPrefixOf(model: string): string {
  const trimmed = model.trim()
  const slash = trimmed.indexOf('/')

  return slash > 0 ? trimmed.slice(0, slash) : ''
}

/** Status-bar trigger label split into renderable parts so a host can style
 *  segments differently (the composer pill dims the provider). `provider` is
 *  '' when unknown or when the model id already carries it as its `provider/`
 *  prefix — a duplicate "openai · openai" reads as a rendering bug. */
export function modelStatusLabelParts(
  model: string,
  options?: { defaultEffort?: string; fastMode?: boolean; provider?: string; reasoningEffort?: string }
): { meta: string; name: string; provider: string } {
  const name = displayModelName(model)

  if (!model.trim()) {
    return { meta: '', name, provider: '' }
  }

  const parts: string[] = []

  // Fast is shown when the speed=fast param is on (options.fastMode) OR the
  // active model is a `…-fast` variant (fast via a separate model id).
  if (options?.fastMode || /-fast$/i.test(modelBaseId(model))) {
    parts.push('Fast')
  }

  // Always surface the effort so the current reasoning level is visible at a
  // glance, not just when non-default.
  parts.push(reasoningEffortLabel(options?.reasoningEffort || options?.defaultEffort || DEFAULT_REASONING_EFFORT))

  const slug = options?.provider?.trim() ?? ''
  const trimmedModel = model.trim()
  const idPrefix = providerPrefixOf(trimmedModel)
  // Only a bare `provider/model` id makes the row's provider redundant with
  // the id's own prefix (e.g. `openai/gpt-5.5` + `openai` → hide). Relay-style
  // ids (`openrouter/anthropic/claude-opus-4.8` + `openrouter`) carry an
  // upstream vendor in the deeper path: the top-level prefix names the relay,
  // so both halves are information and both show.
  const relayPath = idPrefix !== '' && trimmedModel.slice(idPrefix.length + 1).includes('/')

  return {
    meta: parts.join(' '),
    name,
    provider:
      slug && (slug.toLowerCase() !== idPrefix.toLowerCase() || relayPath) ? slug : ''
  }
}

/** Status bar trigger label — model name plus the live session state (effort/fast).
 *  `defaultEffort` is the profile's configured level, used when the surface has
 *  no explicit effort so the label never advertises a default the agent won't use.
 *  `provider` is shown between name and state when it adds information (it is
 *  skipped when the model id already carries the same provider prefix). */
export function formatModelStatusLabel(
  model: string,
  options?: { defaultEffort?: string; fastMode?: boolean; provider?: string; reasoningEffort?: string }
): string {
  const parts = modelStatusLabelParts(model, options)
  const segments = [parts.name]

  if (parts.provider) {
    segments.push(parts.provider)
  }

  if (parts.meta) {
    segments.push(parts.meta)
  }

  return segments.join(' · ')
}
