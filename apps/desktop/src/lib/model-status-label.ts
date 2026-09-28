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

/** Canonical provider labels shared by onboarding and the model pill. OAuth
 *  provider ids stay distinct from their direct-API counterparts so a session on
 *  `xai-oauth` never reads as the plain `xai` key path, and internal route names
 *  never reach user-facing copy. */
export const PROVIDER_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  anthropic: 'Anthropic Account',
  'claude-code': 'Anthropic OAuth: Required Extra Usage Credits to Use Subscription',
  'minimax-oauth': 'MiniMax',
  nous: 'Nous Portal',
  'openai-codex': 'ChatGPT or Codex Subscription',
  'qwen-oauth': 'Qwen Code',
  xai: 'xAI',
  'xai-oauth': 'xAI Grok'
}

export function providerDisplayName(provider: string): string {
  const normalized = provider.trim().toLowerCase()

  return PROVIDER_DISPLAY_NAMES[normalized] ?? provider.trim()
}

/** Strip provider prefix and normalize for display. */
export function modelBaseId(model: string): string {
  const trimmed = model.trim()
  const slash = trimmed.lastIndexOf('/')

  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed
}

// Trailing model-id variants that should render as a grayed tag beside the
// name (e.g. "Opus 4.8" + "Fast") rather than collapsing two distinct ids to
// the same display name. `-flash` splits look-alike pairs the same way
// (#118083): models.dev carries both `deepseek-flash` (alias) and
// `deepseek-v4.1-flash` (full id) for the provider.
const VARIANT_TAGS: ReadonlyArray<readonly [RegExp, string]> = [
  [/-fast$/i, 'Fast'],
  [/-flash$/i, 'Flash'],
  [/-thinking$/i, 'Thinking'],
  [/-preview$/i, 'Preview'],
  [/-latest$/i, 'Latest']
]

const titleCase = (text: string): string => text.replace(/\b\w/g, char => char.toUpperCase()).trim()

// Vendors write their own names in casing the model id does not carry, and
// title-casing the id overrides it: `glm-5.2` reads as "Glm 5.2" instead of
// "GLM 5.2" (#85849). Applied AFTER title-casing so the rule is one pass over
// a normalized string, and only ever to whole words — `Minimax` never touches
// a longer token that merely contains it.
const VENDOR_CASING: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bDeepseek\b/g, 'DeepSeek'],
  [/\bGlm\b/g, 'GLM'],
  [/\bMinimax\b/g, 'MiniMax'],
  [/\bOpenai\b/g, 'OpenAI'],
  [/\bErnie\b/g, 'ERNIE'],
  [/\bMimo\b/g, 'MiMo'],
  [/\bBge\b/g, 'BGE'],
  [/\bVl\b/g, 'VL'],
  [/\bIt\b/g, 'IT'],
  [/\bFp8\b/g, 'FP8'],
  [/\bAi\b/g, 'AI']
]

// Parameter counts and active-parameter counts: vendors write 8B, 235B, A22B —
// never 8b. Matched after title-casing (so the token reads "8b" or "A3b"),
// case-insensitively so the title-cased "A" of "A3b" is still a prefix.
const PARAMETER_COUNT = /\b(a?)(\d+(?:\.\d+)?)b\b/gi

const applyVendorCasing = (text: string): string => {
  let cased = text.replace(PARAMETER_COUNT, (_match, prefix: string, size: string) => `${prefix.toUpperCase()}${size}B`)

  for (const [pattern, replacement] of VENDOR_CASING) {
    cased = cased.replace(pattern, replacement)
  }

  return cased
}

function prettifyBase(base: string): string {
  if (/^claude-/i.test(base)) {
    // Anthropic ids spell the version with hyphens (`haiku-4-5`, `fable-5-1`);
    // the human name is dotted ("Haiku 4.5"), not "Haiku 4 5".
    return applyVendorCasing(
      titleCase(
        base
          .replace(/^claude-/i, '')
          .replace(/(\d)-(?=\d)/g, '$1.')
          .replace(/-/g, ' ')
      )
    )
  }

  if (/^gpt-/i.test(base)) {
    return base.replace(/^gpt-/i, 'GPT-')
  }

  // Title-case this branch too: without it `gemini-2.5-pro` rendered as
  // "Gemini 2.5 pro" — the only branch that left its words lowercase.
  if (/^gemini-/i.test(base)) {
    return applyVendorCasing(titleCase(base.replace(/^gemini-/i, 'Gemini ').replace(/-/g, ' ')))
  }

  return applyVendorCasing(titleCase(base.replace(/-/g, ' ')))
}

// Snapshot date pins vendors append to otherwise-stable ids — `…-20251101`
// and `…-2026-05-17` are the same model on a different day, not a different
// name. Both vendor spellings are recognized, with month/day bounds so a
// numeric-looking suffix (e.g. `-1234-56-78`) is never eaten as a "date".
const DATE_PIN = /-(?:\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])|\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))$/

// Split the trailing suffixes a local id can carry — a variant tag
// (`…-flash`, `…-fast`) and a GGUF quant (`…-UD-Q4_K_XL`, `…-Q8_0`) — in
// EITHER order: `…-flash-Q4_K_XL` and `…-Q4_K_XL-flash` are the same model.
// One decomposition feeds both the catalog rows and the composer pill, so
// the two screens can never disagree on which variant an id carries.
// A date pin can sit under the suffixes (`…-thinking-20251101`), so it is
// peeled in the same loop — otherwise the pin hides the variant from every
// surface that reads it.
function splitTrailingTags(base: string): { base: string; variant: string; quant: string } {
  let variant = ''
  let quant = ''

  for (let progress = true; progress;) {
    progress = false

    if (!variant) {
      for (const [pattern, label] of VARIANT_TAGS) {
        if (pattern.test(base)) {
          variant = label
          base = base.replace(pattern, '')
          progress = true

          break
        }
      }
    }

    if (!quant) {
      const quantMatch = base.match(/-(?:UD-)?(Q\d(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)*|F16|BF16)$/i)

      if (quantMatch) {
        quant = quantMatch[1].split('_')[0].toUpperCase()
        base = base.slice(0, -quantMatch[0].length)
        // Instruct/chat markers are noise once the quant confirmed a local build.
        base = base.replace(/-(?:Instruct|Chat)(?:-\d{4})?$/i, '')
        progress = true
      }
    }

    if (DATE_PIN.test(base)) {
      base = base.replace(DATE_PIN, '')
      progress = true
    }
  }

  return { base, variant, quant }
}

/** Split a model id into a clean display name plus an optional grayed variant
 *  tag, so distinct ids (e.g. `…-4.8` vs `…-4.8-fast`) don't collapse. */
export function modelDisplayParts(model: string): { name: string; tag: string } {
  let base = modelBaseId(model)

  // Anthropic's `[1m]` route suffix selects the 1M-context window. It is a
  // variant of the same model, so it renders as a tag ("Sonnet 5 · 1M") rather
  // than raw brackets that read like an ANSI escape ("Sonnet 5[1m]"). Peel it
  // before the suffix loop so a date pin under it (`…-20251101[1m]`) is
  // trailing for the loop.
  let contextTag = ''
  const contextWindow = base.match(/\[(\d+[mk])\]$/i)

  if (contextWindow) {
    contextTag = contextWindow[1].toUpperCase()
    base = base.slice(0, -contextWindow[0].length)
  }

  const split = splitTrailingTags(base)
  const tags = [split.variant, split.quant, contextTag].filter(Boolean)

  return { name: prettifyBase(split.base) || model.trim() || 'No model', tag: tags.join(' ') }
}

/** Friendly one-line model name for menus and the status bar. The variant
 *  tag is part of the name: `…-4.8` vs `…-4.8-thinking` must never collapse
 *  to the same label on any surface (#88597). */
export function displayModelName(model: string): string {
  const { name, tag } = modelDisplayParts(model)

  return tag ? `${name} ${tag}` : name
}

/** The variant tag a model id carries (Fast, Flash, Thinking, Preview,
 *  Latest) — the one taxonomy both the catalog rows and the composer pill
 *  split on, so distinct ids never render as one model listed twice.
 *  Quant and context-window tags stay picker-row detail. Derived from the
 *  same suffix split as `modelDisplayParts`, so a quant-bearing local id
 *  (`…-flash-Q4_K_XL`, `…-Q4_K_XL-flash`) reports the same variant on the
 *  pill as the catalog row shows. */
export function modelVariantTag(model: string): string {
  return splitTrailingTags(modelBaseId(model)).variant
}

/** Composer model-pill label — model name plus its variant tag (Fast, Flash,
 *  …) when one applies, separated by a `·` so the pill reads as
 *  "name · variant" at a glance. The reasoning level is NOT here: it has its
 *  own pill (`ReasoningPill`), so a long model name can no longer push the
 *  effort out of the truncating span. Fast shows when the speed=fast param is
 *  on OR the active model is a `…-fast` variant — never both. */
export function formatModelPillLabel(model: string, options?: { fastMode?: boolean }): string {
  const name = modelDisplayParts(model).name

  const tag = model.trim() ? (options?.fastMode ? 'Fast' : modelVariantTag(model)) : ''

  return tag ? `${name} · ${tag}` : name
}
