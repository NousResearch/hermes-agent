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
    return applyVendorCasing(`GPT-${titleCase(base.replace(/^gpt-/i, '').replace(/-/g, ' '))}`)
  }

  // Title-case this branch too: without it `gemini-2.5-pro` rendered as
  // "Gemini 2.5 pro" — the only branch that left its words lowercase.
  if (/^gemini-/i.test(base)) {
    return applyVendorCasing(titleCase(base.replace(/^gemini-/i, 'Gemini ').replace(/-/g, ' ')))
  }

  return applyVendorCasing(titleCase(base.replace(/-/g, ' ')))
}

// GGUF quant of a local id: `…-UD-Q4_K_XL`, `…-Q8_0`, `…-BF16`. The UD
// prefix is part of the quant and consumed with it.
const QUANT_ALT = '(?<quant>Q\\d(?:_[A-Z0-9]+)*|IQ\\d(?:_[A-Z0-9]+)*|F16|BF16)'

// Trailing words that can follow a quant in a local id (`…-Q4_K_XL-flash`).
// Only used to recognize the quant's order; the word itself stays in the name.
const LOCAL_TRAILING_WORDS = new Set(['fast', 'flash', 'thinking', 'preview', 'latest'])

/** Extract the GGUF quant of a local-build id in either order —
 *  `…-flash-Q4_K_XL` and `…-Q4_K_XL-flash` are the same model. The trailing
 *  variant word is retained in the base so it renders as part of the name. */
function splitLocalQuant(base: string): { base: string; quant: string } {
  let quantMatch = base.match(new RegExp(`^(?<head>.*?)(?:-UD-|-)${QUANT_ALT}$`, 'i'))

  if (quantMatch && quantMatch.groups) {
    base = quantMatch.groups.head
    // Instruct/chat markers are noise once the quant confirmed a local build.
    base = base.replace(/-(?:Instruct|Chat)(?:-\d{4})?$/i, '')

    return { base, quant: quantMatch.groups.quant.split('_')[0].toUpperCase() }
  }

  quantMatch = base.match(new RegExp(`^(?<head>.*?)(?:-UD-|-)${QUANT_ALT}-(?<tail>[a-z]+)$`, 'i'))

  if (quantMatch && quantMatch.groups && LOCAL_TRAILING_WORDS.has(quantMatch.groups.tail.toLowerCase())) {
    return {
      base: `${quantMatch.groups.head}-${quantMatch.groups.tail}`,
      quant: quantMatch.groups.quant.split('_')[0].toUpperCase()
    }
  }

  return { base, quant: '' }
}

/** Format a trailing 8-digit pin as a date (`20251101` → "2025-11-01");
 *  null when the digits are not a real calendar date. */
function formatDatePin(digits: string): null | string {
  const year = Number(digits.slice(0, 4))
  const month = Number(digits.slice(4, 6))
  const day = Number(digits.slice(6, 8))

  if (month < 1 || month > 12 || day < 1 || day > 31) {
    return null
  }

  // Date.UTC with day 0 gives the previous month's last day.
  if (day > new Date(Date.UTC(year, month, 0)).getUTCDate()) {
    return null
  }

  const pad = (value: number): string => String(value).padStart(2, '0')

  return `${year}-${pad(month)}-${pad(day)}`
}

/** Split a model id into a display name plus an optional tag. Identity words
 *  and date pins stay in the name; only non-identity qualifiers of local ids
 *  (the GGUF quant) and the context-window route become tags. Rows that still
 *  render identically get a distinguishing chip from the menu's section-level
 *  collision fallback. */
export function modelDisplayParts(model: string): { name: string; tag: string } {
  let base = modelBaseId(model)
  const tags: string[] = []

  const { base: quantBase, quant } = splitLocalQuant(base)
  base = quantBase

  if (quant) {
    tags.push(quant)
  }

  // Anthropic's `[1m]` route suffix selects the 1M-context window. It is a
  // variant of the same model, so it renders as a tag ("Sonnet 5 · 1M") rather
  // than raw brackets that read like an ANSI escape ("Sonnet 5[1m]").
  const contextWindow = base.match(/\[(\d+[mk])\]$/i)

  if (contextWindow) {
    tags.push(contextWindow[1].toUpperCase())
    base = base.slice(0, -contextWindow[0].length)
  }

  // A trailing 8-digit date-pin stays visible as text (formatted when it is a
  // real date, otherwise left for generic prettification): a snapshot id and
  // its base id must remain distinct rows.
  const datePin = base.match(/-(\d{8})$/)
  let pinText = ''

  if (datePin) {
    pinText = formatDatePin(datePin[1]) ?? ''

    if (pinText) {
      base = base.slice(0, -datePin[0].length)
    }
  }

  const name = prettifyBase(base) || model.trim() || 'No model'

  return { name: pinText ? `${name} ${pinText}` : name, tag: tags.join(' ') }
}

/** Friendly one-line model name for menus and the status bar. */
export function displayModelName(model: string): string {
  const { name, tag } = modelDisplayParts(model)

  return tag ? `${name} ${tag}` : name
}

/** Composer model-pill label — the model name. The reasoning level has its own
 *  pill (`ReasoningPill`), and an active speed=fast appends "· Fast". */
export function formatModelPillLabel(model: string, options?: { fastMode?: boolean }): string {
  const name = modelDisplayParts(model).name

  return model.trim() && options?.fastMode ? `${name} · Fast` : name
}
