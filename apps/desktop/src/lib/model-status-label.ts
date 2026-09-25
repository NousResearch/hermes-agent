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
 * provider ids stay distinct from their direct-API counterparts so a session on
 * `xai-oauth` never reads as the plain `xai` key path, and internal route names
 * never reach user-facing copy. */
export const PROVIDER_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  anthropic: 'Anthropic API Key',
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
// the same display name.
const VARIANT_TAGS: ReadonlyArray<readonly [RegExp, string]> = [
  [/-fast$/i, 'Fast'],
  [/-thinking$/i, 'Thinking'],
  [/-preview$/i, 'Preview'],
  [/-latest$/i, 'Latest']
]

const titleCase = (text: string): string => text.replace(/\b\w/g, char => char.toUpperCase()).trim()

// Vendor families whose human name is not a plain capitalisation of the id
// token. Anything else capitalises (`qwen` → "Qwen", `kimi` → "Kimi").
const BRAND_CASING: Readonly<Record<string, string>> = {
  deepseek: 'DeepSeek',
  glm: 'GLM',
  gpt: 'GPT',
  minimax: 'MiniMax'
}

// Families that glue the version to the brand with a hyphen ("GPT-5.5",
// "GLM-5.1") instead of a space ("Gemini 3", "Llama 3.3").
const HYPHENATED_BRANDS = new Set(['gpt', 'glm'])

const isVersionToken = (tok: string): boolean => /^\d+(?:\.\d+)*$/.test(tok)
// `v4`, `k2`, `r1` — a single letter fronting a version.
const isLetterVersionToken = (tok: string): boolean => /^[a-z]\d+(?:\.\d+)*$/.test(tok)
// Parameter counts: `27b`, `a22b` (active params), `8x22b` (MoE experts).
const isSizeToken = (tok: string): boolean => /^(?:[a-z]?\d+(?:\.\d+)?|\d+x\d+)b$/.test(tok)
// Precision / quant shorthands with no dedicated tag: `fp8`, `int4`, `nvfp4`.
const isShortCodeToken = (tok: string): boolean => /^[a-z]{2,5}\d{1,2}$/.test(tok)

/** A token the id's author already cased by hand (`MoE`, `QwQ`, `DeepSeek`)
 *  keeps that casing; everything else is derived from the lower-cased token. */
const authorCased = (raw: string): boolean => /[A-Z]/.test(raw.slice(1))

/** Token-class label grammar for non-Anthropic ids (adapted from
 *  block/buzz#7844). Each `-` separated token is classified once — version,
 *  letter-version, parameter size, short code, or word — and rendered from its
 *  own text, so `qwen3-235b-a22b` reads "Qwen3 235B A22B", `glm-5-1` reads
 *  "GLM-5.1" and `gpt-oss-120b` reads "GPT OSS 120B" instead of the title-cased
 *  soup ("Qwen3 235b A22b", "Glm 5 1", "GPT-oss-120b") a blanket
 *  hyphen→space + capitalise pass produced. Versions are never reinterpreted:
 *  only consecutive bare numbers merge into a dotted version, and a brand that
 *  already carries digits (`qwen3-5-…`) keeps the following number separate
 *  rather than guessing "Qwen3.5". */
function labelFromTokens(base: string): string {
  const rawTokens = base.split('-').filter(Boolean)

  if (rawTokens.length === 0) {
    return ''
  }

  const [rawBrand, ...rawRest] = rawTokens

  // A trailing 4-digit snapshot pin (`deepseek-r1-0528`) is release noise, not
  // a version; the 8-digit form was already dropped upstream.
  if (rawRest.length > 0 && /^\d{4}$/.test(rawRest[rawRest.length - 1])) {
    rawRest.pop()
  }

  const brandLower = rawBrand.toLowerCase()
  const family = /^[a-z]+/.exec(brandLower)?.[0] ?? ''

  const brand = authorCased(rawBrand)
    ? rawBrand
    : (BRAND_CASING[family] ?? (family ? family[0].toUpperCase() + family.slice(1) : '')) +
      brandLower.slice(family.length)

  const parts: string[] = []
  let sawVersion = false

  for (let i = 0; i < rawRest.length; i += 1) {
    const raw = rawRest[i]
    const tok = raw.toLowerCase()
    // Consecutive bare numbers (`5-1`, `k2-5`) are one dotted version.
    let minor = ''

    while (
      (isVersionToken(tok) || isLetterVersionToken(tok)) &&
      i + 1 < rawRest.length &&
      isVersionToken(rawRest[i + 1].toLowerCase())
    ) {
      minor += `.${rawRest[i + 1]}`
      i += 1
    }

    if (isVersionToken(tok)) {
      parts.push(tok + minor)
      sawVersion = true
    } else if (isLetterVersionToken(tok)) {
      parts.push(tok[0].toUpperCase() + tok.slice(1) + minor)
      sawVersion = true
    } else if (isSizeToken(tok)) {
      // `8x22b` keeps its lower-case multiplier; the count suffix is always B.
      parts.push(tok.includes('x') ? `${tok.slice(0, -1)}B` : tok.toUpperCase())
    } else if (isShortCodeToken(tok)) {
      parts.push(tok.toUpperCase())
    } else if (authorCased(raw)) {
      parts.push(raw)
    } else if (tok === 'oss') {
      parts.push('OSS')
    } else if (family === 'gpt' && sawVersion && (tok === 'mini' || tok === 'nano')) {
      // OpenAI's own spelling: "GPT-5.4 mini", not "GPT-5.4 Mini".
      parts.push(tok)
    } else {
      parts.push(titleCase(tok))
    }
  }

  return parts.reduce((label, part, index) => {
    const glue = index === 0 && HYPHENATED_BRANDS.has(family) && isVersionToken(part) ? '-' : ' '

    return label + glue + part
  }, brand)
}

function prettifyBase(base: string): string {
  if (/^deepseek-flash$/i.test(base)) {
    return 'DeepSeek V4.1 Flash'
  }

  if (/^claude-/i.test(base)) {
    // Anthropic ids spell the version with hyphens (`haiku-4-5`, `fable-5-1`);
    // the human name is dotted ("Haiku 4.5"), not "Haiku 4 5".
    return titleCase(
      base
        .replace(/^claude-/i, '')
        .replace(/(\d)-(?=\d)/g, '$1.')
        .replace(/-/g, ' ')
    )
  }

  return labelFromTokens(base)
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

  // Anthropic's `[1m]` route suffix selects the 1M-context window. It is a
  // variant of the same model, so it renders as a tag ("Sonnet 5 · 1M") rather
  // than raw brackets that read like an ANSI escape ("Sonnet 5[1m]").
  const contextWindow = base.match(/\[(\d+[mk])\]$/i)

  if (contextWindow) {
    tag = tag ? `${tag} ${contextWindow[1].toUpperCase()}` : contextWindow[1].toUpperCase()
    base = base.slice(0, -contextWindow[0].length)
  }

  // Drop a trailing date-pin (`…-20251101`) — snapshot noise, not a name.
  base = base.replace(/-\d{8}$/, '')

  return { name: prettifyBase(base) || model.trim() || 'No model', tag }
}

/** Friendly one-line model name for menus and the status bar. */
export function displayModelName(model: string): string {
  return modelDisplayParts(model).name
}

/** Composer model-pill label — model name plus Fast when it applies. The
 *  reasoning level is NOT here: it has its own pill (`ReasoningPill`), so a
 *  long model name can no longer push the effort out of the truncating span. */
export function formatModelPillLabel(model: string, options?: { fastMode?: boolean }): string {
  const name = displayModelName(model)

  // Fast is shown when the speed=fast param is on (options.fastMode) OR the
  // active model is a `…-fast` variant (fast via a separate model id).
  if (model.trim() && (options?.fastMode || /-fast$/i.test(modelBaseId(model)))) {
    return `${name} · Fast`
  }

  return name
}
