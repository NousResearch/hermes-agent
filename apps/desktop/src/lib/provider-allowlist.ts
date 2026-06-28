import type { ModelOptionProvider } from '@/types/hermes'

// ApexNodes is a China-first managed product, so the model picker only surfaces
// two kinds of provider:
//
//   1. The APEX-NODES.COM managed relay (the zero-key default). The relay is
//      registered as a *named custom provider* (electron/apex-managed.cjs:
//      MANAGED_PROVIDER_NAME = 'Apex-nodes.com'), so the runtime emits it with
//      slug `custom:apex-nodes.com` (custom_provider_slug() in the runtime's
//      hermes_cli/providers.py: `"custom:" + name.lower().replace(" ", "-")`).
//      A bare `custom` row can also appear for a user's own OpenAI-compatible
//      endpoint. Both are kept — the bare/`custom:*` slugs ARE the BYOK escape
//      hatch for adding a native key behind an OpenAI-compatible URL.
//
//   2. Domestic (国产 / mainland-China) LLM providers, so a user can BYOK their
//      own native key (DeepSeek, Qwen/通义, GLM/智谱, Kimi/Moonshot, MiniMax,
//      StepFun, Xiaomi MiMo, Tencent …). The "编辑模型 / Add provider" flow for
//      these stays intact.
//
// Everything foreign is hidden (OpenAI, Anthropic, xAI/Grok, Google/Gemini,
// GitHub Copilot, Mistral, Cohere, OpenRouter, Nous, Bedrock, Azure, NVIDIA,
// Hugging Face, Ollama, Groq, Together, Fireworks, …) — even if the user has a
// key configured, it won't show up in the picker.
//
// The runtime's PROVIDER_REGISTRY (hermes_cli/auth.py) has no per-provider
// region/origin field to filter on, so this is an explicit slug allowlist that
// mirrors the existing China-first split already used by PROVIDER_GROUPS
// (settings/constants.ts, domestic priority 1–9). Verify any new id against the
// runtime registry before adding it here.
//
// To add a domestic provider: drop its registry slug into DOMESTIC_PROVIDER_SLUGS.

/** Domestic (mainland-China stable, no VPN) provider slugs from the runtime's
 *  PROVIDER_REGISTRY (hermes_cli/auth.py). Lowercase; matched case-insensitively. */
export const DOMESTIC_PROVIDER_SLUGS: ReadonlySet<string> = new Set([
  'deepseek', // DeepSeek (V3.x / R1) — the recommended default
  'zai', // Z.AI / GLM (Zhipu)
  'kimi-coding', // Kimi / Moonshot (international platform.moonshot.ai)
  'kimi-coding-cn', // Kimi / Moonshot (China platform.moonshot.cn)
  'alibaba', // Qwen Cloud / DashScope (通义千问)
  'alibaba-coding-plan', // Alibaba Cloud Coding Plan (Qwen)
  'qwen-oauth', // Qwen OAuth
  'minimax', // MiniMax (international)
  'minimax-oauth', // MiniMax OAuth
  'minimax-cn', // MiniMax (China)
  'stepfun', // StepFun Step Plan
  'xiaomi', // Xiaomi MiMo
  'tencent-tokenhub' // Tencent TokenHub
])

/** True when a provider slug is the ApexNodes managed relay or a user's own
 *  custom / local OpenAI-compatible endpoint. Covers the bare `custom` slug and
 *  any named `custom:<name>` (e.g. `custom:apex-nodes.com`). */
function isCustomOrManagedSlug(slug: string): boolean {
  return slug === 'custom' || slug.startsWith('custom:')
}

/** Whether a provider should appear in the China-first model picker: the
 *  APEX-NODES.COM managed relay / a custom BYOK endpoint, or a domestic
 *  provider. Foreign providers return false. */
export function isPickerVisibleProvider(slug: string): boolean {
  const normalized = String(slug || '')
    .trim()
    .toLowerCase()

  if (!normalized) {
    return false
  }

  return isCustomOrManagedSlug(normalized) || DOMESTIC_PROVIDER_SLUGS.has(normalized)
}

/** Keep only the providers the China-first picker should show (APEX-NODES.COM +
 *  custom BYOK + domestic). Order is preserved. */
export function filterPickerProviders(providers: ModelOptionProvider[]): ModelOptionProvider[] {
  return providers.filter(provider => isPickerVisibleProvider(provider.slug))
}
