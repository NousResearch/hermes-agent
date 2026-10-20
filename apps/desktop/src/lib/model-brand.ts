/** Model author, not the hosting provider (e.g. Claude on OpenRouter). */
export type ModelBrand =
  'openai' | 'anthropic' | 'zai' | 'google' | 'deepseek' | 'qwen' | 'xai' | 'mistral' | 'meta' | 'nous' | 'unknown'

const MODEL_FAMILIES: readonly [ModelBrand, RegExp][] = [
  ['openai', /^(?:gpt-|chatgpt-|o\d+(?:$|[-.:]))/],
  ['anthropic', /^(?:(?:[a-z]{2}\.)?anthropic\.)?claude[-.]/],
  ['zai', /^glm[-.]/],
  ['google', /^(?:gemini[-.]|gemma(?:[-.\d]))/],
  ['deepseek', /^deepseek[-.]/],
  ['qwen', /^(?:qwen(?:[-.\d])|qwq[-.])/],
  ['xai', /^grok[-.]/],
  ['mistral', /^(?:mistral|mixtral|codestral|devstral|ministral|magistral)[-.]/],
  ['meta', /^llama[-.\d]/],
  ['nous', /^hermes[-.]/]
]

/** Namespaced/quantized model IDs retain their family; unknown names stay neutral. */
export function modelBrand(model: string | null | undefined): ModelBrand {
  const name = model?.trim().toLowerCase().split('/').at(-1) ?? ''

  return MODEL_FAMILIES.find(([, pattern]) => pattern.test(name))?.[0] ?? 'unknown'
}
