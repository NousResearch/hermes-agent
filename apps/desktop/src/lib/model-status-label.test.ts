import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelPillLabel,
  modelDisplayParts,
  modelVariantTag
} from './model-status-label'

describe('model-status-label', () => {
  it('strips trailing date-pin snapshots and dots hyphenated Anthropic versions', () => {
    expect(displayModelName('claude-opus-4-5-20251101')).toBe('Opus 4.5')
    expect(displayModelName('anthropic/claude-haiku-4-5-20251001')).toBe('Haiku 4.5')
    expect(displayModelName('claude-fable-5-1')).toBe('Fable 5.1')
  })

  it('strips hyphenated date pins vendors write as YYYY-MM-DD (#124884)', () => {
    // Both spellings pin the same model: `…-20251101` and `…-2026-05-17`.
    // The hyphenated form previously survived the `-\d{8}$` strip and
    // rendered the snapshot date as part of the model name.
    expect(displayModelName('qwen3.7-max-2026-05-17')).toBe('Qwen3.7 Max')
    expect(displayModelName('qwen3.7-max-2026-05-17')).not.toContain('2026')
    expect(modelDisplayParts('qwen3.7-max-2026-05-17')).toEqual({ name: 'Qwen3.7 Max', tag: '' })
    // A base id and its date-pinned snapshot must never render as two rows.
    expect(displayModelName('qwen3.7-max-2026-05-17')).toBe(displayModelName('qwen3.7-max'))
  })

  it('peels a date pin under the variant suffix so pinned variants keep their tag (#124884)', () => {
    // The pin strip used to run AFTER the variant/quant split, so a pinned
    // variant id never matched `…-thinking$` and rendered tagless — and a
    // catalog carrying both the pinned and unpinned id showed two rows
    // differing only by a trailing date.
    expect(modelDisplayParts('glm-5.3-thinking-20251101')).toEqual({ name: 'GLM 5.3', tag: 'Thinking' })
    expect(modelDisplayParts('glm-5.3-thinking-2026-05-17')).toEqual({ name: 'GLM 5.3', tag: 'Thinking' })
    expect(modelVariantTag('glm-5.3-thinking-20251101')).toBe('Thinking')
    expect(formatModelPillLabel('glm-5.3-thinking-20251101')).toBe('GLM 5.3 · Thinking')
  })

  it('never mistakes a plain numeric suffix for a date pin (#124884)', () => {
    // Month 56 and day 78 are not a date; the suffix is the id's own.
    expect(modelDisplayParts('test-1234-56-78')).toEqual({ name: 'Test 1234 56 78', tag: '' })
    // 8-digit groups are only a pin at the very end of the id.
    expect(displayModelName('model-20251101-x')).toBe('Model 20251101 X')
  })

  it('peels a date pin under a quant or context suffix the reviewer cases missed (#124884)', () => {
    // The peel used to be guarded on `!variant && !quant`, so a pin exposed by
    // a quant strip was never re-tested — and the `[1m]` slice ran after the
    // loop, so a pin under it was not trailing either. Both regress vs the
    // unconditional strip this PR replaces.
    expect(modelDisplayParts('qwen3-max-20251101-Q4_K_XL')).toEqual({ name: 'Qwen3 Max', tag: 'Q4' })
    expect(modelDisplayParts('claude-sonnet-4-5-20251101[1m]')).toEqual({ name: 'Sonnet 4.5', tag: '1M' })
    // Base left the variant in the name here (the pin hid it from the split);
    // the peel now keeps it a tag, matching the pin-under-variant case above.
    expect(modelDisplayParts('glm-5.3-thinking-20251101[1m]')).toEqual({ name: 'GLM 5.3', tag: 'Thinking 1M' })
  })

  it('renders the Anthropic 1M-context route suffix as a tag, never raw brackets', () => {
    expect(modelDisplayParts('claude-sonnet-5[1m]')).toEqual({ name: 'Sonnet 5', tag: '1M' })
    expect(modelDisplayParts('claude-fable-5-1[1m]')).toEqual({ name: 'Fable 5.1', tag: '1M' })
    expect(displayModelName('claude-opus-5[1m]')).not.toContain('[')
  })

  it('renders local GGUF ids as a clean name with a quant tag', () => {
    expect(modelDisplayParts('Qwen3.6-27B-UD-Q4_K_XL')).toEqual({ name: 'Qwen3.6 27B', tag: 'Q4' })
    expect(modelDisplayParts('Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL')).toEqual({
      name: 'Nemotron 3 Nano 30B A3B',
      tag: 'Q4'
    })
    expect(modelDisplayParts('Qwen3-4B-Instruct-2507-UD-Q8_K_XL')).toEqual({ name: 'Qwen3 4B', tag: 'Q8' })
    expect(modelDisplayParts('some-model-Q6_K')).toEqual({ name: 'Some Model', tag: 'Q6' })
    // Cloud ids keep their existing behavior.
    expect(modelDisplayParts('anthropic/claude-opus-4.8-fast').tag).toBe('Fast')
  })

  it('keeps the vendor casing the model id does not carry (#85849)', () => {
    expect(displayModelName('glm-5.2')).toBe('GLM 5.2')
    expect(displayModelName('zai-org/glm-5.1')).toBe('GLM 5.1')
    expect(displayModelName('deepseek-v4-flash')).toBe('DeepSeek V4 Flash')
    expect(displayModelName('minimax/minimax-01')).toBe('MiniMax 01')
    expect(displayModelName('xiaomi/mimo-v2.5')).toBe('MiMo V2.5')
    expect(displayModelName('ernie-5.1')).toBe('ERNIE 5.1')
    expect(displayModelName('baai/bge-m3')).toBe('BGE M3')
    expect(displayModelName('openai')).toBe('OpenAI')
  })

  it('capitalises parameter counts the way vendors write them (#85849)', () => {
    expect(displayModelName('qwen3-32b')).toBe('Qwen3 32B')
    expect(displayModelName('qwen/qwen3.5-35b-a3b')).toBe('Qwen3.5 35B A3B')
    expect(displayModelName('meta/llama-3.1-8b-instruct')).toBe('Llama 3.1 8B Instruct')
    expect(displayModelName('llama-3.1-8b-instruct-fp8')).toBe('Llama 3.1 8B Instruct FP8')
    expect(displayModelName('gemma-4-26b-a4b-it')).toBe('Gemma 4 26B A4B IT')
    expect(displayModelName('nemotron-nano-12b-v2-vl')).toBe('Nemotron Nano 12B V2 VL')
  })

  it('title-cases gemini names like every other branch (#85849)', () => {
    expect(displayModelName('gemini-2.5-pro')).toBe('Gemini 2.5 Pro')
    expect(displayModelName('gemini-2.0-flash')).toBe('Gemini 2.0 Flash')
    expect(displayModelName('google/gemini-2.5-flash-lite')).toBe('Gemini 2.5 Flash Lite')
  })

  it('distinguishes the deepseek-flash alias from its deepseek-v4.1-flash sibling (#118083)', () => {
    // models.dev carries both ids for the provider: `deepseek-flash` (alias)
    // and `deepseek-v4.1-flash` (full id). Two distinct ids must never render
    // as near-identical tagless rows the user reads as one model listed twice.
    // The `-flash` variant tag splits the pair the same way `-fast` splits
    // `…-4.8` vs `…-4.8-fast`, and the vendor casing closes the gap that made
    // the alias read "DeepSeek" while its sibling read "Deepseek".
    expect(modelDisplayParts('deepseek-flash')).toEqual({ name: 'DeepSeek', tag: 'Flash' })
    expect(modelDisplayParts('deepseek-v4.1-flash')).toEqual({ name: 'DeepSeek V4.1', tag: 'Flash' })
    // Non-flash siblings stay tagless and distinct from their flash variant.
    expect(modelDisplayParts('deepseek-v4.1')).toEqual({ name: 'DeepSeek V4.1', tag: '' })
  })

  it('keeps the variant tag in the display name so distinct ids never collapse (#88597)', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 Fast')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('DeepSeek V4 Pro Thinking')
    expect(displayModelName('gpt-5.5-preview')).toBe('GPT-5.5 Preview')
    expect(displayModelName('claude-opus-5')).toBe('Opus 5')
    // A base model and its variant must NEVER share a display label.
    expect(displayModelName('claude-opus-5')).not.toBe(displayModelName('claude-opus-5-thinking'))
    // The quant/contextWindow tags ride along the same way.
    expect(displayModelName('Qwen3.6-27B-UD-Q4_K_XL')).toBe('Qwen3.6 27B Q4')
    expect(displayModelName('claude-sonnet-5[1m]')).toBe('Sonnet 5 1M')
  })

  it('keeps the model pill to name + Fast; the effort lives on its own pill', () => {
    expect(formatModelPillLabel('openai/gpt-5.5', { fastMode: true })).toBe('GPT-5.5 · Fast')
    expect(formatModelPillLabel('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8 · Fast')
    expect(formatModelPillLabel('openai/gpt-5.5')).toBe('GPT-5.5')
    expect(formatModelPillLabel('')).toBe('No model')
  })

  it('rides the same variant tags on the pill as the catalog rows (#118083)', () => {
    // A `-flash` id must render its tag on the composer pill too: without it,
    // `gemini-2.5-flash` and `gemini-2.5` produce identical pill text, so a
    // model switch reads as a no-op — the exact look-alike collapse the
    // catalog-row split exists to prevent, one screen over.
    expect(formatModelPillLabel('gemini-2.5-flash')).toBe('Gemini 2.5 · Flash')
    expect(formatModelPillLabel('deepseek-v4.1-flash')).toBe('DeepSeek V4.1 · Flash')
    expect(formatModelPillLabel('qwen3.8-flash')).toBe('Qwen3.8 · Flash')
    // The bare models stay tagless — and distinct from their flash variants.
    expect(formatModelPillLabel('gemini-2.5')).toBe('Gemini 2.5')
    expect(formatModelPillLabel('deepseek-v4.1')).toBe('DeepSeek V4.1')
  })

  it('agrees with the catalog rows on the variant for local ids that carry both a variant and a quant', () => {
    // A local GGUF id can carry the variant and the quant in either order —
    // `…-flash-Q4_K_XL` and `…-Q4_K_XL-flash` are the same model. The row
    // used to split only on whichever suffix it saw first (quant soup in the
    // name for one order, no variant tag for the other), while the pill
    // re-derived the variant from the raw id, so the two screens disagreed
    // (#118083 review).
    expect(modelVariantTag('Qwen3.6-27B-flash-Q4_K_XL')).toBe('Flash')
    expect(modelVariantTag('Qwen3.6-27B-Q4_K_XL-flash')).toBe('Flash')
    // Both orders decompose identically: clean name, variant + quant tag.
    expect(modelDisplayParts('Qwen3.6-27B-flash-Q4_K_XL')).toEqual({ name: 'Qwen3.6 27B', tag: 'Flash Q4' })
    expect(modelDisplayParts('Qwen3.6-27B-Q4_K_XL-flash')).toEqual({ name: 'Qwen3.6 27B', tag: 'Flash Q4' })
    // The pill rides the variant and keeps the quant as picker-row detail.
    expect(formatModelPillLabel('Qwen3.6-27B-flash-Q4_K_XL')).toBe('Qwen3.6 27B · Flash')
    expect(formatModelPillLabel('Qwen3.6-27B-Q4_K_XL-flash')).toBe('Qwen3.6 27B · Flash')
    // Quant-only ids are unchanged: quant stays off the pill.
    expect(modelDisplayParts('Qwen3.6-27B-Q8_0')).toEqual({ name: 'Qwen3.6 27B', tag: 'Q8' })
    expect(formatModelPillLabel('Qwen3.6-27B-Q8_0')).toBe('Qwen3.6 27B')
  })

  describe('currentPickerSelection', () => {
    const store = { model: 'opus', provider: 'anthropic' }
    const options = { model: 'hermes-4', provider: 'nous' }

    it('prefers the sticky composer pick over the profile default pre-session', () => {
      expect(currentPickerSelection(store, options)).toEqual(store)
    })

    it('falls back to options when the store is empty', () => {
      expect(currentPickerSelection({ model: '', provider: '' }, options)).toEqual(options)
    })

    it('uses the complete options pair instead of mixing a partial store selection', () => {
      expect(currentPickerSelection({ model: 'opus', provider: '' }, options)).toEqual(options)
    })

    it('falls back to the store while options are still loading', () => {
      expect(currentPickerSelection(store, undefined)).toEqual(store)
    })
  })
})
