import { expect, test } from 'vitest'

import { pickerModelMutation, slashMutation } from './canonical-protocol'

// Regression for #145363: a custom endpoint whose actual model ID contains
// whitespace (e.g. "Gemini 4 Argon") produced a `config.set key=model` value
// that the whitespace-splitting parser could not read, so the pick stayed on
// the legacy config route and the gateway refused it with
// `use_session_mutation_model`. The parser must take the known
// `--provider <id> [--session]` suffix off the END of the value and treat
// everything before it as the model ID — while still refusing unsupported
// flags instead of swallowing them into the model name.

test('model id containing spaces parses with --provider and --session', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider my-provider --session'))
    .toEqual({ model: 'Gemini 4 Argon', provider: 'my-provider' })
})

test('model id containing spaces parses without --session', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider my-provider'))
    .toEqual({ model: 'Gemini 4 Argon', provider: 'my-provider' })
})

test('model id with multiple internal spaces and a namespaced provider id', () => {
  expect(pickerModelMutation('Qwen3.8 Flash   Next --provider custom:halogen-qwen-3.8 --session'))
    .toEqual({ model: 'Qwen3.8 Flash   Next', provider: 'custom:halogen-qwen-3.8' })
})

test('non-ASCII model id containing spaces', () => {
  expect(pickerModelMutation('模型 大号 极速 --provider local-basalt --session'))
    .toEqual({ model: '模型 大号 极速', provider: 'local-basalt' })
})

test('control: a model id without spaces keeps parsing', () => {
  expect(pickerModelMutation('gpt-6-astra --provider openai --session'))
    .toEqual({ model: 'gpt-6-astra', provider: 'openai' })
})

test('unsupported --global flag is refused, not swallowed into the model name', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider my-provider --global')).toBeNull()
  expect(pickerModelMutation('Gemini 4 Argon --global --provider my-provider')).toBeNull()
})

test('unsupported --once flag is refused', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider my-provider --once')).toBeNull()
})

test('repeated --provider is refused', () => {
  expect(pickerModelMutation('a --provider b --provider c')).toBeNull()
})

test('missing provider value is refused', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider')).toBeNull()
  expect(pickerModelMutation('--provider my-provider')).toBeNull()
})

test('flag-shaped provider is refused', () => {
  expect(pickerModelMutation('Gemini 4 Argon --provider --session')).toBeNull()
})

test('empty and non-string input is refused', () => {
  expect(pickerModelMutation('')).toBeNull()
  expect(pickerModelMutation(null)).toBeNull()
  expect(pickerModelMutation(undefined)).toBeNull()
})

test('/model with a spaced id and --provider takes the canonical mutation', () => {
  expect(slashMutation('/model Gemini 4 Argon --provider my-provider --session'))
    .toEqual({ operation: 'model', payload: { model: 'Gemini 4 Argon', provider: 'my-provider' } })
})
