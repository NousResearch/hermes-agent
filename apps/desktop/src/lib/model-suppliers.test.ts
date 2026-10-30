import { describe, expect, it } from 'vitest'

import { displayModelName, formatModelPillLabel, isGatewayCatalog, modelDisplayParts } from './model-status-label'
import { isKnownSupplier, knownSupplierNamer, supplierNamer, supplierOf } from './model-suppliers'

const MAGPIE = [
  'workbuddy/hy3',
  'workbuddy-ai/hy3',
  'qoder/qfmodel',
  'qoder-cn/qfmodel',
  'trae-cn/deepseek-v4.1-flash-max',
  'codearts/glm-5.3'
]

describe('supplierOf', () => {
  it('names a known magpie supplier', () => {
    expect(supplierOf('workbuddy-ai/hy3')).toEqual({
      codicon: 'sparkle',
      mark: 'WA',
      name: 'WorkBuddy AI'
    })
  })

  it('falls back to initials for an unknown supplier rather than dropping it', () => {
    expect(supplierOf('new-vendor/model-x')).toEqual({ mark: 'NV', name: 'New Vendor' })
  })

  it('declines a bare id, which carries no supplier', () => {
    expect(supplierOf('gpt-6-luna')).toBeNull()
  })
})

describe('isGatewayCatalog', () => {
  it('accepts a magpie catalog: several KNOWN suppliers behind one endpoint', () => {
    expect(isGatewayCatalog(MAGPIE, isKnownSupplier)).toBe(true)
  })

  it('declines an aggregator whose prefixes are model vendors, not magpie suppliers', () => {
    // OpenRouter / Nous Portal ids are `vendor/model` too — promoting them
    // would put a "· Anthropic" chip on every row of every aggregator.
    expect(isGatewayCatalog(['anthropic/claude-sonnet-5', 'openai/gpt-5.5'], isKnownSupplier)).toBe(false)
  })

  it('declines a single-vendor endpoint (one supplier is not a gateway)', () => {
    expect(isGatewayCatalog(['workbuddy/hy3', 'workbuddy/hy4-preview'], isKnownSupplier)).toBe(false)
  })

  it('declines a flat catalog of bare ids', () => {
    expect(isGatewayCatalog(['gpt-6-luna', 'glm-5.3'], isKnownSupplier)).toBe(false)
  })
})

describe('modelDisplayParts on a gateway', () => {
  it('appends the full supplier and drops the variant chip', () => {
    expect(modelDisplayParts('workbuddy/deepseek-v4.1-flash', { supplier: supplierNamer })).toEqual({
      name: 'DeepSeek V4.1 Flash · WorkBuddy',
      tag: ''
    })
    expect(modelDisplayParts('workbuddy-ai/deepseek-v4.1-flash', { supplier: supplierNamer })).toEqual({
      name: 'DeepSeek V4.1 Flash · WorkBuddy AI',
      tag: ''
    })
  })

  it('keeps look-alike names apart', () => {
    const a = modelDisplayParts('qoder/qfmodel', { supplier: supplierNamer }).name
    const b = modelDisplayParts('qoder-cn/qfmodel', { supplier: supplierNamer }).name

    expect(a).toBe('Qfmodel · Qoder')
    expect(b).toBe('Qfmodel · Qoder CN')
    expect(a).not.toBe(b)
  })

  it('folds the whole vendor name into the label, so no Flash/Preview/Max chip survives', () => {
    const parts = modelDisplayParts('trae-cn/glm-5.3-flashx-max', { supplier: supplierNamer })

    expect(parts.name).toBe('GLM 5.3 Flashx Max · Trae CN')
    expect(parts.tag).toBe('')
  })

  it('leaves a non-gateway provider exactly as core painted it', () => {
    expect(modelDisplayParts('anthropic/claude-sonnet-5', { supplier: knownSupplierNamer })).toEqual(
      modelDisplayParts('anthropic/claude-sonnet-5')
    )
    expect(modelDisplayParts('anthropic/claude-sonnet-5').tag).toBe('')
  })
})

describe('displayModelName / pill on a gateway', () => {
  it('carries the supplier through the one-line name', () => {
    expect(displayModelName('workbuddy-ai/hy3', { supplier: knownSupplierNamer })).toBe('Hy3 · WorkBuddy AI')
  })

  it('declines an unknown prefix, so a normal provider keeps its label', () => {
    expect(knownSupplierNamer('anthropic')).toBe('')
    expect(displayModelName('anthropic/claude-opus-4-5', { supplier: knownSupplierNamer })).toBe('Opus 4.5')
  })
})

describe('formatModelPillLabel on a gateway', () => {
  it('keeps the fast/ultrafast speed tag the pill contract promises', () => {
    // The supplier label must not swallow the speed indication: a manual
    // speed=fast pick rides the same gateway model as a normal one.
    expect(formatModelPillLabel('workbuddy/hy3', { supplier: knownSupplierNamer, fastMode: true })).toBe(
      'Hy3 · WorkBuddy · Fast'
    )
    expect(
      formatModelPillLabel('workbuddy/hy3', { supplier: knownSupplierNamer, serviceTier: 'ultrafast' })
    ).toBe('Hy3 · WorkBuddy · Ultrafast')
  })
})
