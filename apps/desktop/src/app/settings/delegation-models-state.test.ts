import { describe, expect, it } from 'vitest'

import {
  delegationDraftValid, delegationModelsPatch, moveDelegationFallback,
  readDelegationModels, updateDelegationFallback
} from './delegation-models-state'

const worker = {
  provider: 'custom:worker', model: 'worker/model', base_url: 'https://worker.invalid/v1',
  key_env: 'WORKER_KEY', api_mode: 'chat_completions', request_overrides: { extra_body: { tier: 'batch' } }
}

describe('delegation model settings contract', () => {
  it('round trips inherited state without any config write', () => {
    expect(delegationModelsPatch({}, readDelegationModels({}))).toEqual({})
  })

  it('preserves a model-only override and its inherited provider', () => {
    const baseline = { model: 'worker/model', provider: '' }
    expect(readDelegationModels(baseline)).toMatchObject(baseline)
    expect(delegationDraftValid(readDelegationModels(baseline), baseline)).toBe(true)
  })

  it('never commits a newly half-filled provider/model selection', () => {
    const draft = { ...readDelegationModels({}), provider: 'custom:new' }
    expect(delegationDraftValid(draft, {})).toBe(false)
    expect(() => delegationModelsPatch({}, draft)).toThrow()
  })

  it('commits both route fields atomically, not a hardcoded choice', () => {
    const draft = { ...readDelegationModels({}), provider: 'custom:any', model: 'arbitrary/model:tag' }
    expect(delegationModelsPatch({}, draft)).toEqual({ provider: draft.provider, model: draft.model })
  })

  it('never rewrites the primary conversation or unrelated delegation settings', () => {
    const baseline = { model: 'old', provider: 'custom:old', max_iterations: 17, reasoning_effort: 'high' }
    const patch = delegationModelsPatch(baseline, { ...readDelegationModels(baseline), model: 'new' })
    expect(patch).toEqual({ provider: 'custom:old', model: 'new' })
    expect(baseline.max_iterations).toBe(17)
    expect(baseline.reasoning_effort).toBe('high')
  })

  it('resets direct endpoint authority only after explicit provider selection', () => {
    const baseline = { model: 'old', provider: '', base_url: 'https://direct.invalid/v1', api_key: 'private-key' }
    const modelOnly = { ...readDelegationModels(baseline), model: 'new' }
    expect(delegationModelsPatch(baseline, modelOnly)).toEqual({ model: 'new', provider: '' })
    expect(delegationModelsPatch(baseline, { ...modelOnly, clearEndpoint: true })).toMatchObject({
      base_url: '', api_key: '', api_mode: '', request_overrides: null
    })
  })

  it('keeps canonical empty distinct from absence and suppresses legacy aliases', () => {
    expect(readDelegationModels({}).mode).toBe('auto')
    expect(readDelegationModels({ fallback_providers: [], fallback_model: worker }).mode).toBe('none')
    expect(readDelegationModels({ fallback_chain: [worker] }).rows).toEqual([worker])
  })

  it('requires an explicit opt-in before a pinned model shares main backups', () => {
    const baseline = { provider: 'custom:worker', model: 'pinned' }
    expect(readDelegationModels(baseline).mode).toBe('auto')
    expect(delegationModelsPatch(baseline, { ...readDelegationModels(baseline), mode: 'inherit' })).toEqual({
      fallback_providers: 'inherit', fallback_chain: null, fallback_model: null
    })
  })

  it('preserves endpoint and credential metadata when reordering backups', () => {
    const other = { ...worker, provider: 'custom:backup', model: 'backup' }
    const baseline = { fallback_providers: [worker, other] }
    const rows = moveDelegationFallback(readDelegationModels(baseline).rows, 1, -1)
    expect(delegationModelsPatch(baseline, { ...readDelegationModels(baseline), rows }).fallback_providers).toEqual([other, worker])
    expect(baseline.fallback_providers).toEqual([worker, other])
  })

  it('drops old auth only for a row whose provider is deliberately replaced', () => {
    const next = { provider: 'custom:another', model: '' }
    expect(updateDelegationFallback(worker, next, true)).toEqual(next)
    expect(updateDelegationFallback(worker, { provider: worker.provider, model: 'next' }, false)).toMatchObject({
      model: 'next', base_url: worker.base_url, key_env: worker.key_env, request_overrides: worker.request_overrides
    })
  })

  it('does not discard valid rows when an added backup is incomplete', () => {
    const baseline = { fallback_providers: [worker] }
    const draft = { ...readDelegationModels(baseline), rows: [worker, { provider: '', model: '' }] }
    expect(delegationDraftValid(draft, baseline)).toBe(false)
    expect(() => delegationModelsPatch(baseline, draft)).toThrow()
    expect(baseline.fallback_providers).toEqual([worker])
  })

  it('does not silently normalize malformed declarations into inheritance', () => {
    const invalid = readDelegationModels({ fallback_providers: ['broken'] })
    expect(invalid.mode).toBe('invalid')
    expect(delegationDraftValid(invalid, {})).toBe(false)
  })

  it('clears aliases on explicit reset so an old backup cannot reappear', () => {
    const baseline = { fallback_chain: [worker] }
    const draft = { ...readDelegationModels(baseline), mode: 'auto' as const }
    expect(delegationModelsPatch(baseline, draft)).toEqual({
      fallback_providers: null, fallback_chain: null, fallback_model: null
    })
  })

  it('keeps no-op reorders referentially stable', () => {
    const rows = [worker]
    expect(moveDelegationFallback(rows, 0, -1)).toBe(rows)
    expect(moveDelegationFallback(rows, 0, 1)).toBe(rows)
  })
})
