import { describe, expect, it } from 'vitest'

import { getSessionStatusLabels } from './session-status-labels'

const baseSession = {
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'session-1',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0
}

describe('getSessionStatusLabels', () => {
  it('derives queued steer labels from counts only', () => {
    const evidence = {
      ...baseSession,
      queued_steer_count: 2,
      queued_steer_text: 'do not leak this queued prompt'
    }

    const labels = getSessionStatusLabels(evidence)

    expect(labels.map(label => label.label)).toContain('Queued steer')
    expect(JSON.stringify(labels)).not.toContain('do not leak')
  })

  it('derives terminal recovery labels from safe status evidence', () => {
    const evidence = {
      ...baseSession,
      model_request_status: 'terminal_recovery_needed',
      model_request_status_message: 'raw recovery diagnostic should not leak'
    } as Parameters<typeof getSessionStatusLabels>[0] & { model_request_status_message: string }

    const labels = getSessionStatusLabels(evidence)

    expect(labels.map(label => label.label)).toContain('Terminal recovery')
    expect(JSON.stringify(labels)).not.toContain('raw recovery')
  })

  it('derives high-context waiting labels without provider payload text', () => {
    const evidence = {
      ...baseSession,
      model_request_high_context: true,
      model_request_status: 'waiting',
      provider_payload: 'do not leak provider payload'
    } as Parameters<typeof getSessionStatusLabels>[0] & { provider_payload: string }

    const labels = getSessionStatusLabels(evidence)

    expect(labels.map(label => label.label)).toContain('High-context wait')
    expect(JSON.stringify(labels)).not.toContain('provider payload')
  })

  it('distinguishes queued steer blocked until tool boundary', () => {
    const evidence = {
      ...baseSession,
      model_request_queued_steer_count: 1,
      queued_steer_text: 'do not leak boundary prompt',
      steer_boundary: 'cannot_steer_until_current_tool_boundary'
    } as Parameters<typeof getSessionStatusLabels>[0] & { queued_steer_text: string }

    const labels = getSessionStatusLabels(evidence)

    expect(labels.map(label => label.label)).toContain('Steer waiting on tool')
    expect(labels.map(label => label.label)).not.toContain('Queued steer')
    expect(JSON.stringify(labels)).not.toContain('boundary prompt')
  })

  it('derives fixed-model policy violation labels from safe model fields', () => {
    const evidence = {
      ...baseSession,
      model_policy_violation: true,
      provider_payload: 'do not leak policy payload',
      required_model: 'gpt-5.5'
    } as Parameters<typeof getSessionStatusLabels>[0] & { provider_payload: string }

    const labels = getSessionStatusLabels(evidence)

    expect(labels.map(label => label.label)).toContain('Model policy')
    expect(JSON.stringify(labels)).toContain('gpt-5.5')
    expect(JSON.stringify(labels)).not.toContain('policy payload')
  })

  it('derives compression tip labels from safe lineage evidence', () => {
    expect(
      getSessionStatusLabels({
        ...baseSession,
        compression_tip_session_id: 'session-2'
      }).map(label => label.label)
    ).toContain('Compression tip')
  })

  it('derives runtime-active labels from active registry evidence even when DB active is stale', () => {
    expect(
      getSessionStatusLabels({
        ...baseSession,
        status_evidence_source: ['db_row', 'active_session_registry']
      }).map(label => label.label)
    ).toContain('Stale DB counter, runtime active')
  })

  it('derives running tool labels from value-free runtime events', () => {
    expect(
      getSessionStatusLabels({
        ...baseSession,
        last_tool_runtime_event: 'tool.start'
      }).map(label => label.label)
    ).toContain('Running tool')
  })

  it('derives monitor-blocked labels from future safe source tokens', () => {
    expect(
      getSessionStatusLabels({
        ...baseSession,
        status_evidence_source: ['blocked_by_monitor']
      }).map(label => label.label)
    ).toContain('Blocked by monitor evidence')
  })

  it('returns no labels for legacy rows without status evidence', () => {
    expect(getSessionStatusLabels(baseSession)).toEqual([])
  })
})
