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
