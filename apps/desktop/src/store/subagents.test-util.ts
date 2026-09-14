import type { SubagentSnapshot } from '@hermes/shared'

import type { SubagentPayload } from './subagents'

/** A whole `subagent.*` frame: the generated payload sends every key, so tests
 *  state only the fields under test and inherit the rest. */
export const subagentEvent = (over: Partial<SubagentPayload> = {}): SubagentPayload => ({
  goal: '',
  task_count: 1,
  task_index: 0,
  subagent_id: null,
  parent_id: null,
  child_session_id: null,
  delegation_id: null,
  depth: null,
  model: null,
  tool_count: null,
  toolsets: null,
  input_tokens: null,
  output_tokens: null,
  reasoning_tokens: null,
  api_calls: null,
  files_read: null,
  files_written: null,
  output_tail: null,
  tool_name: null,
  text: null,
  status: null,
  summary: null,
  duration_seconds: null,
  tool_preview: null,
  ...over
})

/** One `subagent.list` roster row. */
export const subagentRosterRow = (over: Partial<SubagentSnapshot> = {}): SubagentSnapshot => ({
  subagent_id: 'worker',
  parent_id: null,
  depth: 0,
  goal: '',
  delegation_id: null,
  model: null,
  started_at: 0,
  status: 'running',
  tool_count: 0,
  last_tool: null,
  accepting_steer: true,
  ...over
})
