// Regression: the backend's /api/sessions/search payload already carries the
// real session title on every hit (web_routers/sessions.py add_lineage_result
// enriches via get_session_rich_row), but the sidebar search mapper dropped it
// (title: null). sessionTitle() then fell back to the preview — the FTS
// snippet of the matched message — painting rows with raw message content
// (tool JSON, shell commands) as the session name until the row loaded
// (Aug 2026).
import { describe, expect, it } from 'vitest'

import { searchResultToSession } from './index'

describe('searchResultToSession', () => {
  it('uses the backend-provided title for the synthesized row', () => {
    const s = searchResultToSession({
      lineage_root: 'root-1',
      model: 'deepseek-v4-flash',
      role: 'tool',
      session_id: '20260725_132',
      session_started: 1753450000,
      snippet: '{"bytes_written": 1879, "dirs_created": true}',
      source: 'agent',
      title: '  Fact-checking PR review comments  '
    })

    expect(s.title).toBe('Fact-checking PR review comments')
    expect(s.id).toBe('20260725_132')
  })

  it('keeps the FTS snippet as the preview with markers stripped', () => {
    const s = searchResultToSession({
      session_id: 'cron_f7da0e4',
      session_started: null,
      snippet: '...Report "PR #>>>67834<<< still in draft..."',
      role: 'user',
      model: null,
      source: 'cron',
      title: 'PR #67834 CI followup'
    })

    expect(s.title).toBe('PR #67834 CI followup')
    expect(s.preview).toBe('...Report "PR #67834 still in draft..."')
  })

  it('leaves title null when the backend sends none (untitled session)', () => {
    const s = searchResultToSession({
      session_id: '20260720_233',
      session_started: null,
      snippet: '[{"id": "call_d632e8dc7bf740048f242bfe", "c...',
      role: 'assistant',
      model: null,
      source: null
    })

    expect(s.title).toBeNull()
    expect(s.preview).toBe('[{"id": "call_d632e8dc7bf740048f242bfe", "c...')
  })
})
