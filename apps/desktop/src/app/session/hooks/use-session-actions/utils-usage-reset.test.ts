import { beforeEach, describe, expect, it } from 'vitest'

import { $currentUsage, setCurrentCwd, setCurrentUsage } from '@/store/session'

import { applyStoredSessionPreviewRuntimeInfo } from './utils'

// The previous conversation's status-bar figures, as painted by its last turn:
// a 87% prompt-cache hit rate and 42 t/s of rolling throughput.
const PREVIOUS_SESSION_USAGE = {
  avg_tps: 42,
  cache_hit_pct: 87,
  calls: 3,
  compressions: 2,
  input: 100,
  output: 20,
  total: 120
}

// #146430: `$currentUsage` merges instead of replacing (`mergeUsage`), and nothing repaints
// the atom on a session switch, so the two per-session status-bar meters kept showing the
// PREVIOUS conversation's numbers for as long as the newly opened session ran no turn. The
// values themselves are per-session and durable — the displayed number was just false.
describe('applyStoredSessionPreviewRuntimeInfo session-scoped usage reset', () => {
  beforeEach(() => {
    setCurrentCwd('/project-a')
    setCurrentUsage({ ...PREVIOUS_SESSION_USAGE })
  })

  it('drops the previous session cache-hit rate and throughput when the selection changes', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '/project-b', model: 'gpt' }, 'session-b')

    expect($currentUsage.get().cache_hit_pct).toBeUndefined()
    expect($currentUsage.get().avg_tps).toBeUndefined()
  })

  // Only the derived rates are session-identified: the token counters keep the values the
  // cold-resume path repaints from the stored row a moment later.
  it('leaves the token counters for the stored-row repaint', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '/project-b', model: 'gpt' }, 'session-b')

    expect($currentUsage.get()).toMatchObject({ calls: 3, input: 100, output: 20, total: 120 })
  })

  // A same-session re-resume (retry, profile swap) has nothing stale to drop, and blanking
  // the meters there would be the inverse defect — the `—` bug class of #140011.
  it('keeps the meters when the session being resumed is the one already painted', () => {
    applyStoredSessionPreviewRuntimeInfo({ cwd: '/project-a', model: 'gpt' }, 'session-a', { sameSession: true })

    expect($currentUsage.get()).toMatchObject({ avg_tps: 42, cache_hit_pct: 87 })
  })
})
