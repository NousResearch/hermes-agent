import { beforeEach, describe, expect, it } from 'vitest'

import type { DesktopUpdateProgress, DesktopUpdateStage } from '@/global'

import { $updateApply, __ingestProgressForTests, resetUpdateApplyState } from '../updates'

function progress(stage: DesktopUpdateStage, overrides: Partial<DesktopUpdateProgress> = {}): DesktopUpdateProgress {
  return {
    stage,
    message: overrides.message ?? `stage:${stage}`,
    percent: overrides.percent ?? null,
    error: overrides.error ?? null,
    at: overrides.at ?? 0
  }
}

describe('updates store / ingestProgress', () => {
  beforeEach(() => {
    resetUpdateApplyState()
    // Seed an "applying" state so we can verify terminal stages clear it.
    __ingestProgressForTests(progress('prepare'))
    expect($updateApply.get().applying).toBe(true)
  })

  it("clears applying when stage is 'done' (Linux/AppImage finishing path)", () => {
    __ingestProgressForTests(progress('done', { message: 'Backend updated.', percent: 100 }))

    const state = $updateApply.get()
    expect(state.applying).toBe(false)
    expect(state.stage).toBe('done')
  })

  it.each(['restart', 'error', 'manual'] as const)("clears applying when stage is '%s'", (stage) => {
    __ingestProgressForTests(progress(stage))
    expect($updateApply.get().applying).toBe(false)
  })

  it.each(['prepare', 'fetch', 'pull', 'pydeps'] as const)("keeps applying true for intermediate stage '%s'", (stage) => {
    __ingestProgressForTests(progress(stage))
    expect($updateApply.get().applying).toBe(true)
  })
})
