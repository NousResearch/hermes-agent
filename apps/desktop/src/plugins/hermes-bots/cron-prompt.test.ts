/**
 * What a routine actually runs.
 *
 * A job scheduled for the ACTIVE bot profile runs its instruction directly. A
 * job for a different profile keeps the `hermes -p <bot> chat` delegation
 * wrapper so the run lands in that bot's own history — and that wrapper is a
 * shell command line, which is why every operand is single-quoted rather than
 * interpolated. The pre-hardening prompts were built by interpolation, so a
 * routine title or instruction containing `$(…)` executed on the scheduler's
 * machine; `isLegacyDelegatedRoutine` is how those persisted jobs are still
 * recognized and paused.
 */

import { spawnSync } from 'node:child_process'

import { describe, expect, it } from 'vitest'

import { isLegacyDelegatedRoutine, normalizedProfileName, routineInputError, routinePrompt } from './cron'

/** Run the delegation command under a `hermes` stub that prints its argv, so
 *  the assertion is what the SHELL passed — not what the string looks like. */
function argvOf(prompt: string): string[] {
  const command = prompt.slice(prompt.indexOf('hermes '), prompt.lastIndexOf('\n\nIf the command'))
  const result = spawnSync('sh', ['-c', `hermes() { printf '%s\\037' "$@"; }\n${command}`], { encoding: 'utf8' })

  expect(result.status, result.stderr).toBe(0)

  return result.stdout.split('\u001f').slice(0, -1)
}

describe('direct vs delegated execution', () => {
  it('runs the bare instruction when the routine owns the active profile', () => {
    expect(normalizedProfileName(' Default ')).toBe('default')
    expect(routinePrompt('default', 'Health', 'Collect status', ' DEFAULT ')).toBe('Collect status')
  })

  it('keeps the delegation wrapper for a different active profile', () => {
    const prompt = routinePrompt('research', 'Digest', 'Summarize findings', 'default')

    expect(prompt).toMatch(/hermes -p 'research' chat/)
    expect(prompt).toMatch(/\[Scheduled routine\] Summarize findings/)
  })

  it('never re-wraps a direct prompt', () => {
    const instruction = 'Keep "quoted" output intact'

    expect(routinePrompt('ops', 'Check', instruction, 'ops')).toBe(instruction)
  })
})

describe('delegated arguments stay literal shell values', () => {
  it('passes substitutions, backticks and quotes through as text', () => {
    const title = "Audit $(printf TITLE_EXPANDED) `printf TITLE_TICK` 'quoted'"
    const instruction = "Line one $(printf TASK_EXPANDED) `printf TASK_TICK`\nLine two 'quoted'"

    expect(argvOf(routinePrompt('research', title, instruction, 'default'))).toEqual([
      '-p',
      'research',
      'chat',
      '-c',
      `Routine: ${title}`,
      '-q',
      `[Scheduled routine] ${instruction}`
    ])
  })

  it('a recreated routine is no longer recognized as the unsafe legacy shape', () => {
    // The v2 marker is what tells the loader this prompt was composed with
    // quoting, so it must NOT be swept into the security pause.
    const recreated = routinePrompt('research', 'Audit', 'Inspect', 'default')

    expect(
      isLegacyDelegatedRoutine({ job_id: 'x', name: '[bot:research] Audit', prompt_preview: recreated.slice(0, 100) })
    ).toBe(false)
  })

  it('recognizes a persisted pre-hardening prompt', () => {
    const legacy = 'You are running the scheduled routine "Audit" for agent \'research\'.'

    expect(isLegacyDelegatedRoutine({ job_id: 'x', name: '[bot:research] Audit', prompt_preview: legacy })).toBe(true)
    // Untagged jobs are not Bot Mode's to pause, whatever their prompt says.
    expect(isLegacyDelegatedRoutine({ job_id: 'x', name: 'Audit', prompt_preview: legacy })).toBe(false)
  })
})

describe('input the gateway cannot store', () => {
  it('rejects NUL before cron creation, naming the offending field', () => {
    expect(routineInputError('Normal title', 'Normal instruction')).toBeNull()
    expect(routineInputError('Bad\0title', 'Normal instruction')).toMatch(/NUL.*U\+0000/)
    expect(routineInputError('Normal title', 'Bad\0instruction')).toMatch(/NUL.*U\+0000/)
  })
})
