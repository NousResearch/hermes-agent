import { describe, expect, it } from 'vitest'

import { summarizeToolRun, type ToolCallLike } from './run-summary'

function tool(toolName: string, args: Record<string, unknown> = {}, result?: unknown): ToolCallLike {
  return { args, result, toolCallId: `${toolName}-${Math.random()}`, toolName }
}

const edited = (path: string, diff = '') => tool('write_file', { path }, { path, inline_diff: diff })
const read = (path: string) => tool('read_file', { path }, { content: '' })
const ran = (command: string) => tool('terminal', { command }, { exit_code: 0 })

describe('summarizeToolRun', () => {
  it('names a lone edit and counts the rest', () => {
    expect(
      summarizeToolRun([edited('src/use-preview-routing.ts'), read('a.ts'), read('b.ts'), read('c.ts')]).text
    ).toBe('Edited use-preview-routing.ts, explored 3 files')
  })

  it('orders clauses edit, explore, run regardless of call order', () => {
    expect(
      summarizeToolRun([ran('ls'), read('a.ts'), edited('src/attachments.tsx'), read('b.ts'), ran('pwd'), ran('id')])
        .text
    ).toBe('Edited attachments.tsx, explored 2 files, ran 3 commands')
  })

  it('counts commands rather than naming them once they have run', () => {
    expect(summarizeToolRun([ran('git status')]).text).toBe('Ran 1 command')
    expect(summarizeToolRun([read('status.ts'), ran('a'), ran('b'), ran('c'), ran('d'), ran('e')]).text).toBe(
      'Explored status.ts, ran 5 commands'
    )
  })

  it('counts a multi-file edit', () => {
    expect(summarizeToolRun([edited('a.tsx'), edited('b.tsx'), read('c.ts')]).text).toBe(
      'Edited 2 files, explored c.ts'
    )
  })

  it('puts the running category in the present tense and leaves the rest past', () => {
    const live = [edited('a.tsx'), tool('write_file', { path: 'b.tsx' }), ran('x'), ran('y')]

    expect(summarizeToolRun(live).text).toBe('Editing 2 files, ran 2 commands')
  })

  it('names the command that is still running', () => {
    expect(summarizeToolRun([tool('terminal', { command: 'npm run typecheck' })]).text).toMatch(/^Running /)
  })

  it('sums diff stats across the edits in the run', () => {
    const summary = summarizeToolRun([
      edited('a.tsx', '--- a\n+++ b\n+one\n+two\n-old'),
      edited('b.tsx', '+three'),
      ran('ls')
    ])

    expect(summary).toMatchObject({ added: 3, removed: 1 })
  })

  it('reports no diff stats for a run that changed nothing', () => {
    expect(summarizeToolRun([read('a.ts'), ran('ls')])).toMatchObject({ added: 0, removed: 0 })
  })
})
