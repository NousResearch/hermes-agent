import { describe, expect, it } from 'vitest'

import { preprocessMarkdown } from './markdown-preprocess'

describe('preprocessMarkdown transcript-imitation repair', () => {
  it('fences a prose "Assistant called tool" block so the JSON payload renders as code', () => {
    const input =
      'Checking the worktree now.\n\n' +
      'Assistant called tool terminal (call_mt11k2gh_9p0z3mn) with arguments: {"command":"cd $HOME/.hermes && ls","timeout":30}\n\n' +
      'Tool result (call_mt11k2gh_9p0z3mn): {"output":"ok"}\n\n' +
      'User: continue'

    const out = preprocessMarkdown(input)

    expect(out).toContain('```json\n{"command":"cd $HOME/.hermes && ls","timeout":30}\n```')
    expect(out).toContain('```text\n{"output":"ok"}\n```')
    expect(out).toContain('Assistant called tool terminal (call_mt11k2gh_9p0z3mn) with arguments:')
  })

  it('leaves ordinary prose about tool calls untouched', () => {
    const input = 'I called the tool earlier and it worked. Arguments matter here.'

    expect(preprocessMarkdown(input)).toBe(input)
  })

  it('strips leaked chat-template special tokens', () => {
    const input = 'done ' + '<' + '|' + 'close' + '|' + '>' + 'argument' + '<' + '|' + 'sep' + '|' + '>' + ' trailing'

    expect(preprocessMarkdown(input)).toBe('done argument trailing')
  })

  it('escapes shell env-var dollars so KaTeX never pairs them', () => {
    const input = 'Set $HOME and $PATH first, then ${HERMES_HOME}.'

    const out = preprocessMarkdown(input)

    expect(out).toContain('\\$HOME')
    expect(out).toContain('\\$PATH')
    expect(out).toContain('\\${HERMES_HOME}')
  })

  it('keeps real inline math intact', () => {
    const input = 'The eigenvalue satisfies $x^2 = 4$ here.'

    expect(preprocessMarkdown(input)).toContain('$x^2 = 4$')
  })
})
