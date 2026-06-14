import { describe, expect, it } from 'vitest'

import {
  isDraftTextVisible,
  layoutSyntaxDecorationSegments,
  TerminalCommandDraft,
  tokenizeShellCommand
} from './syntax-highlighting'

describe('tokenizeShellCommand', () => {
  it('highlights commands, options, strings, operators, variables, paths, and comments', () => {
    const tokens = tokenizeShellCommand('FOO=bar python ./script.py --flag=$HOME | grep "ok" # done')

    expect(tokens.map(token => [token.kind, token.text])).toEqual([
      ['assignment', 'FOO=bar'],
      ['command', 'python'],
      ['path', './script.py'],
      ['option', '--flag'],
      ['variable', '$HOME'],
      ['operator', '|'],
      ['command', 'grep'],
      ['string', '"ok"'],
      ['comment', '# done']
    ])
  })

  it('marks a command after a control operator as a new command', () => {
    const tokens = tokenizeShellCommand('git status && npm run build')

    expect(tokens.filter(token => token.kind === 'command').map(token => token.text)).toEqual(['git', 'npm'])
    expect(tokens.some(token => token.kind === 'operator' && token.text === '&&')).toBe(true)
  })
})

describe('TerminalCommandDraft', () => {
  it('tracks printable input, cursor movement, insertion, and deletion', () => {
    const draft = new TerminalCommandDraft()

    draft.applyUserInput('git')
    draft.applyUserInput('\x1b[D')
    draft.applyUserInput('u')
    draft.applyUserInput('\x7f')
    draft.applyUserInput('s')

    expect(draft.text).toBe('gist')
    expect(draft.cursor).toBe(3)
  })

  it('clears the draft on submit and shell-history navigation', () => {
    const draft = new TerminalCommandDraft()

    draft.applyUserInput('npm test')
    draft.applyUserInput('\r')
    expect(draft.text).toBe('')

    draft.applyUserInput('npm test')
    draft.applyUserInput('\x1b[A')
    expect(draft.text).toBe('')
  })

  it('ignores terminal escape wrappers without inserting protocol bytes', () => {
    const draft = new TerminalCommandDraft()

    draft.applyUserInput('\x1b[200~printf "ok"\x1b[201~')
    expect(draft.text).toBe('printf "ok"')

    draft.applyUserInput('\x1b[1;5D')
    expect(draft.text).toBe('printf "ok"')

    draft.applyUserInput('\x1bOA')
    draft.applyUserInput('\x1bOP')
    draft.applyUserInput('\x1bPignored\x1b\\')
    expect(draft.text).toBe('printf "ok"')
  })
})

describe('isDraftTextVisible', () => {
  it('confirms echoed draft text at the current cursor, including wrapped rows', () => {
    const lines = ['pytho', 'n    ']

    expect(
      isDraftTextVisible({
        cols: 5,
        cursor: 6,
        cursorX: 1,
        cursorY: 1,
        lineAt: row => lines[row] ?? null,
        text: 'python'
      })
    ).toBe(true)
  })

  it('rejects hidden input that is not echoed in the terminal buffer', () => {
    expect(
      isDraftTextVisible({
        cols: 20,
        cursor: 6,
        cursorX: 10,
        cursorY: 0,
        lineAt: () => 'Password: '.padEnd(20, ' '),
        text: 'secret'
      })
    ).toBe(false)
  })
})

describe('layoutSyntaxDecorationSegments', () => {
  it('splits highlighted tokens across wrapped terminal rows', () => {
    const segments = layoutSyntaxDecorationSegments({
      cols: 5,
      cursor: 6,
      cursorX: 1,
      cursorY: 2,
      tokens: [{ end: 6, kind: 'command', start: 0, text: 'python' }]
    })

    expect(segments).toEqual([
      { kind: 'command', rowOffset: -1, width: 5, x: 0 },
      { kind: 'command', rowOffset: 0, width: 1, x: 0 }
    ])
  })
}
)
