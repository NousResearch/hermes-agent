import { Terminal } from '@xterm/xterm'
import { describe, expect, it } from 'vitest'

import { lineDisciplineWriter, terminalLineOptions } from './pty-output'

const write = (term: Terminal, data: string) => new Promise<void>(resolve => term.write(data, resolve))
const flush = (term: Terminal) => write(term, '')
const row = (term: Terminal, y: number) => term.buffer.active.getLine(y)?.translateToString(true) ?? ''
const make = (pty: boolean) => new Terminal({ ...terminalLineOptions(pty), allowProposedApi: true, cols: 40, rows: 6 })

describe('PTY terminal line discipline', () => {
  it('keeps the column on a bare LF, as a cursor-addressed redraw expects', async () => {
    const term = make(true)

    await write(term, '\x1b[2;3H\x1b[K\n')

    expect(term.buffer.active.cursorX).toBe(2)
    term.dispose()
  })

  it('a redraw of indented rows lands on the indent, as Claude Code emits it', async () => {
    const term = make(true)

    // Previous frame: two indented paragraph rows.
    await write(term, '\x1b[1;3HThe old line\x1b[2;3HWhat was here')
    // Captured repaint shape: address the indent, clear, bare LF, text, clear.
    await write(term, '\x1b[1;3H\x1b[K\nfresh\x1b[K')

    expect(row(term, 1)).toBe('  fresh')
    term.dispose()
  })

  it('a CR and LF split across chunks still make exactly one new line', async () => {
    const term = make(true)

    await write(term, 'one\r')
    await write(term, '\ntwo')

    expect([row(term, 0), row(term, 1), row(term, 2)]).toEqual(['one', 'two', ''])
    term.dispose()
  })

  it('pipe output (bare LF = new line) starts every line at column 1', async () => {
    const term = make(false)

    await write(term, 'one\ntwo')

    expect([row(term, 0), row(term, 1)]).toEqual(['one', 'two'])
    term.dispose()
  })
})

describe('agent terminal line discipline', () => {
  it('switches to PTY discipline when the flag arrives after the terminal opened', async () => {
    const term = make(false)
    const writeChunk = lineDisciplineWriter(term, false)

    writeChunk('$ claude\r\n', false)
    writeChunk('\x1b[3;3HThe old line\x1b[3;3H\x1b[K\nfresh', true)
    await flush(term)

    expect(row(term, 3)).toBe('  fresh')
    term.dispose()
  })

  it('leaves an LNM mode the program set itself alone while the flag is unchanged', async () => {
    const term = make(true)
    const writeChunk = lineDisciplineWriter(term, true)

    writeChunk('\x1b[20h', true)
    writeChunk('one\ntwo', true)
    await flush(term)

    expect([row(term, 0), row(term, 1)]).toEqual(['one', 'two'])
    term.dispose()
  })
})
