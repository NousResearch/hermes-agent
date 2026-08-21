import { describe, expect, it } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import {
  formatCheckpointList,
  formatRollbackDiff,
  formatRollbackRestore,
  parseRollbackCommand,
  ROLLBACK_DIFF_LINE_LIMIT,
  trimLastExchange
} from './desktop-rollback'

const msg = (role: ChatMessage['role'], id: string = role): ChatMessage => ({ id, parts: [], role })

describe('parseRollbackCommand', () => {
  it('routes a bare command (and list/ls aliases) to rollback.list', () => {
    expect(parseRollbackCommand('')).toEqual({ kind: 'list' })
    expect(parseRollbackCommand('   ')).toEqual({ kind: 'list' })
    expect(parseRollbackCommand('list')).toEqual({ kind: 'list' })
    expect(parseRollbackCommand('ls')).toEqual({ kind: 'list' })
    expect(parseRollbackCommand('LIST')).toEqual({ kind: 'list' })
  })

  it('routes diff to rollback.diff and demands a checkpoint', () => {
    expect(parseRollbackCommand('diff 2')).toEqual({ hash: '2', kind: 'diff' })
    expect(parseRollbackCommand('DIFF abc123')).toEqual({ hash: 'abc123', kind: 'diff' })
    expect(parseRollbackCommand('diff')).toEqual({
      kind: 'usage',
      message: 'usage: /rollback diff <checkpoint>'
    })
  })

  it('routes a checkpoint ref to rollback.restore, with optional file path', () => {
    expect(parseRollbackCommand('2')).toEqual({ filePath: '', hash: '2', kind: 'restore' })
    expect(parseRollbackCommand('abc123')).toEqual({ filePath: '', hash: 'abc123', kind: 'restore' })
    expect(parseRollbackCommand('2 src/foo.py')).toEqual({ filePath: 'src/foo.py', hash: '2', kind: 'restore' })
    expect(parseRollbackCommand('  3   path/with space.txt  ')).toEqual({
      filePath: 'path/with space.txt',
      hash: '3',
      kind: 'restore'
    })
  })

  it('never resolves to a slash.exec / unknown plan — every input maps to an RPC or a usage hint', () => {
    const kinds = ['', 'list', 'ls', 'diff', 'diff 2', '2', 'abc123', '2 file.py'].map(a => parseRollbackCommand(a).kind)

    expect(kinds.every(kind => ['diff', 'list', 'restore', 'usage'].includes(kind))).toBe(true)
  })
})

describe('formatCheckpointList', () => {
  it('reports when checkpoints are disabled', () => {
    expect(formatCheckpointList({ enabled: false })).toBe('checkpoints are not enabled')
  })

  it('reports an empty checkpoint set', () => {
    expect(formatCheckpointList({ checkpoints: [], enabled: true })).toBe('no checkpoints found')
  })

  it('numbers checkpoints and shows short hash + metadata', () => {
    const out = formatCheckpointList({
      checkpoints: [
        { hash: '0123456789abcdef', message: 'edited config', timestamp: '2026-06-07T12:00:00Z' },
        { hash: 'fedcba9876543210' }
      ],
      enabled: true
    })

    expect(out).toBe(
      ['Rollback checkpoints', '1. 0123456789  2026-06-07T12:00:00Z · edited config', '2. fedcba9876  (no metadata)'].join(
        '\n'
      )
    )
  })
})

describe('formatRollbackDiff', () => {
  it('reports no changes when stat and diff are both empty', () => {
    expect(formatRollbackDiff({})).toBe('no changes since this checkpoint')
    expect(formatRollbackDiff({ diff: '', stat: '  ' })).toBe('no changes since this checkpoint')
  })

  it('combines stat and the plain diff (ignoring the ANSI rendered field)', () => {
    const out = formatRollbackDiff({ diff: '- old\n+ new', rendered: '[31m- old[0m', stat: '1 file changed' })

    expect(out).toBe('1 file changed\n\n- old\n+ new')
  })

  it('caps long diffs at the line limit and notes the remainder', () => {
    const lines = Array.from({ length: ROLLBACK_DIFF_LINE_LIMIT + 5 }, (_, i) => `+ line ${i}`)
    const out = formatRollbackDiff({ diff: lines.join('\n') })
    const outLines = out.split('\n')

    expect(outLines).toHaveLength(ROLLBACK_DIFF_LINE_LIMIT + 1)
    expect(outLines.at(-1)).toBe(`… 5 more lines (showing first ${ROLLBACK_DIFF_LINE_LIMIT})`)
  })
})

describe('formatRollbackRestore', () => {
  it('surfaces failures with the best available reason', () => {
    expect(formatRollbackRestore({ error: 'bad hash', success: false }, '')).toBe('rollback failed: bad hash')
    expect(formatRollbackRestore({ success: false }, '')).toBe('rollback failed: unknown error')
  })

  it('reports a full-workspace restore', () => {
    expect(formatRollbackRestore({ reason: 'reset to abc123', success: true }, '')).toBe(
      'rollback restored workspace: reset to abc123'
    )
  })

  it('reports a file-scoped restore', () => {
    expect(formatRollbackRestore({ restored_to: 'abc123', success: true }, 'src/foo.py')).toBe(
      'rollback restored src/foo.py: abc123'
    )
  })
})

describe('trimLastExchange', () => {
  it('drops trailing assistant/tool messages plus the user turn that started them', () => {
    const messages = [
      msg('user', 'u1'),
      msg('assistant', 'a1'),
      msg('user', 'u2'),
      msg('tool', 't1'),
      msg('assistant', 'a2')
    ]

    expect(trimLastExchange(messages).map(m => m.id)).toEqual(['u1', 'a1'])
  })

  it('leaves a trailing system message (e.g. a prior slash note) and its turn intact', () => {
    const messages = [msg('user', 'u1'), msg('assistant', 'a1'), msg('system', 's1')]

    expect(trimLastExchange(messages).map(m => m.id)).toEqual(['u1', 'a1', 's1'])
  })

  it('does not mutate the input and tolerates an empty transcript', () => {
    const messages = [msg('user'), msg('assistant')]
    const result = trimLastExchange(messages)

    expect(messages).toHaveLength(2)
    expect(result).toHaveLength(0)
    expect(trimLastExchange([])).toEqual([])
  })
})
