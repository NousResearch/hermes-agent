import { describe, expect, it } from 'vitest'

import { deriveSessionFileKeys, sessionFileKey } from './session-files'

const DIFF = '--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+hello'

function edit(toolName: string, args: Record<string, unknown>, result?: Record<string, unknown>) {
  return { type: 'tool-call', toolName, args, ...(result !== undefined && { result }) }
}

const message = (...parts: unknown[]) => ({ parts })

describe('sessionFileKey', () => {
  it('treats Windows paths as case- and slash-insensitive', () => {
    expect(sessionFileKey('C:\\Users\\Deb\\Notes\\Plan.md')).toBe(sessionFileKey('c:/users/deb/notes/plan.md'))
  })

  it('keeps POSIX paths case-sensitive and drops a trailing slash', () => {
    expect(sessionFileKey('/home/deb/Docs/')).toBe('/home/deb/Docs')
    expect(sessionFileKey('/home/deb/Docs')).not.toBe(sessionFileKey('/home/deb/docs'))
  })

  it('leaves the root folder alone', () => {
    expect(sessionFileKey('/')).toBe('/')
  })
})

describe('deriveSessionFileKeys', () => {
  it('collects files created or edited by write_file and patch', () => {
    const keys = deriveSessionFileKeys([
      message(edit('write_file', { path: 'new.md' }, { files_modified: ['/repo/new.md'], inline_diff: DIFF })),
      message(edit('patch', { path: '/repo/src/a.ts' }, { files_modified: ['/repo/src/a.ts'], inline_diff: DIFF }))
    ])

    expect([...keys].sort()).toEqual(['/repo/new.md', '/repo/src/a.ts'])
  })

  it('matches a Windows path however the tree spells it', () => {
    const keys = deriveSessionFileKeys([
      message(
        edit(
          'write_file',
          { path: 'test.md' },
          { files_modified: ['C:\\Users\\Deb\\Proj\\test.md'], inline_diff: DIFF }
        )
      )
    ])

    expect(keys.has(sessionFileKey('c:/users/deb/proj/TEST.md'))).toBe(true)
  })

  it('counts every file of a multi-file patch', () => {
    const keys = deriveSessionFileKeys([
      message(edit('patch', { mode: 'patch' }, { files_modified: ['/repo/a.ts', '/repo/b.ts'], inline_diff: DIFF }))
    ])

    expect(keys.size).toBe(2)
  })

  it('ignores running, failed and no-op edits', () => {
    const keys = deriveSessionFileKeys([
      message(
        edit('write_file', { path: '/repo/running.md' }),
        edit('write_file', { path: '/repo/failed.md' }, { error: 'denied' }),
        edit('patch', { path: '/repo/noop.md' }, { files_modified: ['/repo/noop.md'] })
      )
    ])

    expect(keys.size).toBe(0)
  })

  it('ignores tools that only read', () => {
    const keys = deriveSessionFileKeys([
      message(edit('read_file', { path: '/repo/a.md' }, { content: 'x', inline_diff: DIFF }))
    ])

    expect(keys.size).toBe(0)
  })

  it('takes the diff from tool-result metadata too', () => {
    const part = {
      ...edit('patch', { path: '/repo/a.ts' }, { files_modified: ['/repo/a.ts'] }),
      toolResultMetadata: { inline_diff: DIFF }
    }

    expect(deriveSessionFileKeys([message(part)]).has('/repo/a.ts')).toBe(true)
  })

  it('resolves a relative path against the workspace folder', () => {
    const keys = deriveSessionFileKeys(
      [message(edit('write_file', { path: './docs/plan.md' }, { inline_diff: DIFF }))],
      'C:\\Users\\Deb\\Proj'
    )

    expect(keys.has(sessionFileKey('C:\\Users\\Deb\\Proj\\docs\\plan.md'))).toBe(true)
  })

  it('lists a file once however often it was edited', () => {
    const write = edit('write_file', { path: '/repo/a.md' }, { files_modified: ['/repo/a.md'], inline_diff: DIFF })

    expect(deriveSessionFileKeys([message(write), message(write, write)]).size).toBe(1)
  })
})
