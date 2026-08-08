import { type ChildProcess, spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { describe, expect, test } from 'vitest'

import { gracefulReleaseSessions, pathContains } from './project-path-release'

describe('project path release', () => {
  test('path containment does not confuse sibling prefixes', () => {
    expect(pathContains('C:\\work\\project', 'C:\\work\\project\\child')).toBe(true)
    expect(pathContains('C:\\work\\project', 'C:\\work\\project-old')).toBe(false)
  })

  test('release targets only local sessions under the requested project', async () => {
    let active = true
    let exits = 0

    const result = await gracefulReleaseSessions('C:\\work\\project', [
      { id: 'match', launchCwd: 'C:\\work\\project', remote: false, writeExit: () => { exits += 1; active = false }, isActive: () => active },
      { id: 'other', launchCwd: 'C:\\work\\other', remote: false, writeExit: () => { exits += 1 }, isActive: () => true },
      { id: 'remote', launchCwd: 'C:\\work\\project', remote: true, writeExit: () => { exits += 1 }, isActive: () => true },
    ])

    expect(result.released).toBe(true)
    expect(result.releasedTerminalIds).toEqual(['match'])
    expect(exits).toBe(1)
  })

  test('release fails closed when a terminal does not exit', async () => {
    let clock = 0

    const result = await gracefulReleaseSessions(
      'C:\\work\\project',
      [{ id: 'busy', launchCwd: 'C:\\work\\project', remote: false, writeExit: () => {}, isActive: () => true }],
      { now: () => clock, pause: async ms => { clock += ms }, timeoutMs: 100 },
    )

    expect(result.released).toBe(false)
    expect(result.activeTerminalIds).toEqual(['busy'])
  })

  test.runIf(process.platform === 'win32')('releases a real child cwd before the directory is renamed', async () => {
    const runtimeRoot = path.resolve('.hermes', 'task-runtime')
    fs.mkdirSync(runtimeRoot, { recursive: true })
    const root = fs.mkdtempSync(path.join(runtimeRoot, 'project-release-'))
    const project = path.join(root, 'project')
    const archived = path.join(root, 'archived')
    fs.mkdirSync(project)
    let child: ChildProcess | null = null
    let active = true

    try {
      child = spawn(
        process.execPath,
        ['-e', "process.stdin.resume(); process.stdin.on('data', () => process.exit(0))"],
        { cwd: project, stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true },
      )
      child.once('exit', () => { active = false })
      await new Promise<void>((resolve, reject) => {
        child?.once('spawn', resolve)
        child?.once('error', reject)
      })

      const result = await gracefulReleaseSessions(project, [{
        id: 'real-child',
        launchCwd: project,
        remote: false,
        writeExit: () => child?.stdin?.write('exit\n'),
        isActive: () => active,
      }])

      expect(result).toEqual({ released: true, releasedTerminalIds: ['real-child'] })
      fs.renameSync(project, archived)
      expect(fs.existsSync(archived)).toBe(true)
    } finally {
      if (active) {child?.kill()}
      fs.rmSync(root, { force: true, recursive: true })
    }
  })
})
