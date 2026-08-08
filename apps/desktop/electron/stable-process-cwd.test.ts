import fs from 'node:fs'
import path from 'node:path'

import { afterEach, describe, expect, test } from 'vitest'

import { setStableProcessCwd } from './stable-process-cwd'

describe('stable process CWD', () => {
  const originalCwd = process.cwd()

  afterEach(() => process.chdir(originalCwd))

  test('desktop startup moves the parent process to the stable Hermes home', () => {
    const calls: string[] = []
    expect(setStableProcessCwd('C:\\HermesHome', value => calls.push(value))).toEqual({ changed: true, error: null })
    expect(calls).toEqual(['C:\\HermesHome'])
  })

  test('cwd setup reports failure without crashing startup', () => {
    const result = setStableProcessCwd('missing', () => { throw new Error('missing') })
    expect(result).toEqual({ changed: false, error: 'missing' })
  })

  test.runIf(process.platform === 'win32')('releases the inherited Windows launch directory', () => {
    const runtimeRoot = path.resolve('.hermes', 'task-runtime')
    fs.mkdirSync(runtimeRoot, { recursive: true })
    const root = fs.mkdtempSync(path.join(runtimeRoot, 'stable-cwd-'))
    const project = path.join(root, 'project')
    const archived = path.join(root, 'archived')
    const stable = path.join(root, 'hermes-home')
    fs.mkdirSync(project)
    fs.mkdirSync(stable)

    try {
      process.chdir(project)
      expect(setStableProcessCwd(stable)).toEqual({ changed: true, error: null })
      fs.renameSync(project, archived)
      expect(fs.existsSync(archived)).toBe(true)
    } finally {
      process.chdir(originalCwd)
      fs.rmSync(root, { force: true, recursive: true })
    }
  })
})
