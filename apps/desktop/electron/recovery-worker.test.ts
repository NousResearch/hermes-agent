import { execFile, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { build } from 'esbuild'
import { expect, test } from 'vitest'

import { prepareRecovery, readRecovery } from './update-recovery'

test('bundled recovery worker waits for the app to exit and restores using real filesystem I/O', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-worker-'))

  try {
    const home = path.join(dir, 'home')
    const app = path.join(dir, 'app')
    const root = path.join(dir, 'recovery')
    fs.mkdirSync(home)
    fs.mkdirSync(app)
    const executable = path.join(app, path.basename(process.execPath))
    fs.copyFileSync(process.execPath, executable)
    fs.writeFileSync(path.join(home, 'MEMORY.md'), 'saved memory')
    await prepareRecovery({ home, installRoot: app, root, version: 'test', executable })
    fs.writeFileSync(path.join(home, 'MEMORY.md'), 'new memory')
    const worker = path.join(dir, 'worker.mjs')
    await build({
      entryPoints: [path.join(import.meta.dirname, 'recovery-worker.ts')],
      bundle: true,
      platform: 'node',
      format: 'esm',
      outfile: worker
    })
    const parent = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 600)'], { windowsHide: true, stdio: 'ignore' })
    await new Promise<void>((resolve, reject) => {
      parent.once('spawn', resolve)
      parent.once('error', reject)
    })
    await promisify(execFile)(process.execPath, [worker, root, String(parent.pid), app], {
      windowsHide: true,
      timeout: 20000
    })
    expect(fs.readFileSync(path.join(home, 'MEMORY.md'), 'utf8')).toBe('saved memory')
    expect(readRecovery(root)?.status).toBe('restored')
    expect(fs.existsSync(path.join(root, 'restore-error.txt'))).toBe(false)
  } finally {
    await fs.promises.rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 })
  }
}, 30000)
