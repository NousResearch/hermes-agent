import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

const driver = path.resolve(import.meta.dirname, '../../tests/install/e2e-assets/desktop-smoke.ts')

test('an app that dies before Playwright attaches leaves a failed receipt carrying its exit code and stderr', (): void => {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-smoke-launch-'))

  try {
    const dir = (name: string): string => {
      const target = path.join(work, name)
      fs.mkdirSync(target)

      return target
    }

    const out = path.join(work, 'out')

    // Node as the "app": it rejects Playwright's Chromium switches and exits before any debugger line.
    const result = spawnSync(process.execPath, [driver, '--exe', process.execPath, '--root', dir('root'), '--origin', 'source',
      '--home', dir('home'), '--user-data', dir('user-data'), '--out', out, '--phase', 'installed',
      '--expect-commit', '0'.repeat(40)], { cwd: work, encoding: 'utf8', timeout: 60_000 })

    expect(result.status).toBe(1)
    const receipt = JSON.parse(fs.readFileSync(path.join(out, 'desktop-chat-installed.json'), 'utf8'))
    expect(receipt.status).toBe('failed')
    expect(receipt.error).toContain('Process failed to launch!')
    expect(receipt.error).toMatch(/<process did exit: exitCode=[1-9]\d*/)
    expect(receipt.error).toMatch(/\]\[err\] .*bad option: --no-sandbox/)
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
})
