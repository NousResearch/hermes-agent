import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { createRecoveryController } from './recovery-controller'

test('blocks startup until the stopped backend has been snapshotted, then releases before restart', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-gate-'))

  try {
    const home = path.join(dir, 'home')
    const app = path.join(dir, 'app')
    fs.mkdirSync(home)
    fs.mkdirSync(app)
    fs.writeFileSync(path.join(app, 'Czesiek.exe'), 'binary')
    fs.writeFileSync(path.join(home, 'MEMORY.md'), 'memory')
    let stopped!: (value: { unlocked: boolean }) => void

    const stop = vi.fn(
      () =>
        new Promise<{ unlocked: boolean }>(resolve => {
          stopped = resolve
        })
    )

    const restart = vi.fn(async () => {
      await controller.wait()
    })

    const controller = createRecoveryController({
      root: path.join(dir, 'recovery'),
      home,
      executable: path.join(app, 'Czesiek.exe'),
      version: 'test',
      supported: true,
      stop,
      restart,
      quit: vi.fn()
    })

    const preparation = controller.prepare()
    let released = false

    const waiting = controller.wait().then(() => {
      released = true
    })

    await Promise.resolve()
    expect(released).toBe(false)
    await expect(controller.prepare()).rejects.toThrow('Trwa już')
    stopped({ unlocked: true })
    await preparation
    await waiting
    expect(controller.status().available).toBe(true)
    expect(restart).toHaveBeenCalledOnce()
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
