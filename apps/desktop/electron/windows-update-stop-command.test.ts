import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { promisify } from 'node:util'

import { expect, test } from 'vitest'

import { windowsPowerShellExecutable } from './windows-powershell-path'
import { formatWindowsHolderStopCommand } from './windows-update-force-release'

const execFileAsync = promisify(execFile)

test.skipIf(process.platform !== 'win32')('the copied command ignores a mismatched identity and stops the exact disposable process', async () => {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { windowsHide: true, stdio: 'ignore' })
  const exited = once(child, 'exit')
  const run = (command: string) => execFileAsync(windowsPowerShellExecutable(), ['-NoProfile', '-NonInteractive', '-Command', command], { windowsHide: true, timeout: 15_000 })

  try {
    await once(child, 'spawn')
    const { stdout } = await run(`(Get-Process -Id ${child.pid}).StartTime.ToUniversalTime().Subtract([datetime]'1970-01-01').TotalSeconds | ConvertTo-Json -Compress`)
    const createdAt = Number(stdout.trim())
    expect(createdAt).toBeGreaterThan(0)
    await run(formatWindowsHolderStopCommand([{ pid: child.pid!, createdAt: createdAt - 10 }])!)
    expect(child.exitCode).toBeNull()
    expect(() => process.kill(child.pid!, 0)).not.toThrow()
    await run(formatWindowsHolderStopCommand([{ pid: child.pid!, createdAt }])!)
    await exited
    expect(child.exitCode).not.toBeNull()
  } finally {
    if (child.exitCode === null) { child.kill() }
    await exited
  }
}, 45_000)
