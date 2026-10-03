import { expect, test, vi } from 'vitest'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const handlers = vi.hoisted(() => new Map<string, (...args: any[]) => any>())
vi.mock('electron', () => ({ app: { getVersion: () => 'test', getPath: () => process.env.HOME }, ipcMain: { handle: (name: string, handler: (...args: any[]) => any) => handlers.set(name, handler), on: () => {} } }))
vi.mock('node-pty', () => ({ default: {} }))
vi.mock('./pc-ipc', () => ({ registerPcIpc: () => {} }))
vi.mock('./remote-ipc', () => ({ registerRemoteIpc: () => {} }))
import { registerTerminalIpc } from './terminal-ipc'
test.runIf(process.platform === 'linux')('explicit Bash runs locally without Windows Git Bash configuration', async () => {
  registerTerminalIpc({ pcConnectionScope: () => 'test', isWindows: false, findOnPath: () => null, rememberLog: () => {}, activeSshTerminalTarget: () => null, sshBinary: () => '/usr/bin/ssh', ensureBackend: async () => undefined, getSshConnectionState: () => undefined })
  const result = await handlers.get('hermes:desktop:exec')!({}, { shell: 'bash', command: 'uname -s', cwd: '/tmp' })
  expect(result.success).toBe(true)
  expect(result.output.trim()).toBe('Linux')
  expect(result.shell).toBe('bash')
})

test.runIf(process.platform === 'win32' && Boolean(process.env.HERMES_GIT_BASH_PATH))('large Bash writes preserve bytes and leave stdin at EOF', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'hermes-bash-write-'))
  try {
    registerTerminalIpc({ pcConnectionScope: () => 'test', isWindows: true, findOnPath: () => null, rememberLog: () => {}, activeSshTerminalTarget: () => null, sshBinary: () => 'ssh', ensureBackend: async () => undefined, getSshConnectionState: () => undefined })
    const text = 'Unicode α; quotes " and dollar $ and backslash \\\n'.repeat(300)
    const encoded = Buffer.from(text).toString('base64')
    const result = await handlers.get('hermes:desktop:exec')!({}, { shell: 'bash', cwd: directory, command: `printf %s '${encoded}' | base64 -d > probe.txt\ncat >/dev/null\nprintf DONE` })
    expect(result.success).toBe(true)
    expect(result.output).toBe('DONE')
    expect(readFileSync(join(directory, 'probe.txt'), 'utf8')).toBe(text)
    const failure = await handlers.get('hermes:desktop:exec')!({}, { shell: 'bash', cwd: directory, command: 'exit 7' })
    expect(failure.returncode).toBe(7)
    expect(failure.success).toBe(false)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
