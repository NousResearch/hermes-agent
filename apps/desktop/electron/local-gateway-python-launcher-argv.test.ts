import type * as ChildProcessAPI from 'node:child_process'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { afterEach, expect, test, vi } from 'vitest'

const spawned = vi.hoisted(() => [] as { command: string; args: string[]; options: any }[])

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcessAPI>()

  return { ...actual, spawn: (command: string, args: string[], options: any) => {
    spawned.push({ command, args, options })
    const child = Object.assign(new EventEmitter(), { stdout: new PassThrough(), kill: () => true })
    setImmediate(() => {
      child.stdout.end(JSON.stringify([process.execPath, '-I', '-c', 'bootstrap']))
      child.emit('close', 0)
    })

    return child
  } }
})

const { ticketRuntimeFor } = await import('./local-gateway-python')

const realPlatform = process.platform

afterEach(() => {
  Object.defineProperty(process, 'platform', { value: realPlatform })
  spawned.length = 0
})

// RT-10 (#138774): a HERMES_DESKTOP_HERMES=...\hermes.cmd override is a shell backend. With
// shell:true Node hands cmd.exe one unquoted line, which a profile path with a space splits at
// `C:\Users\John` (#74064). The runtime-command probe must quote the launcher like every other
// shell spawn (windowsShellCommand), and keep the probe flag a separate argument.
test('a .cmd override in a path with spaces asks its launcher for the runtime as one quoted token', async () => {
  Object.defineProperty(process, 'platform', { value: 'win32' })
  const command = 'C:\\Users\\John Doe\\AppData\\Local\\hermes\\bin\\hermes.cmd'
  const root = 'C:\\Users\\John Doe\\AppData\\Local\\hermes\\hermes-agent'

  await expect(ticketRuntimeFor({ command, kind: 'command', shell: true, root }))
    .resolves.toEqual({ command: process.execPath, env: { PYTHONPATH: root } })
  expect(spawned).toHaveLength(1)
  expect(spawned[0].command).toBe(`"${command}"`)
  expect(spawned[0].args).toEqual(['--print-runtime-command'])
  expect(spawned[0].options).toMatchObject({ shell: true, cwd: root, windowsHide: true })
})

test('a launcher spawned without a shell keeps its bare path', async () => {
  Object.defineProperty(process, 'platform', { value: 'win32' })
  const command = 'C:\\Users\\John Doe\\AppData\\Local\\hermes\\bin\\hermes.exe'

  await ticketRuntimeFor({ command, kind: 'command', shell: false, root: 'C:\\r' })
  expect(spawned[0].command).toBe(command)
  expect(spawned[0].options).toMatchObject({ shell: false })
})
