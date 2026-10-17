import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, expect, test } from 'vitest'

import { ticketRuntimeFor } from './local-gateway-python'

let dir: string

beforeEach(async () => {
  dir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'ticket-runtime-')))
})
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

// A POSIX script stands in for the installation launcher's --print-runtime-command boundary.
async function launcher(body: string): Promise<string> {
  const file = path.join(dir, 'hermes')
  await fs.writeFile(file, `#!/bin/sh\nprintf '%s\\n' "$@" > "${dir}/argv"\n${body}\n`, { mode: 0o755 })

  return file
}

test('a Python backend is the helper interpreter as-is', async () => {
  const backend = { command: '/venv/bin/python', kind: 'python', shell: false, env: { A: '1' } }
  await expect(ticketRuntimeFor(backend)).resolves.toBe(backend)
})

test.skipIf(process.platform === 'win32')(
  'an installed launcher backend mints with the interpreter it reports',
  async () => {
    // The normal Windows install resolves to kind 'command' (the installation launcher).
    const python = process.execPath
    const command = await launcher(`printf '%s' '${JSON.stringify([python, '-I', '-c', 'bootstrap'])}'`)
    const root = path.join(dir, 'hermes-agent')
    await fs.mkdir(root)

    await expect(
      ticketRuntimeFor({ command, kind: 'command', shell: false, root, env: { PYTHONPATH: '/scrubbed', B: '2' } })
    ).resolves.toEqual({ command: python, env: { PYTHONPATH: root, B: '2' } })
    expect((await fs.readFile(path.join(dir, 'argv'), 'utf8')).trim()).toBe('--print-runtime-command')
  }
)

test.skipIf(process.platform === 'win32')(
  'a launcher without a usable runtime command is refused with its reason',
  async () => {
    const root = dir

    for (const [body, reason] of [
      ['exit 3', 'launcher exited 3'],
      ["printf 'not json'", 'did not report a runtime command'],
      [`printf '%s' '["relative/python"]'`, 'invalid runtime command'],
      [`printf '%s' '["${dir}/missing-python"]'`, 'invalid runtime command'],
      [`printf '%s' '{}'`, 'invalid runtime command']
    ]) {
      const command = await launcher(body)
      await expect(ticketRuntimeFor({ command, kind: 'command', shell: false, root })).rejects.toThrow(reason)
    }
  }
)

test('backends with no interpreter to name stay refused', async () => {
  for (const backend of [
    { command: 'hermes', kind: 'command', shell: false },
    { command: '"python.cmd"', kind: 'python', shell: true }
  ]) {
    await expect(ticketRuntimeFor(backend)).rejects.toThrow('requires the installed Hermes Python runtime')
  }
})
