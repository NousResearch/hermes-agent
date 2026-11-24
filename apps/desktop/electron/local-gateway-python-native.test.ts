import { spawn } from 'node:child_process'
import type * as ChildProcessAPI from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { closeGatewayTicketBridges, createGatewayTicketResolver } from './local-gateway-python'

const observed = vi.hoisted(() => ({ helpers: 0 }))
vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcessAPI>()

  return { ...actual, spawn: (...args: Parameters<typeof actual.spawn>) => {
    if (Array.isArray(args[1]) && args[1][0] === '-P') { observed.helpers++ }

    return actual.spawn(...args)
  } }
})

test.skipIf(process.platform !== 'win32')('native pipe tickets reuse one helper and revalidate every owner', async () => {
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'ticket-bridge-')))
  const root = path.resolve(import.meta.dirname, '../../..')
  const python = process.env.HERMES_TEST_PYTHON || 'python'
  const env = { ...process.env, PYTHONPATH: root, HERMES_HOME: home, PYTHONUTF8: '1' }

  const code = `
import json, sys
from pathlib import Path
from gateway.runtime_bootstrap_windows import NativeControlServer
counter = 0
def handle(raw, subject):
    global counter
    request = json.loads(raw)
    if request['params']['instance_id'] == 'refused':
        return json.dumps({'protocol':1, 'id':1, 'ok':False}).encode() + b'\\n'
    counter += 1
    return json.dumps({'protocol':1, 'id':1, 'ok':True, 'result':{
        'profile_id':str(Path(sys.argv[1]).resolve()), 'instance_id':'owner',
        'runtime_protocol':1, 'ticket':f'grant-{counter}'}}).encode() + b'\\n'
server = NativeControlServer(Path(sys.argv[1]), handle)
server.start()
print('ready', flush=True)
try:
    sys.stdin.read()
finally:
    server.close()
`

  const server = spawn(python, ['-c', code, home], { env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
  server.stderr.resume()

  try {
    await once(server.stdout, 'data')
    const endpoint = { profile_id: home, instance_id: 'owner', runtime_protocol: 1, control_home: null }
    const resolveBackend = vi.fn(async () => ({ command: python, env, kind: 'python', shell: false }))
    const mint = createGatewayTicketResolver(resolveBackend, () => root)
    const before = observed.helpers

    const [first, second] = await Promise.all([
      mint(endpoint, 'interactive'),
      mint(endpoint, 'native-http')
    ])

    expect(first).not.toBe(second)
    expect(observed.helpers - before).toBe(1)
    expect(resolveBackend).toHaveBeenCalledOnce()
    await expect(mint({ ...endpoint, instance_id: 'stale' }, 'interactive')).rejects.toThrow('Gateway ticket')
    await expect(mint({ ...endpoint, instance_id: 'refused' }, 'interactive')).rejects.toMatchObject({ reason: 'invalid_control_response' })
    await expect(mint(endpoint, 'interactive')).resolves.toMatch(/^grant-/)
    expect(observed.helpers - before).toBe(1)
    closeGatewayTicketBridges()
    await expect(mint(endpoint, 'native-http')).resolves.toMatch(/^grant-/)
    expect(observed.helpers - before).toBe(2)
    expect(resolveBackend).toHaveBeenCalledTimes(2)
  } finally {
    closeGatewayTicketBridges()
    const stopped = once(server, 'close')
    server.stdin.end()
    await stopped
    await fs.rm(home, { recursive: true, force: true })
  }
}, 60_000)
