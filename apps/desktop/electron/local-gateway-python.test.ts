import fs from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { mintGatewayTicketWithPython } from './local-gateway-python'

// This exercises the real helper process and protocol on this host. The native
// probe separately exercises Windows SID checks and HTTP/WS admission.
test.skipIf(process.platform === 'win32')('Python ticket bridge pins profile, owner, protocol and purpose', async () => {
  const home = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'gw-bridge-')))
  const root = path.resolve('../..')
  // The JS-only CI runner has no repository venv; this helper uses stdlib only.
  const python = process.env.HERMES_TEST_PYTHON || 'python3'
  const endpoint = { profile_id: home, instance_id: 'owner', runtime_protocol: 1 }
  const requests: any[] = []
  let override = {}

  const server = net.createServer(socket => socket.once('data', chunk => {
    const request = JSON.parse(chunk.toString())
    requests.push(request)
    socket.end(JSON.stringify({ protocol: 1, id: 1, ok: true, result: { ...endpoint, ticket: 'private-grant', ...override } }) + '\n')
  }))

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)
  const backend = { command: python, env: { PYTHONPATH: root, HERMES_HOME: '/must-not-win' } }
  const cwd = path.join(home, 'project')
  await fs.mkdir(path.join(cwd, 'hermes_cli'), { recursive: true })
  await fs.writeFile(path.join(cwd, 'hermes_cli', '__init__.py'), '')
  await fs.writeFile(path.join(cwd, 'hermes_cli', 'gateway_client.py'), 'def _session_ticket(*args, **kwargs):\n    return "untrusted-project-ticket"\n')

  try {
    for (const purpose of ['interactive', 'native-http'] as const) {
      await expect(mintGatewayTicketWithPython(backend, cwd, endpoint, purpose)).resolves.toBe('private-grant')
      expect(requests.at(-1).params).toEqual({ profile_id: home, instance_id: 'owner', purpose })
    }

    for (const invalid of [{ instance_id: 'other' }, { profile_id: '/other' }, { runtime_protocol: 2 }, { ticket: '' }]) {
      override = invalid
      await expect(mintGatewayTicketWithPython(backend, cwd, endpoint, 'interactive')).rejects.toThrow('Gateway ticket')
    }
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
  }
})
