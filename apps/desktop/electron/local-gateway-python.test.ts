import fs from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'

import { expect, test } from 'vitest'

import { closeGatewayTicketBridges, mintGatewayTicketWithPython } from './local-gateway-python'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

// This exercises the real helper process and protocol on this host. The native
// probe separately exercises Windows SID checks and HTTP/WS admission.
test.skipIf(process.platform === 'win32')('Python ticket bridge pins profile, owner, protocol and purpose', async () => {
  const home = await shortSocketTmpDir('gw-bridge-')
  const mux = await shortSocketTmpDir('gw-bridge-mux-')
  const root = path.resolve('../..')
  // The JS-only CI runner has no repository venv; this helper uses stdlib only.
  const python = process.env.HERMES_TEST_PYTHON || 'python3'
  // Real serialized endpoints include control_home (null for the launch profile).
  // The bridge must merge the default rather than pass a duplicate Python keyword.
  const endpoint = { profile_id: home, instance_id: 'owner', runtime_protocol: 1, control_home: null }
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
      // Only the interactive (shared-primary) socket asks for host scope; native-http stays profile-bound.
      expect(requests.at(-1).params).toEqual({ profile_id: home, instance_id: 'owner', purpose, ...(purpose === 'interactive' ? { scope: 'host' } : {}) })
    }

    for (const invalid of [{ instance_id: 'other' }, { profile_id: '/other' }, { runtime_protocol: 2 }, { ticket: '' }]) {
      override = invalid
      await expect(mintGatewayTicketWithPython(backend, cwd, endpoint, 'interactive')).rejects.toThrow('Gateway ticket')
    }

    // The profile home gets Python's full home policy: a private-group 0775 home mints, the same
    // home with a named-user ACL (group bits become the mask) is refused with its reason.
    // The expected verdict is Python's own home_mode_unsafe for this host's group (a CI runner's
    // primary group need not be private), so this is a parity check, not a host assumption.
    override = {}
    const { execFileSync } = await import('node:child_process')

    const unsafe = () => execFileSync(python, ['-c', 'import sys; from pathlib import Path; '
      + 'from hermes_cli.gateway_runtime_discovery import home_mode_unsafe as u; p = Path(sys.argv[1]); '
      + 'print(int(u(p.lstat(), p)))', home], { env: { ...process.env, PYTHONPATH: root }, encoding: 'utf8' }).trim() === '1'

    const expectVerdict = async () => {
      const mint = mintGatewayTicketWithPython(backend, cwd, endpoint, 'interactive')

      await (unsafe() ? expect(mint).rejects.toMatchObject({ reason: 'unsafe_control_permissions' })
        : expect(mint).resolves.toBe('private-grant'))
    }

    await fs.chmod(home, 0o775)
    await expectVerdict()
    let acl = false

    try { execFileSync('setfacl', ['-m', 'u:nobody:rwx', home]); acl = true } catch { acl = false }

    if (acl) {
      expect(unsafe(), 'a named-user ACL makes the group bits a mask').toBe(true)
      await expectVerdict()
      execFileSync('setfacl', ['-b', home])
    }

    await fs.chmod(home, 0o700)

    // A served secondary carries the multiplexer's control_home: the ticket is minted by THAT
    // home's socket (the profile home has none), still bound to the secondary's identity.
    override = {}
    await new Promise<void>(resolve => server.close(() => resolve()))
    await new Promise<void>(resolve => server.listen(path.join(mux, 'gateway.sock'), resolve))
    await fs.chmod(path.join(mux, 'gateway.sock'), 0o600)
    await expect(mintGatewayTicketWithPython(backend, cwd, { ...endpoint, control_home: mux }, 'interactive')).resolves.toBe('private-grant')
    expect(requests.at(-1).params).toEqual({ profile_id: home, instance_id: 'owner', purpose: 'interactive', scope: 'host' })
  } finally {
    closeGatewayTicketBridges()
    await new Promise<void>(resolve => server.close(() => resolve()))
    await fs.rm(home, { recursive: true, force: true })
    await fs.rm(mux, { recursive: true, force: true })
  }
})
