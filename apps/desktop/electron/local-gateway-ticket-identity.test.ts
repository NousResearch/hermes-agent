import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from 'vitest'

import {
  type GatewayEndpoint,
  isStaleLocalGatewayError,
  mintLocalGatewayTicket,
  nativeGatewayHttpHeaders,
  reservedHome
} from './local-gateway'
import { shortSocketTmpDir } from './local-gateway.test-helpers'

// A gateway ticket names a profile home by the identity the gateway reserved for it: its
// `Path.resolve()`, which Desktop must reproduce exactly or the gateway refuses the ticket.

const ORIGIN = 'http://127.0.0.1:1234'
const LINK = process.platform === 'win32' ? 'junction' : 'dir'
const PYTHON = process.env.HERMES_TEST_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

const endpointFor = (profile_id: string): GatewayEndpoint => ({
  profile_id,
  instance_id: 'owner',
  authority_epoch: 1,
  runtime_protocol: 1,
  api_origin: ORIGIN,
  capabilities: ['session-authority-v1'],
  supervisor: 'none',
  control_home: null
})

async function sameDirectory(a: string, b: string): Promise<boolean> {
  const [left, right] = await Promise.all([a, b].map(dir => fs.stat(dir, { bigint: true }).catch(() => null)))

  return Boolean(left && right && left.dev === right.dev && left.ino === right.ino)
}

async function pythonResolve(homes: string[]): Promise<string[]> {
  const script =
    'import json, sys; from pathlib import Path; print(json.dumps([str(Path(p).resolve()) for p in sys.argv[1:]]))'
  const { stdout } = await promisify(execFile)(PYTHON, ['-c', script, ...homes], { windowsHide: true })

  return JSON.parse(stdout)
}

// #138754: macOS realpath(3) answers a case-insensitive volume in its stored letter case, while
// Python keeps the given spelling of ordinary components (POSIX), answers the final on-disk path
// (Windows), and joins a relative link target onto the prefix resolved so far.
test('a profile home resolves to exactly the home the gateway reserves for it', async () => {
  const base = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'gw-identity-')))
  const home = path.join(base, 'home')
  await Promise.all([fs.mkdir(path.join(home, 'profiles'), { recursive: true }), fs.mkdir(path.join(base, 'other'))])
  await fs.mkdir(path.join(home, 'work-data'))
  // Windows junctions take an absolute target; a POSIX profile link is often relative.
  await fs.symlink(
    process.platform === 'win32' ? path.join(home, 'work-data') : '../work-data',
    path.join(home, 'profiles', 'work'),
    LINK
  )
  await fs.symlink(path.join(base, 'other'), path.join(base, 'link'), LINK)
  // A letter-case respelling exists only on a case-insensitive volume (Windows, macOS by default).
  const spelled = path.join(base, 'HOME')
  const respelled = (await sameDirectory(home, spelled)) ? [spelled, path.join(spelled, 'profiles', 'work')] : []
  const homes = [home, path.join(home, 'profiles', 'work'), path.join(base, 'link'), ...respelled]

  try {
    expect(await Promise.all(homes.map(reservedHome))).toEqual(await pythonResolve(homes))
  } finally {
    await fs.rm(base, { recursive: true, force: true })
  }
})

/** A POSIX control socket that mints like the gateway's: only for a profile id it serves, exactly. */
async function controlSocket(home: string, served: string[]) {
  const requests: Array<Record<string, unknown>> = []

  const server = net.createServer(socket =>
    socket.once('data', chunk => {
      const params = JSON.parse(chunk.toString()).params
      requests.push(params)
      socket.end(
        JSON.stringify(
          served.includes(params.profile_id)
            ? {
                protocol: 1,
                id: 1,
                ok: true,
                result: {
                  profile_id: params.profile_id,
                  instance_id: 'owner',
                  runtime_protocol: 1,
                  ticket: `grant-${requests.length}`
                }
              }
            : { protocol: 1, id: 1, ok: false, error: 'PermissionError: bootstrap binding rejected' }
        ) + '\n'
      )
    })
  )

  await new Promise<void>(resolve => server.listen(path.join(home, 'gateway.sock'), resolve))
  await fs.chmod(path.join(home, 'gateway.sock'), 0o600)

  return { requests, close: () => new Promise<void>(resolve => server.close(() => resolve())) }
}

// The gateway serves a linked `profiles/<name>` at its resolved home, so it never mints for the
// joined path. Identity is not a licence to follow links in an ADVERTISED home: one that leads to
// another directory is still refused, without asking the gateway. A profile with no directory is
// a definitive answer, not a stale gateway: re-ensuring the shared connection cannot create it.
test.skipIf(process.platform === 'win32')(
  'a linked profile mints for the home the gateway reserved; an advertised home that leads elsewhere is refused',
  async () => {
    const [root, target, links] = await Promise.all(['gw-root-', 'gw-target-', 'gw-links-'].map(shortSocketTmpDir))
    await fs.mkdir(path.join(root, 'profiles'), { mode: 0o700 })
    await fs.symlink(target, path.join(root, 'profiles', 'work'), 'dir')
    await fs.symlink(root, path.join(links, 'home'), 'dir')
    const gateway = await controlSocket(root, [root, target])
    const descriptor = { gatewayEndpoint: endpointFor(root), baseUrl: ORIGIN }

    try {
      expect(await nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=work`)).toEqual({
        'X-Hermes-Gateway-Ticket': 'grant-1'
      })
      await expect(mintLocalGatewayTicket(endpointFor(path.join(links, 'home')), 'native-http')).rejects.toThrow(
        'Noncanonical gateway profile'
      )
      await expect(
        mintLocalGatewayTicket({ ...endpointFor(root), control_home: path.join(links, 'home') }, 'native-http')
      ).rejects.toThrow('Noncanonical gateway profile')
      const gone = await nativeGatewayHttpHeaders(descriptor, `${ORIGIN}/api/sessions?profile=gone`).catch(
        error => error
      )
      expect([gone.message.includes(`Profile 'gone' does not exist`), isStaleLocalGatewayError(gone)]).toEqual([
        true,
        false
      ])
      expect(gateway.requests).toEqual([{ profile_id: target, instance_id: 'owner', purpose: 'native-http' }])
    } finally {
      await gateway.close()
      await Promise.all([root, target, links].map(dir => fs.rm(dir, { recursive: true, force: true })))
    }
  }
)
