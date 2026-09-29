import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { describe, expect, it } from 'vitest'

import {
  createTailcatForwarder,
  pairTailcatCode,
  parseTailcatCode,
  redactTailcat,
  resolveTailcatBinary,
  tailcatAddressFingerprint
} from './tailcat-connection'

const ADDRESS = `tc${'A'.repeat(150)}`

function fakeChild() {
  const child = new EventEmitter() as any
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.exitCode = null
  child.killed = false

  child.kill = () => {
    child.killed = true
    child.exitCode = 0
    child.emit('exit', null)
  }

  return child
}

describe('connection codes', () => {
  it('round-trip the fields the backend renders, and reject anything else', () => {
    expect(parseTailcatCode(` hermes-tailcat:${ADDRESS}:41234:s3cret \n`)).toEqual({
      address: ADDRESS,
      port: 41234,
      secret: 's3cret'
    })

    for (const bad of [
      '',
      `${ADDRESS}:41234:s3cret`,
      `hermes-tailcat:${ADDRESS}:41234`,
      `hermes-tailcat:${ADDRESS}:0:s`,
      `hermes-tailcat:${ADDRESS}:70000:s`,
      'hermes-tailcat:notanaddress:41234:s'
    ]) {
      expect(parseTailcatCode(bad)).toBeNull()
    }
  })

  it('never lets an address into text meant for logs or the UI', () => {
    expect(redactTailcat(`dial ${ADDRESS} failed`)).not.toContain(ADDRESS)
    // Addresses share a long prefix and suffix; the fingerprint must still tell them apart.
    expect(tailcatAddressFingerprint(`${ADDRESS}x`)).not.toBe(tailcatAddressFingerprint(`${ADDRESS}y`))
  })
})

describe('binary resolution', () => {
  it('prefers PATH, then the pinned PM copy, and installs only when asked', async () => {
    const calls: string[][] = []
    const pmBin = '/pm/tools/tailcat/bin'

    const deps = (onPath: null | string, installed: boolean) => ({
      exists: (p: string) => installed && p === `${pmBin}/tailcat`,
      findOnPath: () => onPath,
      platform: 'linux' as const,
      runPm: async (args: string[]) => {
        calls.push(args)

        if (args[0] === 'install') {
          installed = true
        }

        // Real `hermes pm env` output: progress lines, then indented JSON ({} until installed).
        return `Preparing the isolated Hermes runtime…\n  ✓ Installing Python dependencies\n${JSON.stringify(installed ? { PATH: pmBin } : {}, null, 2)}\n`
      }
    })

    expect(await resolveTailcatBinary(deps('/usr/local/bin/tailcat', false), { install: true })).toBe('/usr/local/bin/tailcat')
    expect(calls).toEqual([])

    expect(await resolveTailcatBinary(deps(null, false))).toBeNull()
    expect(calls.some(args => args[0] === 'install')).toBe(false)

    expect(await resolveTailcatBinary(deps(null, false), { install: true })).toBe(`${pmBin}/tailcat`)
    expect(calls.some(args => args[0] === 'install')).toBe(true)
  })
})

describe('forwarder', () => {
  it('shares one tunnel per connection and replaces it after it dies', async () => {
    const spawned: any[] = []

    const forwarder = createTailcatForwarder({
      resolveBinary: async () => '/bin/tailcat',
      spawn: (_command, args) => {
        const child = fakeChild()
        spawned.push({ args, child })
        setImmediate(() => child.stderr.write(`forwarding 127.0.0.1:${40000 + spawned.length} -> ${ADDRESS}:41234\n`))

        return child
      }
    })

    expect(await forwarder.ensure('mini', ADDRESS, 41234)).toBe(40001)
    expect(await forwarder.ensure('mini', ADDRESS, 41234)).toBe(40001)
    expect(spawned).toHaveLength(1)
    expect(spawned[0].args).toEqual(['forward', '--key=new', ADDRESS, '0:41234'])

    spawned[0].child.kill()
    expect(await forwarder.ensure('mini', ADDRESS, 41234)).toBe(40002)

    forwarder.stopAll()
    expect(spawned[1].child.killed).toBe(true)
  })

  it('reports a tunnel that exits before opening without leaking the address', async () => {
    const forwarder = createTailcatForwarder({
      resolveBinary: async () => '/bin/tailcat',
      spawn: () => {
        const child = fakeChild()

        setImmediate(() => {
          child.stderr.write(`dial ${ADDRESS}: relay refused\n`)
          child.exitCode = 1
          child.emit('exit', 1)
        })

        return child
      }
    })

    const error = await forwarder.ensure('mini', ADDRESS, 41234).catch(e => e)
    expect(error.code).toBe('forward-failed')
    expect(error.message).toContain('relay refused')
    expect(error.message).not.toContain(ADDRESS)
    expect(forwarder.has('mini')).toBe(false)
  })

  it('explains a missing binary instead of spawning nothing', async () => {
    const forwarder = createTailcatForwarder({ resolveBinary: async () => null, spawn: () => fakeChild() })

    await expect(forwarder.ensure('mini', ADDRESS, 1)).rejects.toMatchObject({ code: 'missing-binary' })
  })
})

describe('pairing', () => {
  const forwarder = {
    ensure: async () => 40001,
    has: () => true,
    stop: () => {},
    stopAll: () => {}
  } as any

  it('redeems the code through the tunnel and returns what the registry stores', async () => {
    const requests: any[] = []

    const fetchImpl = (async (url: string, init: any) => {
      requests.push({ body: JSON.parse(init.body), url })

      return new Response(JSON.stringify({ device: { id: 'dev1' }, token: 'device-token' }), { status: 200 })
    }) as any

    const code = `hermes-tailcat:${ADDRESS}:41234:s3cret`

    expect(await pairTailcatCode(forwarder, 'k', code, 'laptop', fetchImpl)).toEqual({
      address: ADDRESS,
      deviceId: 'dev1',
      port: 41234,
      token: 'device-token'
    })
    expect(requests).toEqual([{ body: { code, name: 'laptop' }, url: 'http://127.0.0.1:40001/api/share/pair' }])
  })

  it('surfaces a used or expired code as a rejected code', async () => {
    const fetchImpl = (async () =>
      new Response(JSON.stringify({ detail: 'Connection code expired or already used.' }), { status: 403 })) as any

    await expect(
      pairTailcatCode(forwarder, 'k', `hermes-tailcat:${ADDRESS}:41234:s`, 'laptop', fetchImpl)
    ).rejects.toMatchObject({ code: 'code-rejected', message: 'Connection code expired or already used.' })
  })
})
