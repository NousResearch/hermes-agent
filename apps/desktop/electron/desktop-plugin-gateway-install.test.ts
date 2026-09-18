import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import {
  GATEWAY_HALF_MARKER,
  gatewayHalfFolderName,
  installGatewayDesktopHalf,
  readGatewayHalfMarker,
  readInstalledGatewayHalves,
  sha256Hex
} from './desktop-plugin-gateway-install'

const roots: string[] = []

function makeRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-gw-half-'))
  roots.push(root)

  return root
}

const HALF = "export default { id: 'mission-control', register() {} }\n"
const HALF_V2 = "export default { id: 'mission-control', register() { /* v2 */ } }\n"

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

describe('gatewayHalfFolderName', () => {
  it('accepts a plain plugin name', () => {
    expect(gatewayHalfFolderName('mission-control')).toBe('mission-control')
  })

  it.each(['../escape', 'a/b', 'a\\b', '..', '.', '.hidden', '', '  ', 'x'.repeat(129)])(
    'refuses %j rather than sanitizing it into a folder the plugin id no longer matches',
    name => {
      expect(gatewayHalfFolderName(name)).toBeNull()
    }
  )
})

describe('installGatewayDesktopHalf', () => {
  it('writes plugin.js + a gateway marker, and reports the digest', async () => {
    const root = makeRoot()
    const digest = sha256Hex(HALF)

    const result = await installGatewayDesktopHalf({
      key: 'mission-control',
      name: 'mission-control',
      root,
      sha256: digest,
      source: 'user',
      text: HALF
    })

    expect(result).toMatchObject({ ok: true, sha256: digest })
    expect(fs.readFileSync(path.join(root, 'mission-control', 'plugin.js'), 'utf8')).toBe(HALF)

    const marker = JSON.parse(fs.readFileSync(path.join(root, 'mission-control', GATEWAY_HALF_MARKER), 'utf8'))
    expect(marker).toMatchObject({
      bytes: Buffer.byteLength(HALF, 'utf8'),
      key: 'mission-control',
      name: 'mission-control',
      sha256: digest,
      source: 'user'
    })
  })

  it('refuses bytes that do not match the digest the gateway reported, and writes nothing', async () => {
    const root = makeRoot()

    const result = await installGatewayDesktopHalf({
      name: 'mission-control',
      root,
      sha256: sha256Hex('something else'),
      text: HALF
    })

    expect(result).toMatchObject({ ok: false, reason: 'integrity' })
    expect(fs.existsSync(path.join(root, 'mission-control'))).toBe(false)
  })

  it('refuses an HTML error page served in place of the half', async () => {
    const root = makeRoot()

    const result = await installGatewayDesktopHalf({
      name: 'mission-control',
      root,
      text: '<!doctype html><html><body>502 Bad Gateway</body></html>'
    })

    expect(result).toMatchObject({ ok: false, reason: 'unavailable' })
    expect(fs.existsSync(path.join(root, 'mission-control'))).toBe(false)
  })

  it('refuses an empty payload', async () => {
    const root = makeRoot()

    expect(await installGatewayDesktopHalf({ name: 'mission-control', root, text: '' })).toMatchObject({
      ok: false,
      reason: 'unavailable'
    })
  })

  it('never replaces a folder this path did not install unless forced', async () => {
    const root = makeRoot()
    fs.mkdirSync(path.join(root, 'mission-control'), { recursive: true })
    fs.writeFileSync(path.join(root, 'mission-control', 'plugin.js'), 'hand copied')

    const refused = await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF })

    expect(refused).toMatchObject({ ok: false, reason: 'exists' })
    expect(fs.readFileSync(path.join(root, 'mission-control', 'plugin.js'), 'utf8')).toBe('hand copied')

    const forced = await installGatewayDesktopHalf({ force: true, name: 'mission-control', root, text: HALF })

    expect(forced.ok).toBe(true)
    expect(fs.readFileSync(path.join(root, 'mission-control', 'plugin.js'), 'utf8')).toBe(HALF)
  })

  it('refreshes its own folder when the gateway serves a newer half, and is a no-op otherwise', async () => {
    const root = makeRoot()

    await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF })

    const same = await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF })

    expect(same).toMatchObject({ ok: true, unchanged: true, sha256: sha256Hex(HALF) })

    const newer = await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF_V2 })

    expect(newer).toMatchObject({ ok: true, sha256: sha256Hex(HALF_V2) })
    expect(fs.readFileSync(path.join(root, 'mission-control', 'plugin.js'), 'utf8')).toBe(HALF_V2)
    expect((await readGatewayHalfMarker(root, 'mission-control'))?.sha256).toBe(sha256Hex(HALF_V2))
  })

  it('never replaces a FILE that already sits at the plugin folder name', async () => {
    const root = makeRoot()
    fs.writeFileSync(path.join(root, 'mission-control'), 'not a folder')

    const result = await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF })

    expect(result).toMatchObject({ ok: false, reason: 'exists' })
    expect(fs.readFileSync(path.join(root, 'mission-control'), 'utf8')).toBe('not a folder')
  })

  it('reports an io failure instead of leaving a half-installed folder', async () => {
    const root = makeRoot()
    // A FILE where the ROOT must be: the folder cannot be created at all.
    fs.writeFileSync(path.join(root, 'not-a-dir'), 'x')
    const brokenRoot = path.join(root, 'not-a-dir', 'desktop-plugins')

    const result = await installGatewayDesktopHalf({ name: 'mission-control', root: brokenRoot, text: HALF })

    expect(result).toMatchObject({ ok: false, reason: 'io' })
    expect(fs.existsSync(brokenRoot)).toBe(false)
  })
})

describe('readInstalledGatewayHalves', () => {
  it('maps every folder this path installed, by folder name', async () => {
    const root = makeRoot()
    await installGatewayDesktopHalf({ name: 'mission-control', root, text: HALF })
    fs.mkdirSync(path.join(root, 'by-hand'), { recursive: true })
    fs.writeFileSync(path.join(root, 'by-hand', 'plugin.js'), HALF)

    const halves = await readInstalledGatewayHalves(root)

    expect(Object.keys(halves)).toEqual(['mission-control'])
    expect(halves['mission-control']?.sha256).toBe(sha256Hex(HALF))
  })

  it('answers {} for a root that does not exist yet', async () => {
    expect(await readInstalledGatewayHalves(path.join(os.tmpdir(), 'hermes-gw-half-missing'))).toEqual({})
  })
})
