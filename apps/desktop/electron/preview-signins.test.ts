import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { CookiesSetDetails, Session } from 'electron'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ ipcMain: { handle: vi.fn() }, session: { fromPartition: vi.fn() } }))

import {
  cookieSetDetails,
  type ExportVerdict,
  parseExportVerdict,
  PreviewSigninsRecord,
  runSigninExport,
  syncPreviewSignins
} from './preview-signins'

// A cookie jar keyed like Chromium's: (url host+path, name).
function fakeJar() {
  const jar = new Map<string, CookiesSetDetails>()
  const key = (url: string, name: string) => `${new URL(url).host}${new URL(url).pathname}|${name}`

  const session = {
    cookies: {
      set: vi.fn(async (d: CookiesSetDetails) => void jar.set(key(d.url, d.name || ''), d)),
      remove: vi.fn(async (url: string, name: string) => void jar.delete(key(url, name)))
    }
  } as unknown as Session

  return { jar, session }
}

const dirs: string[] = []

function rig(verdict: ExportVerdict, local = true) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'preview-signins-'))
  dirs.push(dir)
  const { jar, session } = fakeJar()
  const exportSignins = vi.fn(async () => verdict)

  const deps = {
    session: () => session,
    store: new PreviewSigninsRecord(path.join(dir, 'preview-signins.json')),
    resolveRoute: vi.fn(async (_s: unknown, profile: null | string) => ({ profile: profile || 'default', local })),
    exportSignins,
    log: vi.fn()
  }

  return { deps, dir, exportSignins, jar }
}

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

const sender = {} as Electron.WebContents
const SESSION = { name: 'sid', value: 'secret-value', domain: 'app.test', path: '/', httpOnly: true, secure: true, sameSite: 'Lax' as const }

describe('syncPreviewSignins', () => {
  it('imports exported cookies into the partition and records identities without values', async () => {
    const { deps, dir, jar } = rig({ status: 'ok', cookies: [SESSION] })

    const result = await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })

    expect(result).toMatchObject({ status: 'imported', imported: 1, failed: 0 })
    expect(jar.get('app.test/|sid')?.value).toBe('secret-value')
    expect(fs.readFileSync(path.join(dir, 'preview-signins.json'), 'utf8')).not.toContain('secret-value')
    expect(JSON.stringify(result)).not.toContain('secret-value')
  })

  it('consent off removes exactly what was imported, not cookies the user signed into by hand', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    jar.set('own.test/|mine', { url: 'https://own.test/', name: 'mine', value: 'kept' })
    deps.exportSignins.mockClear()

    const result = await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: null })

    expect(result).toMatchObject({ status: 'cleared', cleared: 1 })
    expect(jar.has('app.test/|sid')).toBe(false)
    expect(jar.get('own.test/|mine')?.value).toBe('kept')
    expect(deps.exportSignins).not.toHaveBeenCalled()
  })

  it('a backend reporting consent off clears the earlier import', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    deps.exportSignins.mockResolvedValueOnce({ status: 'disabled' })

    const result = await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })

    expect(result).toMatchObject({ status: 'disabled', cleared: 1 })
    expect(jar.size).toBe(0)
  })

  it('a remote backend never runs the export and never touches the partition', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] }, false)

    const on = await syncPreviewSignins(deps, sender, { connectionId: 'ssh-box', enabled: true, profile: null })
    const off = await syncPreviewSignins(deps, sender, { connectionId: 'ssh-box', enabled: false, profile: null })

    expect(on.status).toBe('remote')
    expect(off.status).toBe('remote')
    expect(deps.exportSignins).not.toHaveBeenCalled()
    expect(jar.size).toBe(0)
  })

  it("one profile's consent-off keeps a cookie another consenting profile also imported", async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'default' })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: 'work' })

    expect(jar.get('app.test/|sid')?.value).toBe('secret-value')
  })

  it('a re-import drops imported cookies the profile no longer has', async () => {
    const other = { ...SESSION, name: 'old' }
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION, other] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    deps.exportSignins.mockResolvedValueOnce({ status: 'ok', cookies: [SESSION] })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })

    expect([...jar.keys()]).toEqual(['app.test/|sid'])
  })
})

describe('cookieSetDetails', () => {
  it('keeps a host-only cookie host-only and a domain cookie domain-wide', () => {
    const hostOnly = cookieSetDetails({ name: 'a', value: 'v', domain: 'app.test', path: '/x' })

    expect(hostOnly).toMatchObject({ url: 'http://app.test/x' })
    expect(hostOnly).not.toHaveProperty('domain')
    expect(cookieSetDetails({ name: 'a', value: 'v', domain: '.app.test', secure: true, expires: 2e9 })).toMatchObject({
      url: 'https://app.test/',
      domain: '.app.test',
      expirationDate: 2e9
    })
  })
})

describe('runSigninExport', () => {
  it('reads the verdict from the child stdout pipe', async () => {
    const line = JSON.stringify({ status: 'ok', cookies: [SESSION], skipped: 2 })

    const verdict = await runSigninExport(
      { command: process.execPath, args: ['-e', `process.stdout.write('notice\\n' + ${JSON.stringify(line)} + '\\n')`] },
      process.cwd(),
      '/h',
      { PATH: process.env.PATH ?? '' }
    )

    expect(verdict).toEqual({ status: 'ok', cookies: [SESSION], skipped: 2 })
  })

  it('a child with no verdict is an error naming its last stderr line, never an empty import', async () => {
    const verdict = await runSigninExport(
      { command: process.execPath, args: ['-e', "process.stderr.write('boom\\n'); process.exit(3)"] },
      process.cwd(),
      '/h',
      { PATH: process.env.PATH ?? '' }
    )

    expect(verdict).toEqual({ status: 'error', error: 'boom' })
    expect(parseExportVerdict('garbage')).toBeNull()
  })
})
