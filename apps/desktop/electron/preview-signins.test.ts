import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { Cookie, CookiesGetFilter, CookiesSetDetails, Session } from 'electron'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ ipcMain: { handle: vi.fn() }, session: { fromPartition: vi.fn() } }))

import {
  cookieSetDetails,
  dropPreviewSigninsForProfile,
  type ExportVerdict,
  migratePreviewSigninsForProfile,
  parseExportVerdict,
  PreviewSigninsRecord,
  runSigninExport,
  syncPreviewSignins
} from './preview-signins'

// A cookie jar keyed like Chromium's: (url host+path, name). Tracks host-only vs domain and value so
// the removal/compare logic (P2-6) can be exercised, and supports `cookies.get` (value compare, migrate).
function fakeJar() {
  const jar = new Map<string, CookiesSetDetails>()
  const key = (host: string, path: string, name: string) => `${host}${path}|${name}`
  const keyFor = (url: string, name: string) => key(new URL(url).host, new URL(url).pathname, name)

  const asCookie = (d: CookiesSetDetails): Cookie =>
    ({
      name: d.name || '',
      value: d.value ?? '',
      domain: d.domain ?? new URL(d.url).host,
      path: d.path ?? '/',
      hostOnly: !d.domain,
      secure: Boolean(d.secure),
      httpOnly: Boolean(d.httpOnly),
      sameSite: d.sameSite,
      expirationDate: d.expirationDate
    }) as unknown as Cookie

  const session = {
    cookies: {
      set: vi.fn(async (d: CookiesSetDetails) => void jar.set(keyFor(d.url, d.name || ''), d)),
      remove: vi.fn(async (url: string, name: string) => void jar.delete(keyFor(url, name))),
      get: vi.fn(async (filter: CookiesGetFilter) => {
        const out: Cookie[] = []

        for (const d of jar.values()) {
          if (filter.name && (d.name || '') !== filter.name) {
            continue
          }

          if (filter.url && new URL(d.url).host !== new URL(filter.url).host) {
            continue
          }

          out.push(asCookie(d))
        }

        return out
      })
    }
  } as unknown as Session

  return { jar, session }
}

const dirs: string[] = []

// One store, per-profile jars keyed by the normalized profile key (`previewPartitionForProfile` →
// 'default' for null/empty). Mirrors main's `session: profile => session.fromPartition(...)`.
function rig(verdict: ExportVerdict, local = true) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'preview-signins-'))
  dirs.push(dir)
  const jars = new Map<string, ReturnType<typeof fakeJar>>()

  const jarFor = (profile: string) => {
    const k = profile.trim() || 'default'
    let entry = jars.get(k)

    if (!entry) {
      entry = fakeJar()
      jars.set(k, entry)
    }

    return entry
  }

  const exportSignins = vi.fn(async () => verdict)

  const deps = {
    session: (profile: string) => jarFor(profile).session,
    store: new PreviewSigninsRecord(path.join(dir, 'preview-signins.json')),
    resolveRoute: vi.fn(async (_s: unknown, profile: null | string) => ({ profile: profile || 'default', local })),
    exportSignins,
    log: vi.fn()
  }

  return { deps, dir, exportSignins, jarFor, jar: jarFor('default').jar }
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

  // P1-3 (SECURITY): per-profile jars. Rewritten from the pre-fix assertion that a shared jar must
  // keep another profile's same-name cookie on consent-off (the `claimedByOthers` guard): each profile
  // now has its OWN jar, so isolation — not a shared-jar carve-out — is the invariant.
  it('imports land only in the owning profile jar; another profile jar is untouched (P1-3)', async () => {
    const { deps, jarFor } = rig({ status: 'ok', cookies: [SESSION] })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })

    // The import went to work's jar, NOT default's and NOT another profile's.
    expect(jarFor('work').jar.get('app.test/|sid')?.value).toBe('secret-value')
    expect(jarFor('default').jar.has('app.test/|sid')).toBe(false)
    expect(jarFor('personal').jar.has('app.test/|sid')).toBe(false)
  })

  it("a profile's consent-off clears only its own jar, never another profile's (P1-3)", async () => {
    const { deps, jarFor } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'personal' })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: 'work' })

    // Work's jar cleared; personal's identical cookie in its SEPARATE jar is untouched. (Mutant: revert
    // `session`/partition to the shared constant → personal's jar would see work's clear.)
    expect(jarFor('work').jar.has('app.test/|sid')).toBe(false)
    expect(jarFor('personal').jar.get('app.test/|sid')?.value).toBe('secret-value')
  })

  // P2-6: a same-name host-only cookie must survive consent-off of the imported DOMAIN cookie, and a
  // cookie the user has replaced by hand (different value) must be left alone.
  it('consent-off removes the imported domain cookie but not a same-name host-only cookie (P2-6)', async () => {
    const domainCookie = { ...SESSION, domain: '.app.test' }
    const { deps, jar } = rig({ status: 'ok', cookies: [domainCookie] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    // A separate host-only cookie of the SAME name the user signed into by hand.
    jar.set('app.test/|sid#host', { url: 'https://app.test/', name: 'sid', value: 'hand-typed', path: '/' })

    const result = await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: null })

    expect(result).toMatchObject({ status: 'cleared' })
    expect(jar.get('app.test/|sid#host')?.value).toBe('hand-typed')
  })

  it('consent-off leaves a cookie the user replaced by hand after import (P2-6)', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    // User replaces the imported cookie's value by hand: it is no longer ours.
    jar.set('app.test/|sid', { url: 'https://app.test/', name: 'sid', value: 'rotated-by-user', path: '/', domain: '.app.test' })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: null })

    expect(jar.get('app.test/|sid')?.value).toBe('rotated-by-user')
  })

  // P2-8 (main): a record-write failure must not leave cookies installed-but-unowned. With write-ahead
  // the record is persisted BEFORE the jar is mutated, so a failure there installs nothing.
  it('a record-write failure installs no cookies (P2-8 main)', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    vi.spyOn(deps.store, 'write').mockImplementationOnce(() => {
      throw new Error('disk full')
    })

    // The write-ahead failure propagates (the IPC handler turns it into an error result). The
    // guarantee that matters: NO cookie was installed, so none is ever installed-but-unowned.
    await expect(syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })).rejects.toThrow('disk full')
    expect(jar.size).toBe(0)
  })

  it('a removal failure retains the claim so a retry still finds the cookie (P2-8 main)', async () => {
    const { deps, jarFor, dir } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    const jar = jarFor('default').jar
    const remove = jarFor('default').session.cookies.remove as unknown as ReturnType<typeof vi.fn>
    // The remove throws: the clear cannot complete.
    remove.mockRejectedValueOnce(new Error('locked'))

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: null })

    // Cookie still installed, so the record must still name it (no unowned cookie).
    expect(jar.has('app.test/|sid')).toBe(true)
    const record = JSON.parse(fs.readFileSync(path.join(dir, 'preview-signins.json'), 'utf8'))
    expect(record.default?.some((id: { name: string }) => id.name === 'sid')).toBe(true)
  })

  it('a re-import drops imported cookies the profile no longer has', async () => {
    const other = { ...SESSION, name: 'old' }
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION, other] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    deps.exportSignins.mockResolvedValueOnce({ status: 'ok', cookies: [SESSION] })

    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })

    expect([...jar.keys()]).toEqual(['app.test/|sid'])
  })

  // F3 (SECURITY): a re-import whose cookie value CHANGED must record the FRESH value's hash, or a
  // later consent-off's `stillOurs` compares the live (new) value against the stale hash, rejects its
  // own import and drops the claim WITHOUT removing the cookie — leaving it installed but unowned.
  it('consent-off removes a cookie whose value changed and was re-imported', async () => {
    const { deps, jar } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })

    // The real-profile cookie value rotated; a re-import installs the new value under the same identity.
    deps.exportSignins.mockResolvedValueOnce({ status: 'ok', cookies: [{ ...SESSION, value: 'rotated-value' }] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: null })
    expect(jar.get('app.test/|sid')?.value).toBe('rotated-value')

    const off = await syncPreviewSignins(deps, sender, { connectionId: null, enabled: false, profile: null })

    expect(off).toMatchObject({ status: 'cleared', cleared: 1 })
    expect(jar.has('app.test/|sid')).toBe(false)
  })
})

// P2-7: the per-profile record + jar follow a profile rename/delete.
describe('profile rename/delete hooks (P2-7)', () => {
  it('rename moves the record key and the jar cookies old→new', async () => {
    const { deps, jarFor, dir } = rig({ status: 'ok', cookies: [{ ...SESSION, domain: '.app.test' }] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })

    await migratePreviewSigninsForProfile(deps, 'work', 'job')

    const record = JSON.parse(fs.readFileSync(path.join(dir, 'preview-signins.json'), 'utf8'))
    expect(record.work).toBeUndefined()
    expect(record.job?.length).toBe(1)
    expect(jarFor('work').jar.has('app.test/|sid')).toBe(false)
    expect(jarFor('job').jar.get('app.test/|sid')?.value).toBe('secret-value')
  })

  // F6 (SECURITY): rename copies the cookie's cross-site restriction. Dropping `sameSite` reverts a
  // Strict cookie to Electron's default ('unspecified'), changing whether embedded sign-in requests
  // carry it. Initial import preserves it (cookieSetDetails maps Strict→'strict'), so rename must too.
  it('rename preserves a cookie sameSite restriction', async () => {
    const strict = { ...SESSION, domain: '.app.test', sameSite: 'Strict' as const }
    const { deps, jarFor } = rig({ status: 'ok', cookies: [strict] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })
    expect(jarFor('work').jar.get('app.test/|sid')?.sameSite).toBe('strict')

    await migratePreviewSigninsForProfile(deps, 'work', 'job')

    expect(jarFor('job').jar.get('app.test/|sid')?.sameSite).toBe('strict')
  })

  it('delete drops the record and clears the jar', async () => {
    const { deps, jarFor, dir } = rig({ status: 'ok', cookies: [SESSION] })
    await syncPreviewSignins(deps, sender, { connectionId: null, enabled: true, profile: 'work' })

    await dropPreviewSigninsForProfile(deps, 'work')

    const record = JSON.parse(fs.readFileSync(path.join(dir, 'preview-signins.json'), 'utf8'))
    expect(record.work).toBeUndefined()
    expect(jarFor('work').jar.has('app.test/|sid')).toBe(false)
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
