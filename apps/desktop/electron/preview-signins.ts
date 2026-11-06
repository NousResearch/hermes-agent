/**
 * Sign the in-app Browser tab (`persist:hermes-preview`) in with the user's real browser profile.
 *
 * Gated by `browser.use_real_profile` and by a LOCAL backend: the real profile lives on the backend host,
 * so a remote/SSH/Cloud connection (another machine) never imports. Main decides "local" from the route's
 * resolved connection, never from a renderer flag.
 *
 * No cookie decryption here or in the backend: the user's real browser, running on the Hermes-owned profile
 * snapshot, decrypts its own store, and `hermes browser export-signins --json` (a local child of the owning
 * profile, which re-checks consent itself) reads it over CDP and writes the list to its stdout pipe. Values
 * go from that pipe straight into the partition's cookie jar; the renderer gets counts only. The identities
 * imported (url, name, domain, path; never values) are recorded under userData so consent-off removes
 * exactly those, and nothing the user signed into by hand inside the in-app browser.
 */
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { type CookiesSetDetails, ipcMain, type Session, session } from 'electron'

import { PREVIEW_GUEST_PARTITION } from './preview-guest-throttle'
import { hiddenWindowsChildOptions, windowsShellCommand } from './windows-child-options'

const EXPORT_TIMEOUT_MS = 90_000
const EXPORT_STDOUT_LIMIT = 32 * 1024 * 1024
export const PREVIEW_SIGNINS_RECORD_FILE = 'preview-signins.json'

export interface ExportedCookie {
  name: string
  value: string
  domain: string
  path?: string
  expires?: number
  httpOnly?: boolean
  secure?: boolean
  session?: boolean
  sameSite?: 'Lax' | 'None' | 'Strict'
}

export type ExportVerdict =
  | { status: 'disabled' }
  | { status: 'error'; error: string }
  | { status: 'ok'; cookies: ExportedCookie[]; skipped?: number }

export interface CookieIdentity {
  url: string
  name: string
  domain: string
  path: string
}

/** What the renderer sees: counts and a reason, never a cookie. */
export interface PreviewSigninsResult {
  status: 'cleared' | 'disabled' | 'error' | 'imported' | 'remote'
  imported: number
  skipped: number
  failed: number
  cleared: number
  error?: string
}

export interface BackendCommand {
  command: string | null
  args: string[]
  env?: NodeJS.ProcessEnv
  shell?: boolean
}

/** Run the export child and parse its one-line verdict. The pipe is the only place the values exist. */
export function runSigninExport(
  backend: BackendCommand,
  cwd: string,
  home: string,
  parentEnv: NodeJS.ProcessEnv
): Promise<ExportVerdict> {
  return new Promise(resolve => {
    if (!backend.command) {
      resolve({ status: 'error', error: 'Hermes runtime is not available' })

      return
    }

    const env = { ...parentEnv, HERMES_HOME: home, ...backend.env }
    const child = spawn(windowsShellCommand(backend.command, Boolean(backend.shell)), backend.args, hiddenWindowsChildOptions({ cwd, env, shell: backend.shell, stdio: ['ignore', 'pipe', 'pipe'] }))
    const chunks: Buffer[] = []
    let size = 0
    let stderr = ''
    let failure = ''

    const timer = setTimeout(() => {
      failure = 'timed out reading sign-ins from the real-profile browser'
      child.kill()
    }, EXPORT_TIMEOUT_MS)

    child.stdout?.on('data', (data: Buffer) => {
      size += data.length

      if (size > EXPORT_STDOUT_LIMIT) {
        failure = 'sign-in export exceeded its size limit'
        child.kill()

        return
      }

      chunks.push(data)
    })
    child.stderr?.on('data', data => {
      stderr = (stderr + data.toString()).slice(-2048)
    })
    child.on('error', () => {
      clearTimeout(timer)
      resolve({ status: 'error', error: 'could not run the Hermes sign-in export' })
    })
    child.on('close', code => {
      clearTimeout(timer)

      if (failure) {
        resolve({ status: 'error', error: failure })

        return
      }

      const verdict = parseExportVerdict(Buffer.concat(chunks).toString('utf8'))
      const lastStderr = stderr.trim().split(/\r?\n/).filter(Boolean).pop()

      resolve(verdict ?? { status: 'error', error: lastStderr || `sign-in export exited ${code ?? 'abnormally'} without a result` })
    })
  })
}

/** The last JSON line of the child's stdout (a CLI may print notices before it). */
export function parseExportVerdict(stdout: string): ExportVerdict | null {
  const line = stdout.trim().split(/\r?\n/).reverse().find(l => l.startsWith('{'))

  if (!line) {
    return null
  }

  try {
    const parsed = JSON.parse(line)

    if (parsed?.status === 'disabled') {
      return { status: 'disabled' }
    }

    if (parsed?.status === 'error') {
      return { status: 'error', error: String(parsed.error || 'sign-in export failed') }
    }

    if (parsed?.status === 'ok' && Array.isArray(parsed.cookies)) {
      return { status: 'ok', cookies: parsed.cookies, skipped: Number(parsed.skipped) || 0 }
    }
  } catch {
    // fall through: unparseable is "no result"
  }

  return null
}

const SAME_SITE: Record<string, CookiesSetDetails['sameSite']> = { Lax: 'lax', None: 'no_restriction', Strict: 'strict' }

/** Electron `cookies.set` details for one exported cookie, or null when it cannot be represented. */
export function cookieSetDetails(cookie: ExportedCookie): CookiesSetDetails | null {
  if (!cookie?.name || !cookie.domain || typeof cookie.value !== 'string') {
    return null
  }

  const host = cookie.domain.replace(/^\./, '')
  const cookiePath = cookie.path || '/'

  const details: CookiesSetDetails = {
    url: `${cookie.secure ? 'https' : 'http'}://${host}${cookiePath}`,
    name: cookie.name,
    value: cookie.value,
    path: cookiePath,
    secure: Boolean(cookie.secure),
    httpOnly: Boolean(cookie.httpOnly),
    sameSite: (cookie.sameSite && SAME_SITE[cookie.sameSite]) || 'unspecified'
  }

  // A leading dot is a domain cookie; without one the cookie is host-only, which Electron expresses by
  // leaving `domain` out (passing it would widen the cookie to subdomains).
  if (cookie.domain.startsWith('.')) {
    details.domain = cookie.domain
  }

  if (!cookie.session && typeof cookie.expires === 'number' && cookie.expires > 0) {
    details.expirationDate = cookie.expires
  }

  return details
}

export function identityOf(details: CookiesSetDetails): CookieIdentity {
  return { url: details.url, name: details.name || '', domain: details.domain || '', path: details.path || '/' }
}

const identityKey = (id: CookieIdentity) => `${id.url}\u0000${id.name}\u0000${id.domain}\u0000${id.path}`

type RecordFile = Record<string, CookieIdentity[]>

export class PreviewSigninsRecord {
  constructor(private readonly file: string) {}

  read(): RecordFile {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.file, 'utf8'))

      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
    } catch {
      return {}
    }
  }

  write(next: RecordFile): void {
    fs.mkdirSync(path.dirname(this.file), { recursive: true })
    const tmp = `${this.file}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(next), { encoding: 'utf8', mode: 0o600 })
    fs.renameSync(tmp, this.file)
  }
}

async function removeIdentities(jar: Session, ids: CookieIdentity[], keep: Set<string>): Promise<number> {
  let removed = 0

  for (const id of ids) {
    if (keep.has(identityKey(id))) {
      continue
    }

    try {
      await jar.cookies.remove(id.url, id.name)
      removed += 1
    } catch {
      // already gone
    }
  }

  return removed
}

/** Identities another profile still claims: shared partition, so one profile's consent-off must not sign
 *  the in-app browser out of what a still-consenting profile imported. */
function claimedByOthers(record: RecordFile, profile: string): Set<string> {
  return new Set(
    Object.entries(record)
      .filter(([name]) => name !== profile)
      .flatMap(([, ids]) => (Array.isArray(ids) ? ids : []).map(identityKey))
  )
}

export async function clearImportedSignins(jar: Session, store: PreviewSigninsRecord, profile: string): Promise<number> {
  const record = store.read()
  const mine = Array.isArray(record[profile]) ? record[profile] : []

  if (!mine.length && !(profile in record)) {
    return 0
  }

  const removed = await removeIdentities(jar, mine, claimedByOthers(record, profile))
  const { [profile]: _dropped, ...rest } = record
  store.write(rest)

  return removed
}

/** Replace `profile`'s imported set with `cookies`: set each, then drop identities it no longer has. */
export async function applyImportedSignins(
  jar: Session,
  store: PreviewSigninsRecord,
  profile: string,
  cookies: ExportedCookie[]
): Promise<{ imported: number; failed: number }> {
  const record = store.read()
  const previous = Array.isArray(record[profile]) ? record[profile] : []
  const now: CookieIdentity[] = []
  let failed = 0

  for (const cookie of cookies) {
    const details = cookieSetDetails(cookie)

    if (!details) {
      failed += 1

      continue
    }

    try {
      await jar.cookies.set(details)
      now.push(identityOf(details))
    } catch {
      failed += 1
    }
  }

  const current = new Set(now.map(identityKey))
  await removeIdentities(jar, previous, new Set([...current, ...claimedByOthers(record, profile)]))
  store.write({ ...store.read(), [profile]: now })

  return { imported: now.length, failed }
}

export interface PreviewSigninsDeps {
  session: () => Session
  store: PreviewSigninsRecord
  /** The owning profile and whether its resolved backend is on THIS machine (main computes both). */
  resolveRoute: (
    sender: Electron.WebContents,
    profile: null | string,
    connectionId: null | string
  ) => Promise<{ profile: string; local: boolean }>
  exportSignins: (profile: string) => Promise<ExportVerdict>
  log: (message: string) => void
}

const empty = (status: PreviewSigninsResult['status'], extra: Partial<PreviewSigninsResult> = {}): PreviewSigninsResult => ({
  status,
  imported: 0,
  skipped: 0,
  failed: 0,
  cleared: 0,
  ...extra
})

/** One sync for a LOCAL route (remote: no-op): `enabled=false` clears without spawning anything;
 *  `enabled=true` imports when the backend confirms consent, and clears when it reports consent off. */
export async function syncPreviewSignins(
  deps: PreviewSigninsDeps,
  sender: Electron.WebContents,
  request: { connectionId: null | string; enabled: boolean; profile: null | string }
): Promise<PreviewSigninsResult> {
  const { profile, local } = await deps.resolveRoute(sender, request.profile, request.connectionId)

  // A remote backend never imported (its profile is another machine's), so it has nothing to clear either;
  // checking first keeps a remote `default` from clearing what the local `default` imported.
  if (!local) {
    return empty('remote')
  }

  if (!request.enabled) {
    const cleared = await clearImportedSignins(deps.session(), deps.store, profile)

    if (cleared) {
      deps.log(`[preview-signins] consent off for "${profile}": removed ${cleared} imported cookie(s)`)
    }

    return empty('cleared', { cleared })
  }

  const verdict = await deps.exportSignins(profile)

  if (verdict.status === 'disabled') {
    const cleared = await clearImportedSignins(deps.session(), deps.store, profile)

    return empty('disabled', { cleared })
  }

  if (verdict.status === 'error') {
    deps.log(`[preview-signins] import for "${profile}" failed: ${verdict.error}`)

    return empty('error', { error: verdict.error })
  }

  const { imported, failed } = await applyImportedSignins(deps.session(), deps.store, profile, verdict.cookies)
  deps.log(`[preview-signins] imported ${imported} cookie(s) for "${profile}" (${failed} failed, ${verdict.skipped ?? 0} skipped)`)

  return empty('imported', { imported, failed, skipped: verdict.skipped ?? 0 })
}

/** main.ts registration: the partition and the record file are owned here; main supplies route + spawn. */
export function registerPreviewSignins(opts: {
  userData: string
  resolveRoute: PreviewSigninsDeps['resolveRoute']
  exportSignins: PreviewSigninsDeps['exportSignins']
  log: PreviewSigninsDeps['log']
}): void {
  registerPreviewSigninsIpc(ipcMain, {
    session: () => session.fromPartition(PREVIEW_GUEST_PARTITION),
    store: new PreviewSigninsRecord(path.join(opts.userData, PREVIEW_SIGNINS_RECORD_FILE)),
    resolveRoute: opts.resolveRoute,
    exportSignins: opts.exportSignins,
    log: opts.log
  })
}

export function registerPreviewSigninsIpc(ipc: Pick<Electron.IpcMain, 'handle'>, deps: PreviewSigninsDeps): void {
  // One sync at a time: a toggle flip during an import must land after it, never interleave with it.
  let chain: Promise<unknown> = Promise.resolve()

  ipc.handle('hermes:preview-signins:sync', (event, payload: unknown) => {
    const body = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
    const profile = typeof body.profile === 'string' && body.profile.trim() ? body.profile.trim() : null
    const connectionId = typeof body.connectionId === 'string' && body.connectionId.trim() ? body.connectionId.trim() : null
    const request = { connectionId, enabled: body.enabled === true, profile }
    const run = chain.then(() => syncPreviewSignins(deps, event.sender, request))
    chain = run.catch(() => undefined)

    return run.catch((error: unknown) => empty('error', { error: error instanceof Error ? error.message : String(error) }))
  })
}
