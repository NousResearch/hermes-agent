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
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { type CookiesSetDetails, ipcMain, type Session, session } from 'electron'

import { previewPartitionForProfile } from './preview-guest-throttle'
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
  /** Host-only (no leading-dot domain) vs a domain cookie: a same-name pair can coexist, so the
   *  removal must target the exact one it imported, never url+name (which removes both). */
  hostOnly: boolean
  /** SHA-256 of the imported value (NEVER the value itself — the record is written to userData and
   *  must stay value-free), so a cookie the user has since replaced by hand reads as no-longer-ours. */
  valueHash: string
}

/** Hash a cookie value for the ownership record. A fixed salt separates it from any other digest use. */
function hashCookieValue(value: string): string {
  return createHash('sha256').update(`preview-signins\u0000${value}`).digest('hex')
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
  return {
    url: details.url,
    name: details.name || '',
    domain: details.domain || '',
    path: details.path || '/',
    hostOnly: !details.domain,
    valueHash: hashCookieValue(details.value ?? '')
  }
}

const identityKey = (id: CookieIdentity) => `${id.url}\u0000${id.name}\u0000${id.domain}\u0000${id.path}\u0000${id.hostOnly ? 'h' : 'd'}`

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

/**
 * True when the cookie currently in the jar still matches what we imported for `id`: a host-only vs
 * domain distinction on top of url+name, and the SAME value we wrote. A cookie the user has since
 * replaced by hand (different value) is no-longer-ours and must be left alone.
 */
async function stillOurs(jar: Session, id: CookieIdentity): Promise<boolean> {
  let live: Electron.Cookie[]

  try {
    live = await jar.cookies.get({ url: id.url, name: id.name })
  } catch {
    return false
  }

  return live.some(c => {
    const liveHostOnly = (c as { hostOnly?: boolean }).hostOnly === true || !String(c.domain ?? '').startsWith('.')

    return hashCookieValue(c.value) === id.valueHash && liveHostOnly === id.hostOnly && (c.path || '/') === id.path
  })
}

/**
 * Remove each identity that is still ours (value unchanged) and not in `keep`. Returns the identities
 * that could NOT be removed (a `cookies.remove` failure) so the caller retains the claim: a retry must
 * still find them, never orphan a cookie by dropping its ownership record while it is installed.
 */
async function removeIdentities(jar: Session, ids: CookieIdentity[], keep: Set<string>): Promise<{ removed: number; retained: CookieIdentity[] }> {
  let removed = 0
  const retained: CookieIdentity[] = []

  for (const id of ids) {
    if (keep.has(identityKey(id))) {
      retained.push(id)

      continue
    }

    // The user manually replaced this cookie: it is no longer what we imported, so leave it AND drop
    // the claim (we no longer own it). Treat a gone cookie as removed.
    if (!(await stillOurs(jar, id))) {
      removed += 1

      continue
    }

    try {
      await jar.cookies.remove(id.url, id.name)
      removed += 1
    } catch {
      // Could not remove it: keep the claim so a later clear/re-sync retries and the cookie never
      // lingers unowned.
      retained.push(id)
    }
  }

  return { removed, retained }
}

export async function clearImportedSignins(jar: Session, store: PreviewSigninsRecord, profile: string): Promise<number> {
  const record = store.read()
  const mine = Array.isArray(record[profile]) ? record[profile] : []

  if (!mine.length && !(profile in record)) {
    return 0
  }

  const { removed, retained } = await removeIdentities(jar, mine, new Set())

  if (retained.length) {
    // A removal failed: keep this profile's claim on exactly the cookies still installed so a retry
    // can finish the job. Dropping the key here would orphan them.
    store.write({ ...store.read(), [profile]: retained })
  } else {
    const { [profile]: _dropped, ...rest } = store.read()
    store.write(rest)
  }

  return removed
}

/** Replace `profile`'s imported set with `cookies`: record the intended identities BEFORE touching
 *  the jar (so a record-write failure installs nothing), set each, then reconcile the record to what
 *  actually landed and drop identities this profile no longer has. */
export async function applyImportedSignins(
  jar: Session,
  store: PreviewSigninsRecord,
  profile: string,
  cookies: ExportedCookie[]
): Promise<{ imported: number; failed: number }> {
  const record = store.read()
  const previous = Array.isArray(record[profile]) ? record[profile] : []

  const intended: { details: CookiesSetDetails; id: CookieIdentity }[] = []
  let failed = 0

  for (const cookie of cookies) {
    const details = cookieSetDetails(cookie)

    if (!details) {
      failed += 1

      continue
    }

    intended.push({ details, id: identityOf(details) })
  }

  // Write-ahead: name every cookie we are ABOUT to install (union with the previous claim) before any
  // jar mutation. If this throws, no cookie is set, so nothing is ever installed-but-unowned. Only
  // after this durably lands do we touch the jar.
  const intendedIds = intended.map(({ id }) => id)
  store.write({ ...record, [profile]: dedupeById([...previous, ...intendedIds]) })

  const now: CookieIdentity[] = []

  for (const { details, id } of intended) {
    try {
      await jar.cookies.set(details)
      now.push(id)
    } catch {
      failed += 1
    }
  }

  // Reconcile: the claim is now exactly what landed plus anything we failed to remove from the old set.
  // `retained` includes each previous identity still ours but KEPT (its key is in `current`): that is
  // the OLD record for a cookie whose value just changed, carrying the stale valueHash. `now` must win
  // the dedupe so the claim records the freshly-installed hash — otherwise a later consent-off's
  // `stillOurs` compares the live (new) value against the old hash, rejects its own import, and drops
  // the claim without removing the cookie (installed-but-unowned). Freshly-imported records last.
  const current = new Set(now.map(identityKey))
  const { retained } = await removeIdentities(jar, previous, current)
  store.write({ ...store.read(), [profile]: dedupeById([...retained, ...now]) })

  return { imported: now.length, failed }
}

function dedupeById(ids: CookieIdentity[]): CookieIdentity[] {
  const seen = new Map<string, CookieIdentity>()

  for (const id of ids) {
    seen.set(identityKey(id), id)
  }

  return [...seen.values()]
}

export interface PreviewSigninsDeps {
  /** The cookie jar for one owning profile. Each profile has its OWN partition (P1-3): a profile's
   *  import, consent-off and rename/delete touch only its jar, never another profile's. */
  session: (profile: string) => Session
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

/** The record key for a profile: the SAME normalized key its partition is derived from, so the record,
 *  the jar and the webview all agree on one identity per profile. */
export function previewRecordKey(profile: string | null | undefined): string {
  return (profile ?? '').trim() || 'default'
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
  const key = previewRecordKey(profile)
  const jar = deps.session(key)

  // A remote backend never imported (its profile is another machine's), so it has nothing to clear either;
  // checking first keeps a remote `default` from clearing what the local `default` imported.
  if (!local) {
    return empty('remote')
  }

  if (!request.enabled) {
    const cleared = await clearImportedSignins(jar, deps.store, key)

    if (cleared) {
      deps.log(`[preview-signins] consent off for "${key}": removed ${cleared} imported cookie(s)`)
    }

    return empty('cleared', { cleared })
  }

  const verdict = await deps.exportSignins(profile)

  if (verdict.status === 'disabled') {
    const cleared = await clearImportedSignins(jar, deps.store, key)

    return empty('disabled', { cleared })
  }

  if (verdict.status === 'error') {
    deps.log(`[preview-signins] import for "${key}" failed: ${verdict.error}`)

    return empty('error', { error: verdict.error })
  }

  const { imported, failed } = await applyImportedSignins(jar, deps.store, key, verdict.cookies)
  deps.log(`[preview-signins] imported ${imported} cookie(s) for "${key}" (${failed} failed, ${verdict.skipped ?? 0} skipped)`)

  return empty('imported', { imported, failed, skipped: verdict.skipped ?? 0 })
}

/** Rename hook (P2-7): move a profile's imported-sign-ins record and jar from old→new, following the
 *  `migrateTilesForProfile`/`migratePreviewTabsForProfile` family. Cookies move by re-setting them on
 *  the new jar from the recorded identities (values are not in the record, so a jar-to-jar copy reads
 *  the live cookie) and clearing the old jar. */
export async function migratePreviewSigninsForProfile(
  deps: Pick<PreviewSigninsDeps, 'session' | 'store' | 'log'>,
  oldProfile: string,
  newProfile: string
): Promise<void> {
  const from = previewRecordKey(oldProfile)
  const to = previewRecordKey(newProfile)

  if (from === to) {
    return
  }

  const record = deps.store.read()
  const mine = Array.isArray(record[from]) ? record[from] : []

  if (!mine.length && !(from in record)) {
    return
  }

  const oldJar = deps.session(from)
  const newJar = deps.session(to)
  const moved: CookieIdentity[] = []

  for (const id of mine) {
    try {
      const live = await oldJar.cookies.get({ url: id.url, name: id.name })
      const match = live.find(c => hashCookieValue(c.value) === id.valueHash && (c.path || '/') === id.path)

      if (match) {
        await newJar.cookies.set({
          url: id.url,
          name: id.name,
          value: match.value,
          path: id.path,
          ...(id.hostOnly ? {} : { domain: id.domain }),
          secure: Boolean(match.secure),
          httpOnly: Boolean(match.httpOnly),
          // Carry the cross-site restriction (F6): dropping it reverts a `Strict`/`Lax`
          // cookie to Electron's default, changing whether embedded sign-in requests send
          // it. Electron's Cookie.sameSite is already in cookies.set's lowercase enum.
          ...(match.sameSite ? { sameSite: match.sameSite } : {}),
          ...(typeof match.expirationDate === 'number' ? { expirationDate: match.expirationDate } : {})
        })
        moved.push(id)
      }
    } catch {
      // A cookie we could not read/copy is left behind; it is cleared from the old jar below.
    }
  }

  await removeIdentities(oldJar, mine, new Set())

  const base = deps.store.read()
  const { [from]: _gone, ...rest } = base
  deps.store.write({ ...rest, [to]: dedupeById([...(Array.isArray(base[to]) ? base[to] : []), ...moved]) })
  deps.log(`[preview-signins] migrated ${moved.length} imported cookie(s) "${from}" → "${to}"`)
}

/** Delete hook (P2-7): drop a profile's record and clear its jar, following the
 *  `dropTilesForProfile`/`dropPreviewTabsForProfile` family. */
export async function dropPreviewSigninsForProfile(
  deps: Pick<PreviewSigninsDeps, 'session' | 'store' | 'log'>,
  profile: string
): Promise<void> {
  const key = previewRecordKey(profile)
  const cleared = await clearImportedSignins(deps.session(key), deps.store, key)

  if (cleared) {
    deps.log(`[preview-signins] profile "${key}" deleted: removed ${cleared} imported cookie(s)`)
  }
}

/** main.ts registration: the partition and the record file are owned here; main supplies route + spawn. */
export function registerPreviewSignins(opts: {
  userData: string
  resolveRoute: PreviewSigninsDeps['resolveRoute']
  exportSignins: PreviewSigninsDeps['exportSignins']
  log: PreviewSigninsDeps['log']
}): void {
  registerPreviewSigninsIpc(ipcMain, {
    session: (profile: string) => session.fromPartition(previewPartitionForProfile(profile)),
    store: new PreviewSigninsRecord(path.join(opts.userData, PREVIEW_SIGNINS_RECORD_FILE)),
    resolveRoute: opts.resolveRoute,
    exportSignins: opts.exportSignins,
    log: opts.log
  })
}

export function registerPreviewSigninsIpc(ipc: Pick<Electron.IpcMain, 'handle'>, deps: PreviewSigninsDeps): void {
  // One sync/rename/delete at a time: a toggle flip during an import must land after it, never
  // interleave with it, and a rename/delete must not race an in-flight import of the same profile.
  let chain: Promise<unknown> = Promise.resolve()

  const serialize = <T>(work: () => Promise<T>): Promise<T> => {
    const run = chain.then(work)
    chain = run.catch(() => undefined)

    return run
  }

  ipc.handle('hermes:preview-signins:sync', (event, payload: unknown) => {
    const body = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
    const profile = typeof body.profile === 'string' && body.profile.trim() ? body.profile.trim() : null
    const connectionId = typeof body.connectionId === 'string' && body.connectionId.trim() ? body.connectionId.trim() : null
    const request = { connectionId, enabled: body.enabled === true, profile }

    return serialize(() => syncPreviewSignins(deps, event.sender, request)).catch((error: unknown) =>
      empty('error', { error: error instanceof Error ? error.message : String(error) })
    )
  })

  // P2-7: the renderer's local profile rename/delete flow (session-states.ts, beside
  // migrate/dropTilesForProfile) calls these so the per-profile cookie record + jar — which live in
  // Electron main (userData) — follow the profile. Fire-and-forget: the renderer does not block on it.
  ipc.handle('hermes:preview-signins:profile-renamed', (_event, payload: unknown) => {
    const body = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
    const oldProfile = typeof body.oldProfile === 'string' ? body.oldProfile : ''
    const newProfile = typeof body.newProfile === 'string' ? body.newProfile : ''

    return serialize(() => migratePreviewSigninsForProfile(deps, oldProfile, newProfile)).then(
      () => ({ ok: true }),
      (error: unknown) => {
        deps.log(`[preview-signins] rename migration failed: ${error instanceof Error ? error.message : String(error)}`)

        return { ok: false }
      }
    )
  })

  ipc.handle('hermes:preview-signins:profile-deleted', (_event, payload: unknown) => {
    const body = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}
    const profile = typeof body.profile === 'string' ? body.profile : ''

    return serialize(() => dropPreviewSigninsForProfile(deps, profile)).then(
      () => ({ ok: true }),
      (error: unknown) => {
        deps.log(`[preview-signins] delete cleanup failed: ${error instanceof Error ? error.message : String(error)}`)

        return { ok: false }
      }
    )
  })
}
