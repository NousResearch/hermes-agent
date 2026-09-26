/**
 * E2E at-rest contract for the remote-gateway session token (issue #77486).
 *
 * The reported bug: configuring a remote gateway persisted the dashboard
 * session token as PLAINTEXT into `connection.json` under the app's userData
 * dir (macOS `~/Library/Application Support/Hermes/connection.json`, Windows
 * `AppData\Roaming\Hermes\connection.json`). Anything that can read the file
 * — a backup, a sync client, another local process, a support bundle — got a
 * live gateway credential.
 *
 * The contract these tests encode is deliberately stated WITHOUT naming a
 * storage strategy:
 *
 *   1. ABSENT FROM DISK. After the app has been configured with a remote
 *      gateway token, the token's plaintext value must not appear anywhere in
 *      `connection.json`, in any sibling file the app writes under userData,
 *      or in HERMES_HOME (logs included).
 *   2. STILL FUNCTIONAL. After a restart, the app must still be able to USE
 *      that credential — it decrypts the stored blob and puts the exact
 *      original token on the wire.
 *   3. UNREADABLE BY OTHER LOCAL ACCOUNTS. `connection.json` must not be
 *      group/other-accessible, whether the app just wrote it or inherited it
 *      from an older install.
 *
 * All three matter and none is sufficient alone. (1) alone is trivially
 * satisfied by a "fix" that drops the token on the floor; (2) alone is
 * satisfied by the bug itself. So (2) is verified through the app's own
 * connection test against a fake gateway that records the
 * `X-Hermes-Session-Token` header it receives — a dropped or mangled token
 * cannot produce that header.
 *
 * (3) is orthogonal to (1) and invisible to it: safeStorage keeps the token
 * opaque no matter what the file's mode is, so a 0644 `connection.json` passes
 * the raw-bytes scan every time while still exposing the ciphertext blob, the
 * gateway URL and the SSH host/user/keyPath to any other local account. It is
 * asserted explicitly (see `expectOwnerOnlyMode`) because no amount of
 * encryption evidence implies it.
 *
 * We deliberately do NOT assert `encoding === 'safeStorage'` or any other
 * shape of the stored blob. That would be a change-detector: a fix that moved
 * to the OS keychain proper, to an async safeStorage provider, or to a
 * separate credential file would break the test while being *more* correct.
 * The load-bearing assertion is the raw-bytes absence of the secret.
 *
 * Three at-rest paths, hence three tests:
 *
 *   1. A NEWLY configured token. The app's own write path routes through the
 *      strict `encryptDesktopSecret`; this test holds it there against
 *      regression, and pins the mode of the file it actually wrote.
 *   2. An EXISTING `connection.json` at the old 0644. Covers the read-side
 *      tighten, and ONLY the mode — its token is already ciphertext.
 *   3. A CORRUPT `connection.json` at 0644. The tighten must not be gated on
 *      the parse succeeding: a truncated file still holds the token bytes, and
 *      the parse failure is swallowed, so nothing would ever come back for it.
 *
 * Legacy plaintext `connection.json` payloads are deliberately NOT migrated
 * yet (blocked on #62319's opt-in marker, writing through the config
 * sanitizer, and surfacing token-rotation guidance).
 *
 * Environment limits are encoded rather than papered over. Electron's
 * safeStorage is unavailable on Linux with no keyring, which is the shape of
 * this suite's CI runner (ubuntu-latest, see .github/workflows/e2e-desktop.yml).
 * The absence assertion is unconditional there — it is the security
 * requirement, and it must hold in every environment. Only the *other* half is
 * conditional: with secure storage the save must succeed, and without it the
 * save must fail loudly (which is what the current strict `encryptDesktopSecret`
 * does) instead of quietly writing plaintext. See the branch comments in each
 * test for the reasoning.
 *
 * Prerequisite: `npm run build` must have been run so dist/ exists.
 */

import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as path from 'node:path'

import { buildAppEnv, createSandbox, launchDesktop, type Sandbox } from './fixtures'
import { allowErrorBanners, type ElectronApplication, expect, type Page, test } from './test'

/**
 * The secret under test. Long, random-looking, and unique to this spec so a
 * raw-bytes scan cannot produce a false negative by colliding with ordinary
 * config content. Kept to `[A-Za-z0-9-]` on purpose: encodeURIComponent() is
 * the identity function over this alphabet, so the raw-bytes needle also
 * covers the URL-encoded form the WS dialer builds (`?token=…`).
 */
const SENTINEL_TOKEN = 'hermes-e2e-at-rest-sentinel-Zq7Z4hV9nX2pL8sK3tB6wR1yM5jD0fG'

/** Skip absurdly large files during the leak scan (Chromium caches). */
const MAX_SCAN_BYTES = 16 * 1024 * 1024

/**
 * One fixed Electron app name for this spec, instead of the timestamped one
 * `buildAppEnv` generates. On macOS the safeStorage keychain item is derived
 * from the app name, so a per-launch name would (a) make the post-restart
 * decrypt fail for the wrong reason and (b) leave a fresh keychain entry on
 * the developer's login keychain on every run. Safe because the suite runs
 * one worker at a time and both launches here are sequential; the
 * single-instance lock keys off userData, which is per-sandbox.
 */
const STABLE_APP_NAME = 'HermesE2EAtRestStorage'

// ─── Fake gateway ───────────────────────────────────────────────────────

interface FakeGateway {
  url: string
  /** Every `X-Hermes-Session-Token` value the app has sent us. */
  sessionTokens: string[]
  close: () => Promise<void>
}

/**
 * A minimal stand-in for a remote Hermes gateway. It serves the public
 * `/api/status` probe (which the desktop connection test hits first, with the
 * session token in a header) and refuses the WebSocket upgrade immediately so
 * the second leg of the connection test fails fast instead of burning the
 * probe's 10s connect timeout. We only care about the header it captured.
 *
 * The e2e mock-server is an OpenAI-compatible *inference* mock, not a gateway,
 * so it cannot answer /api/status — hence this small local server.
 */
async function startFakeGateway(): Promise<FakeGateway> {
  const sessionTokens: string[] = []

  const server = http.createServer((req, res) => {
    const token = req.headers['x-hermes-session-token']

    if (typeof token === 'string' && token) {
      sessionTokens.push(token)
    }

    if (req.url?.startsWith('/api/status')) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ auth_required: false, ok: true, version: '0.0.0-e2e-fake' }))

      return
    }

    res.writeHead(404, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ detail: 'not found' }))
  })

  // Refuse the WS leg at once: the connection test's WS probe should return a
  // fast failure rather than hang. The status header is already captured.
  server.on('upgrade', (req, socket) => {
    const token = new URL(req.url ?? '/', 'http://127.0.0.1').searchParams.get('token')

    if (token) {
      sessionTokens.push(token)
    }

    socket.destroy()
  })

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  const { port } = server.address() as AddressInfo

  return {
    close: () =>
      new Promise<void>(resolve => {
        server.closeAllConnections?.()
        server.close(() => resolve())
      }),
    sessionTokens,
    url: `http://127.0.0.1:${port}`,
  }
}

// ─── On-disk leak scanning ──────────────────────────────────────────────

interface Needle {
  bytes: Buffer
  label: string
}

/**
 * The forms a leak could take. Raw bytes, not JSON.parse + field inspection:
 * the point is that the secret is nowhere in the file — including inside a
 * nested field, a cached WS URL, or a field name nobody thought to check.
 *
 * The base64 needle catches the cheapest wrong "fix": base64 is an encoding,
 * not encryption, so a token that is merely base64'd is still plaintext at
 * rest. A real ciphertext will contain neither needle.
 */
function secretNeedles(secret: string): Needle[] {
  return [
    { bytes: Buffer.from(secret, 'utf8'), label: 'plaintext' },
    { bytes: Buffer.from(Buffer.from(secret, 'utf8').toString('base64'), 'utf8'), label: 'base64' },
  ]
}

/** Relative paths of every file under `root` whose bytes contain a needle. */
function scanTreeForSecret(root: string, needles: Needle[]): string[] {
  const hits: string[] = []

  const walk = (dir: string): void => {
    let entries: fs.Dirent[]

    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }

    for (const entry of entries) {
      const full = path.join(dir, entry.name)

      if (entry.isDirectory()) {
        walk(full)

        continue
      }

      if (!entry.isFile()) {
        continue
      }

      try {
        if (fs.statSync(full).size > MAX_SCAN_BYTES) {
          continue
        }
      } catch {
        continue
      }

      let buf: Buffer

      try {
        buf = fs.readFileSync(full)
      } catch {
        continue
      }

      for (const needle of needles) {
        if (buf.includes(needle.bytes)) {
          hits.push(`${path.relative(root, full)} [${needle.label}]`)
        }
      }
    }
  }

  walk(root)

  return hits
}

/**
 * Read a file's bytes, or an empty buffer when it does not exist. A correct
 * fix is allowed to delete/replace `connection.json` rather than rewrite it,
 * and the refusal path may never create it at all — neither should crash the
 * scan before its assertion runs.
 */
function readIfExists(filePath: string): Buffer {
  try {
    return fs.readFileSync(filePath)
  } catch {
    return Buffer.alloc(0)
  }
}

/**
 * The stored token's `encoding` tag, for diagnostics only — never its value.
 * Reported on failure so a red run says *why* (e.g. still `plain`) instead of
 * only that a scan matched. Deliberately NOT an assertion: which encoding a
 * correct fix chooses is its own business.
 */
function storedTokenEncoding(connectionFile: string): string {
  try {
    const parsed = JSON.parse(readIfExists(connectionFile).toString('utf8'))

    return String(parsed?.remote?.token?.encoding ?? '<none>')
  } catch {
    return '<unparsable>'
  }
}

/**
 * Assert a credential file is not readable or writable by group/other.
 *
 * This is the one contract the raw-bytes scan above structurally cannot see:
 * safeStorage keeps the token opaque regardless of the file's mode, so a
 * world-readable `connection.json` passes every absence assertion in this file
 * while still handing the URL, the SSH host/user/keyPath, and the ciphertext
 * blob to any other local account. Encryption and permissions are independent
 * halves of "at rest", and only one of them was covered here.
 *
 * Asserted as `mode & 0o077 === 0` rather than `=== 0o600`: the requirement is
 * that nobody else can reach the file, and pinning the exact bits would make
 * this a change-detector against a future 0400 or a setgid-dir umask.
 *
 * POSIX only. `tightenSecretFileMode` no-ops on Windows deliberately (Node maps
 * chmod to the read-only bit there, and userData is already ACL'd to the user
 * profile — see the docstring in electron/hardening.ts, and PR #77527 for the
 * one place ACLs are being handled). Mode bits are advisory on Windows, so
 * asserting them would go red for behaviour the fix never claimed. The suite
 * runs ubuntu-latest today (.github/workflows/e2e-desktop.yml); nothing else in
 * this spec is platform-specific, and this assertion should not be what
 * changes that.
 */
function expectOwnerOnlyMode(filePath: string, why: string): void {
  if (process.platform === 'win32') {
    return
  }

  const mode = fs.statSync(filePath).mode & 0o777

  expect(mode & 0o077, `${why} (mode ${mode.toString(8)})`).toBe(0)
}

// ─── App helpers ────────────────────────────────────────────────────────

/**
 * Launch the desktop app against `sandbox` with a fake boot failure injected.
 *
 * The credential path we are testing is entirely main-process (IPC handler →
 * coerce → safeStorage → userData write) and does not need a live agent
 * backend, so we skip spawning `hermes serve` (no Python needed, ~3s launch,
 * hermetic). This is also a real user situation rather than an artificial one:
 * the boot-failure overlay's own recovery affordance is "Connection settings",
 * i.e. pointing the app at a remote gateway is exactly what a user does from
 * this state. BOOT_FAKE_ERROR short-circuits startHermes() *before* remote
 * resolution, so no launch ever dials the fake gateway on its own.
 */
async function launchAgainst(sandbox: Sandbox): Promise<{ app: ElectronApplication; page: Page }> {
  const env = buildAppEnv(sandbox, {
    HERMES_DESKTOP_APP_NAME: STABLE_APP_NAME,
    HERMES_DESKTOP_BOOT_FAKE_ERROR: 'E2E at-rest storage spec: local backend intentionally not started',
  })

  const { app, page } = await launchDesktop(env)

  // The capability bridge is what we drive; it lands with the preload, well
  // before the app would be "ready" in the boot sense.
  await page.waitForFunction(
    () => Boolean((window as unknown as { hermesDesktop?: Record<string, unknown> }).hermesDesktop?.saveConnectionConfig),
    undefined,
    { timeout: 60_000 },
  )

  return { app, page }
}

/**
 * Ask the running app where userData actually is, the same way the app does
 * (`app.getPath('userData')`). The fixtures point userData at a temp sandbox,
 * so a home-relative hardcoded path would test the wrong file — or no file.
 */
async function resolveUserDataDir(app: ElectronApplication): Promise<string> {
  return app.evaluate(({ app: electronApp }) => electronApp.getPath('userData'))
}

interface SafeStorageCapability {
  available: boolean
  backend: string
}

/**
 * What secure storage is actually capable of on THIS host, asked after ready
 * (on Linux the answer is meaningless before then).
 *
 * `backend` matters for the honest reading of a green run: on Linux with no
 * keyring, Electron can still report encryption as available while selecting
 * the `basic_text` backend, which encrypts with a hardcoded password — the
 * bytes on disk are not the plaintext, but they are not meaningfully
 * protected either. We record it rather than assert on it, because which
 * posture Hermes should take there (refuse to save vs. accept basic_text) is
 * a product decision, not something this test should silently ratify.
 */
async function readSafeStorageCapability(app: ElectronApplication): Promise<SafeStorageCapability> {
  return app.evaluate(async ({ app: electronApp, safeStorage }) => {
    await electronApp.whenReady()

    let available = false
    let backend = 'unavailable'

    try {
      available = safeStorage.isEncryptionAvailable()
    } catch {
      available = false
    }

    try {
      // Linux-oriented API; other platforms may not implement it.
      backend = safeStorage.getSelectedStorageBackend?.() ?? 'n/a'
    } catch {
      backend = 'n/a'
    }

    return { available, backend }
  })
}

interface SaveOutcome {
  config: { remoteTokenPreview?: null | string; remoteTokenSet?: boolean; remoteUrl?: string } | null
  error: null | string
}

/**
 * Drive the app's REAL save surface: the same `saveConnectionConfig` payload
 * Settings → Gateway sends (see src/app/settings/gateway-settings.tsx). We use
 * save rather than apply so the app persists the credential without trying to
 * re-home onto the fake gateway.
 */
async function saveRemoteToken(page: Page, remoteUrl: string, remoteToken?: string): Promise<SaveOutcome> {
  return page.evaluate(
    async ([url, token]) => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      try {
        const config = await desktop.saveConnectionConfig({
          mode: 'remote',
          remoteAuthMode: 'token',
          ...(token ? { remoteToken: token } : {}),
          remoteUrl: url,
        })

        return { config, error: null }
      } catch (error) {
        return { config: null, error: error instanceof Error ? error.message : String(error) }
      }
    },
    [remoteUrl, remoteToken ?? ''] as const,
  )
}

/**
 * Make the app USE the stored credential. No token in the payload, so the main
 * process must read `connection.json`, decrypt what it stored, and put the
 * plaintext on the wire itself. `buildRemoteBlock` throws "Remote gateway
 * session token is required." when the stored blob no longer decrypts, so a
 * fix that dropped the token fails here instead of quietly passing the
 * absence assertion.
 */
async function exerciseStoredToken(page: Page, remoteUrl: string): Promise<{ error: null | string }> {
  return page.evaluate(async url => {
    const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

    try {
      await desktop.testConnectionConfig({ mode: 'remote', remoteUrl: url })

      return { error: null }
    } catch (error) {
      // A failing WS leg is expected (the fake gateway refuses the upgrade).
      // The assertion is on what the gateway received, not on this result.
      return { error: error instanceof Error ? error.message : String(error) }
    }
  }, remoteUrl)
}

// ─── Tests ──────────────────────────────────────────────────────────────

let gateway: FakeGateway | null = null
let sandbox: Sandbox | null = null
let app: ElectronApplication | null = null

test.beforeAll(async () => {
  gateway = await startFakeGateway()
})

test.afterAll(async () => {
  await gateway?.close()
  gateway = null
})

test.beforeEach(() => {
  // Boot is intentionally failed in this spec (see launchAgainst), so the
  // boot-failure overlay's error banner is expected, not a failure.
  allowErrorBanners()
})

test.afterEach(async () => {
  await app?.close().catch(() => undefined)
  app = null
  sandbox?.cleanup()
  sandbox = null
})

test.describe('remote gateway session token at rest', () => {
  test('with keychain encryption opted IN, a newly configured token is never written to userData in plaintext, and still works after restart', async () => {
    const fake = gateway!
    sandbox = createSandbox('at-rest-fresh')

    // Keychain-backed encryption is opt-in (default OFF — see
    // electron/secret-storage-policy.ts). This test covers the opted-IN
    // posture, so seed the policy the way the Settings toggle writes it.
    fs.writeFileSync(
      path.join(sandbox.userDataDir, 'secure-token-storage.json'),
      JSON.stringify({ migrated: true, on: true }),
      'utf8',
    )

    const first = await launchAgainst(sandbox)
    app = first.app

    const capability = await readSafeStorageCapability(app)
    const userDataDir = await resolveUserDataDir(app)
    const connectionFile = path.join(userDataDir, 'connection.json')

    test.info().annotations.push({
      description: `isEncryptionAvailable=${capability.available} backend=${capability.backend}`,
      type: 'safeStorage',
    })

    const saved = await saveRemoteToken(first.page, fake.url, SENTINEL_TOKEN)

    // Defined degradation, not a silent plaintext write. Where secure storage
    // works, the save must succeed. Where it genuinely does not (headless
    // Linux with no keyring, per Electron's safeStorage docs), refusing the
    // save with a loud error is an acceptable outcome — what is NEVER
    // acceptable is reporting success while leaving the secret readable on
    // disk. The absence assertion below runs in both branches.
    if (capability.available) {
      expect(
        saved.error,
        'secure storage is available on this host, so saving a remote gateway token must succeed',
      ).toBeNull()
      expect(saved.config?.remoteTokenSet).toBe(true)
    } else {
      expect(
        saved.error,
        'secure storage is unavailable, so the save must fail loudly rather than persist a plaintext token',
      ).not.toBeNull()
    }

    // Guard against a vacuous pass: when the save succeeded, the artifact must
    // exist and must be the file the app really wrote for THIS connection.
    // Without this, "no plaintext on disk" would also be true if nothing had
    // been saved at all. Only asserted on the success branch — a refused save
    // legitimately leaves no file behind.
    const rawConnection = readIfExists(connectionFile)

    if (capability.available) {
      expect(fs.existsSync(connectionFile), `expected the app to write ${connectionFile}`).toBe(true)
      expect(
        rawConnection.includes(Buffer.from(fake.url, 'utf8')),
        'connection.json should record the configured gateway URL (proves this is the real artifact)',
      ).toBe(true)

      // The write path's OTHER half of at-rest: opaque bytes AND owner-only
      // permissions. Deliberately here, on the file this test just proved the
      // app really wrote, rather than in a unit test — nothing in the repo
      // imports electron/main.ts (it imports electron), so this is the only
      // place that can witness the app's own write actually going out at 0600
      // instead of the 0644 umask default.
      expectOwnerOnlyMode(
        connectionFile,
        'connection.json is group/other-accessible, so the encrypted token blob, gateway URL and SSH fields are readable by other local accounts',
      )
    }

    // ── The load-bearing assertion ─────────────────────────────────────
    const needles = secretNeedles(SENTINEL_TOKEN)

    const connectionHits = needles.filter(needle => rawConnection.includes(needle.bytes)).map(needle => needle.label)
    expect(
      connectionHits,
      `the gateway session token must not be recoverable from ${connectionFile} ` +
        `(stored token encoding is "${storedTokenEncoding(connectionFile)}")`,
    ).toEqual([])

    // …and not in any sibling file the app writes alongside it, nor in
    // HERMES_HOME (desktop.log lives there).
    expect(
      scanTreeForSecret(userDataDir, needles),
      'the gateway session token leaked into a userData file',
    ).toEqual([])
    expect(
      scanTreeForSecret(sandbox.hermesHome, needles),
      'the gateway session token leaked into a HERMES_HOME file (logs included)',
    ).toEqual([])

    if (!capability.available) {
      // Nothing was stored, so there is no round trip to verify. The refusal
      // itself was already asserted above.
      return
    }

    // ── Secondary: the credential must still be USABLE ─────────────────
    // Restart against the same userData so the token comes off disk, not out
    // of a live process's memory.
    await app.close().catch(() => undefined)
    app = null

    const second = await launchAgainst(sandbox)
    app = second.app

    expect(
      await resolveUserDataDir(app),
      'the restarted app must resolve the same userData dir, or this is not a round trip',
    ).toBe(userDataDir)

    const reread = await second.page.evaluate(async () => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      return desktop.getConnectionConfig()
    })

    expect(reread.remoteTokenSet, 'the stored token must survive a restart').toBe(true)
    expect(reread.remoteUrl).toBe(fake.url)

    const before = fake.sessionTokens.length
    await exerciseStoredToken(second.page, fake.url)

    // The gateway is the witness: the app decrypted its stored blob and put
    // the original secret on the wire. A dropped, truncated, or re-encoded
    // token cannot produce this.
    expect(
      fake.sessionTokens.slice(before),
      'the app must send the exact stored token to the gateway after a restart',
    ).toContain(SENTINEL_TOKEN)
  })

  /**
   * The DEFAULT posture: keychain encryption opted out (no policy file at
   * all). Saving a token must (a) succeed without ever touching safeStorage
   * — this is the whole point of the opt-in: no macOS Keychain dialog on
   * machines with a broken login keychain — (b) store the token with a
   * non-safeStorage encoding at 0600, and (c) round-trip it across a
   * restart. The plaintext-on-disk trade-off is the user's chosen (default)
   * mode; owner-only file bits remain the at-rest boundary.
   */
  test('with the default policy (no keychain), a token saves without secure storage, is owner-only on disk, and survives a restart', async () => {
    const fake = gateway!
    sandbox = createSandbox('at-rest-default')

    const first = await launchAgainst(sandbox)
    app = first.app

    const userDataDir = await resolveUserDataDir(app)
    const connectionFile = path.join(userDataDir, 'connection.json')

    // Must succeed regardless of host keyring state — the default policy
    // never consults safeStorage, so "no keyring" cannot refuse the save.
    const saved = await saveRemoteToken(first.page, fake.url, SENTINEL_TOKEN)

    expect(saved.error, 'the default (opted-out) policy must save without secure storage').toBeNull()
    expect(saved.config?.remoteTokenSet).toBe(true)

    // Not a safeStorage blob, and owner-only on disk.
    expect(storedTokenEncoding(connectionFile)).not.toBe('safeStorage')
    expectOwnerOnlyMode(
      connectionFile,
      'connection.json is group/other-accessible; owner-only bits are the at-rest boundary for opted-out storage',
    )

    // Round trip across a restart, same witness as the opted-in test.
    await app.close().catch(() => undefined)
    app = null

    const second = await launchAgainst(sandbox)
    app = second.app

    const reread = await second.page.evaluate(async () => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      return desktop.getConnectionConfig()
    })

    expect(reread.remoteTokenSet, 'the stored token must survive a restart').toBe(true)

    const before = fake.sessionTokens.length
    await exerciseStoredToken(second.page, fake.url)

    expect(
      fake.sessionTokens.slice(before),
      'the app must send the exact stored token to the gateway after a restart',
    ).toContain(SENTINEL_TOKEN)
  })

  /**
   * The read side of the same contract: an install written BEFORE the file was
   * owner-only keeps its 0644 bits until something chmods it, and the write
   * path cannot fix it — `fs.writeFileSync(path, data, { mode })` applies
   * `mode` only when it CREATES the file. Waiting for the user's next Settings
   * save would leave the file group/other-readable indefinitely, which is why
   * `readDesktopConnectionConfig` tightens on a cache miss.
   *
   * Scoped to the MODE, and deliberately independent of the deferred legacy
   * plaintext migration. The fixture's token is already safeStorage ciphertext
   * (the app wrote it), so nothing here re-encrypts anything. Tightening a
   * permission bit neither performs a migration nor claims to.
   *
   * The fixture is produced by the app itself rather than hand-written, so the
   * only difference from a real pre-fix install is the one bit under test.
   */
  test('an install whose connection.json predates owner-only mode is tightened on read', async () => {
    const fake = gateway!
    sandbox = createSandbox('at-rest-tighten')

    const first = await launchAgainst(sandbox)
    app = first.app

    const capability = await readSafeStorageCapability(app)

    test.info().annotations.push({
      description: `isEncryptionAvailable=${capability.available} backend=${capability.backend}`,
      type: 'safeStorage',
    })

    if (!capability.available) {
      // Without secure storage the save is refused by design, so there is no
      // app-written artifact to loosen and re-read. The refusal itself is
      // already asserted in the first test.
      test.skip(true, 'secure storage unavailable on this host — no app-written connection.json to tighten')

      return
    }

    const userDataDir = await resolveUserDataDir(app)
    const connectionFile = path.join(userDataDir, 'connection.json')

    const saved = await saveRemoteToken(first.page, fake.url, SENTINEL_TOKEN)
    expect(saved.error, 'the fixture write must succeed, or there is nothing to tighten').toBeNull()

    await app.close().catch(() => undefined)
    app = null

    // Regress the file to what a pre-fix install has on disk. Everything else
    // about it — including the encrypted token — is exactly what the app wrote.
    fs.chmodSync(connectionFile, 0o644)
    expect(fs.statSync(connectionFile).mode & 0o077, 'the fixture must start group/other-accessible').not.toBe(0)

    const seededMtimeMs = fs.statSync(connectionFile).mtimeMs

    // A fresh process starts with an empty config cache, so the first read is a
    // miss and the tighten runs. `getConnectionConfig()` forces that read
    // through the app's own IPC surface.
    const second = await launchAgainst(sandbox)
    app = second.app

    const reread = await second.page.evaluate(async () => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      return desktop.getConnectionConfig()
    })

    expectOwnerOnlyMode(
      connectionFile,
      'a pre-existing world-readable connection.json was not tightened when the app read it',
    )

    // The tighten must be a chmod, not a rewrite. It sits INSIDE the function
    // whose cache keys on mtimeMs, so if it ever moved mtime it would
    // invalidate that cache on every read and re-tighten forever. chmod moves
    // ctime only, which is what makes the placement safe — this pins it.
    expect(
      Math.abs(fs.statSync(connectionFile).mtimeMs - seededMtimeMs),
      'tightening must not rewrite the file: mtime is the config cache key, so moving it would invalidate the cache the tighten sits inside',
    ).toBeLessThan(1)

    // And tightening must not have cost the user their credential — the whole
    // reason this happens on read instead of by deleting the file.
    expect(reread.remoteTokenSet, 'the stored token must survive being tightened').toBe(true)
    expect(reread.remoteUrl).toBe(fake.url)
  })

  /**
   * The tighten must not be gated on the file being valid JSON.
   *
   * A truncated `connection.json` — an interrupted write on an older build, a
   * half-finished hand edit, a partially restored backup — still contains the
   * token bytes, and `JSON.parse` throws straight into the `catch` that falls
   * back to local mode. That fallback is never written back, so nothing
   * re-tightens the file later. With the chmod sequenced AFTER the parse,
   * exactly the file that is both corrupt AND world-readable would be the one
   * file never tightened, permanently.
   *
   * This is the only test that can tell the two orderings apart: every other
   * test here uses a parseable file, where either ordering tightens. Asserting
   * `mode === 'local'` is what makes it load-bearing — it proves the parse
   * really threw, so a green mode assertion cannot be explained by anything
   * downstream of the parse.
   *
   * Needs no secure storage: it is a chmod on a file that is never decrypted,
   * so it holds on the keyring-less CI runner too.
   */
  test('a corrupt connection.json is tightened even though it never parses', async () => {
    sandbox = createSandbox('at-rest-tighten-corrupt')

    const connectionFile = path.join(sandbox.userDataDir, 'connection.json')

    // Truncated mid-token: unparseable, yet the secret bytes are right there.
    fs.writeFileSync(
      connectionFile,
      `{"mode":"remote","remote":{"authMode":"token","token":{"encoding":"plain","value":"${SENTINEL_TOKEN}`,
      { encoding: 'utf8', mode: 0o644 },
    )
    fs.chmodSync(connectionFile, 0o644)
    expect(fs.statSync(connectionFile).mode & 0o077, 'the fixture must start group/other-accessible').not.toBe(0)

    const seededMtimeMs = fs.statSync(connectionFile).mtimeMs

    const launched = await launchAgainst(sandbox)
    app = launched.app

    const reread = await launched.page.evaluate(async () => {
      const desktop = (window as unknown as { hermesDesktop: any }).hermesDesktop

      return desktop.getConnectionConfig()
    })

    expect(
      reread.mode,
      'the fixture must be unparseable, so the app falls back to local — otherwise this test proves nothing about ordering',
    ).toBe('local')

    expectOwnerOnlyMode(
      connectionFile,
      'a corrupt world-readable connection.json still holding token bytes was left group/other-accessible',
    )

    // Same cache invariant as above: chmod, not rewrite.
    expect(
      Math.abs(fs.statSync(connectionFile).mtimeMs - seededMtimeMs),
      'tightening must not rewrite the file: mtime is the config cache key',
    ).toBeLessThan(1)
  })
})
