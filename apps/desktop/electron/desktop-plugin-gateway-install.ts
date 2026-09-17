/**
 * Install a Desktop UI half the CONNECTED gateway serves into this app's own
 * app-level `desktop-plugins` root.
 *
 * The desktop app extends THIS APP, so its plugin root is local to the machine
 * the app runs on (`<hermes home>/desktop-plugins`, resolved by Electron —
 * #66899). Against a REMOTE backend (SSH / URL+token / cloud) the machine that
 * holds the plugin is not the machine that runs the app, so the unified-package
 * copy step (`desktop-plugins-root.ts`) has nothing local to copy and the half
 * never appears: every dashboard/desktop plugin built on the backend strands
 * there and the owner hand-copies a file to their laptop.
 *
 * The bytes come from the gateway's `plugins.manage desktop_half` RPC — the
 * SAME session-authenticated channel whose `list` reports `has_desktop_half`,
 * never the dashboard's static asset route (`/dashboard-plugins/<name>/…`),
 * which is deliberately unauthenticated because the SPA loads plugin JS with
 * `<script src>` and so cannot carry an auth header. Pulling evaluated CODE
 * from an unauthenticated route would turn any local writer of that route into
 * renderer code execution — this module therefore never fetches, it only
 * VERIFIES and WRITES what an authenticated caller hands it.
 *
 * Electron-free and pure so the write/refuse contract is unit-testable.
 */

import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

/** Marker beside a half this app pulled from a gateway. Its presence is what
 *  lets a later pull REFRESH the folder without touching a folder the user
 *  installed by hand. */
export const GATEWAY_HALF_MARKER = '.hermes-gateway.json'

/** Refuse an absurd payload before touching the disk. Real halves are tens of
 *  KB (Mission Control's is 50 KB); the cap is the loader's own read limit. */
export const GATEWAY_HALF_MAX_BYTES = 16 * 1024 * 1024

export interface GatewayHalfMarker {
  bytes: number
  /** Plugin name/key on the gateway, for provenance in the UI. */
  key?: string
  name: string
  /** Gateway-side provenance ('user' | 'git' | 'bundled'). */
  source?: string
  /** sha256 of `plugin.js` as written. */
  sha256: string
  installedAt: string
}

export interface GatewayHalfInstallOptions {
  /** Plugin name on the gateway (also the local folder name). */
  name: string
  /** Expected sha256 of `text`, as reported by the gateway. Absent = compute it. */
  sha256?: null | string
  /** Gateway-side provenance, copied into the marker. */
  source?: null | string
  /** Registry key on the gateway, copied into the marker. */
  key?: null | string
  /** Overwrite a folder that has no gateway marker (a hand-made install). */
  force?: boolean
  /** App-level `desktop-plugins` root (Electron's desktopPluginsRoot()). */
  root: string
  /** The half's source text, exactly as the gateway served it. */
  text: string
}

export type GatewayHalfRefusal =
  | 'already-current'
  | 'exists'
  | 'integrity'
  | 'invalid'
  | 'io'
  | 'oversize'
  | 'unavailable'

export interface GatewayHalfInstallResult {
  error?: string
  /** True when the local copy was already this exact revision. */
  unchanged?: boolean
  ok: boolean
  path?: string
  reason?: GatewayHalfRefusal
  sha256?: string
}

/**
 * A plugin name is a single path segment: the gateway's need not be, so the
 * local folder name is derived here and anything that could escape the root
 * (separators, `..`, an absolute path, a leading dot) is refused rather than
 * sanitized — a silent rename would write a folder the plugin's own id no
 * longer matches.
 */
export function gatewayHalfFolderName(raw: unknown): null | string {
  const name = String(raw ?? '').trim()

  if (!name || name.length > 128) {
    return null
  }

  if (name === '.' || name === '..' || name.startsWith('.') || /[/\\\0]/.test(name)) {
    return null
  }

  return name
}

export function sha256Hex(text: string): string {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex')
}

/** A half is EVALUATED as ESM in the renderer with the app's own authority.
 *  An error page, a proxy interstitial or an empty body must never be written
 *  as `plugin.js` — the loader would then report a syntax error (or worse, run
 *  part of something else) against a file that looks installed. */
function looksLikePluginModule(text: string): boolean {
  const head = text.slice(0, 4096).toLowerCase()

  if (head.includes('<!doctype html') || head.includes('<html')) {
    return false
  }

  return /(^|\n)\s*export\s+default\b/.test(text)
}

async function readMarker(dir: string): Promise<GatewayHalfMarker | null> {
  try {
    const parsed = JSON.parse(await fs.promises.readFile(path.join(dir, GATEWAY_HALF_MARKER), 'utf8'))

    return parsed && typeof parsed.sha256 === 'string' ? (parsed as GatewayHalfMarker) : null
  } catch {
    return null
  }
}

/** The revision this app last pulled for `name`, or null when the folder was
 *  not installed by this path. */
export async function readGatewayHalfMarker(root: string, name: string): Promise<GatewayHalfMarker | null> {
  const folder = gatewayHalfFolderName(name)

  return folder ? readMarker(path.join(root, folder)) : null
}

/** Every half this app pulled from a gateway, keyed by folder name — the
 *  renderer's "what is already here" map for the install rows. */
export async function readInstalledGatewayHalves(root: string): Promise<Record<string, GatewayHalfMarker>> {
  let entries: fs.Dirent[]

  try {
    entries = await fs.promises.readdir(root, { withFileTypes: true })
  } catch {
    return {}
  }

  const out: Record<string, GatewayHalfMarker> = {}

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    const marker = await readMarker(path.join(root, entry.name))

    if (marker) {
      out[entry.name] = marker
    }
  }

  return out
}

/**
 * Write one gateway-served half into the app-level root.
 *
 * Refusals are NAMED so the caller can say something true:
 *  - `invalid`         — the name is not a single safe path segment;
 *  - `oversize`        — the payload exceeds the cap;
 *  - `integrity`       — the text does not hash to the sha256 the gateway reported;
 *  - `unavailable`     — the text is empty / not a plugin module;
 *  - `exists`          — a folder is already there that this path did not install
 *                        (pass `force` to replace it);
 *  - `io`              — the write itself failed.
 *
 * A folder WITH this marker is refreshed (that is the point: the gateway
 * advertises a newer half), and a refresh whose digest already matches is
 * `unchanged` so a repeat pull is a no-op.
 */
export async function installGatewayDesktopHalf(
  options: GatewayHalfInstallOptions
): Promise<GatewayHalfInstallResult> {
  const folder = gatewayHalfFolderName(options.name)

  if (!folder) {
    return { ok: false, reason: 'invalid', error: `"${String(options.name)}" is not a usable plugin folder name` }
  }

  const text = String(options.text ?? '')
  const bytes = Buffer.byteLength(text, 'utf8')

  if (bytes === 0) {
    return { ok: false, reason: 'unavailable', error: 'the gateway returned an empty desktop half' }
  }

  if (bytes > GATEWAY_HALF_MAX_BYTES) {
    return {
      ok: false,
      reason: 'oversize',
      error: `the desktop half is ${bytes} bytes — larger than the ${GATEWAY_HALF_MAX_BYTES}-byte cap`
    }
  }

  const digest = sha256Hex(text)
  const expected = String(options.sha256 ?? '').trim().toLowerCase()

  if (expected && expected !== digest) {
    return {
      ok: false,
      reason: 'integrity',
      error: `the desktop half does not match the digest the gateway reported (${digest} != ${expected})`
    }
  }

  if (!looksLikePluginModule(text)) {
    return { ok: false, reason: 'unavailable', error: 'the gateway returned something that is not a plugin module' }
  }

  const target = path.join(options.root, folder)
  const existing = await readMarker(target)
  const existedBefore = fs.existsSync(target)

  if (existedBefore) {
    if (!existing && !options.force) {
      return {
        ok: false,
        reason: 'exists',
        error: `"${folder}" is already installed on this machine and was not installed from a gateway`
      }
    }

    if (existing && existing.sha256 === digest) {
      return { ok: true, unchanged: true, path: target, sha256: digest }
    }
  }

  const marker: GatewayHalfMarker = {
    bytes,
    installedAt: new Date().toISOString(),
    key: options.key ?? undefined,
    name: folder,
    sha256: digest,
    source: options.source ?? undefined
  }

  // Land the code first, then the marker: a crash between the two leaves a
  // half without provenance (the next pull REFUSES it unless forced) rather
  // than a marker describing bytes that are not there.
  const temp = path.join(target, `.plugin.js.${crypto.randomBytes(4).toString('hex')}.part`)

  try {
    await fs.promises.mkdir(target, { recursive: true })

    await fs.promises.writeFile(temp, text, 'utf8')
    await fs.promises.rename(temp, path.join(target, 'plugin.js'))
    await fs.promises.writeFile(
      path.join(target, GATEWAY_HALF_MARKER),
      JSON.stringify(marker, null, 2) + '\n',
      'utf8'
    )
  } catch (error) {
    // A partial folder must not read as an installed half — but only remove what
    // THIS call created. `mkdir` can fail because a FILE already sits at the
    // target (a plugin name that happens to collide with one), and deleting the
    // user's file to clean up after a failed install would be a data loss of its
    // own.
    await fs.promises.rm(temp, { force: true }).catch(() => undefined)

    if (!existedBefore) {
      await fs.promises.rm(target, { force: true, recursive: true }).catch(() => undefined)
    }

    return { ok: false, reason: 'io', error: error instanceof Error ? error.message : String(error) }
  }

  return { ok: true, path: target, sha256: digest }
}
