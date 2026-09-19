/**
 * Persisted Chromium command-line switches for the Desktop app.
 *
 * `HERMES_DESKTOP_DISABLE_GPU` and the remote-display / WSL detection cover
 * the cases we can recognise, but a GPU driver that crashes the renderer on a
 * particular box is not one of them — and an app launched from Finder, the
 * Start menu or a Store package has no command line and no shell environment
 * to put `--disable-gpu` in. Anthropic's Claude desktop app solved the same
 * problem with a `chromiumFlags` key in its config file (v2.110.0 changelog);
 * this is the Hermes equivalent: `chromium-flags.json` under userData is read
 * BEFORE app `ready` (switches only apply pre-launch) and appended verbatim.
 *
 *   { "flags": ["--disable-gpu", "--disable-gpu-compositing"] }
 *
 * The crash page writes this file when the user picks "Restart with software
 * rendering", so the recovery survives the relaunch and every launch after it.
 * The file lives in userData, i.e. it carries exactly the trust of a user
 * editing their own launch arguments; it is never written from a renderer.
 *
 * Pure + injectable (fs passed in) so it is testable without Electron.
 */

import fs from 'node:fs'
import path from 'node:path'

export const CHROMIUM_FLAGS_FILENAME = 'chromium-flags.json'
export const SOFTWARE_RENDERING_FLAGS = ['--disable-gpu', '--disable-gpu-compositing'] as const

/** Hard cap: a runaway file cannot turn the command line into a novel. */
const MAX_FLAGS = 32

export interface ChromiumFlagsFsLike {
  readFileSync: (file: string, encoding: BufferEncoding) => string
  writeFileSync: (file: string, data: string) => void
  mkdirSync: (dir: string, options: { recursive: boolean }) => unknown
}

export function chromiumFlagsPath(userData: string): string {
  return path.join(userData, CHROMIUM_FLAGS_FILENAME)
}

/**
 * Keep only entries that look like a Chromium switch (`--name` or
 * `--name=value`, no whitespace or control characters). Anything else is
 * dropped rather than passed through: a typo must not become an argv entry
 * that Chromium interprets as a URL or a file path.
 */
export function sanitizeChromiumFlags(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return []
  }

  const seen = new Set<string>()
  const flags: string[] = []

  for (const entry of raw) {
    if (typeof entry !== 'string') {
      continue
    }

    const flag = entry.trim()

    if (!/^--[A-Za-z0-9][A-Za-z0-9_-]*(=\S*)?$/.test(flag) || seen.has(flag)) {
      continue
    }

    seen.add(flag)
    flags.push(flag)

    if (flags.length >= MAX_FLAGS) {
      break
    }
  }

  return flags
}

/** Flags from `chromium-flags.json`; a missing or malformed file yields none. */
export function readChromiumFlags(userData: string, fsLike: ChromiumFlagsFsLike = fs): string[] {
  let text: string

  try {
    text = fsLike.readFileSync(chromiumFlagsPath(userData), 'utf8')
  } catch {
    return []
  }

  try {
    const parsed: unknown = JSON.parse(text)
    const flags = parsed && typeof parsed === 'object' ? (parsed as { flags?: unknown }).flags : undefined

    return sanitizeChromiumFlags(flags)
  } catch {
    return []
  }
}

/** Persist `flags` (sanitized, deduped) for every future launch. */
export function writeChromiumFlags(userData: string, flags: readonly string[], fsLike: ChromiumFlagsFsLike = fs): string[] {
  const clean = sanitizeChromiumFlags([...flags])

  fsLike.mkdirSync(userData, { recursive: true })
  fsLike.writeFileSync(chromiumFlagsPath(userData), `${JSON.stringify({ flags: clean }, null, 2)}\n`)

  return clean
}

/** Split `--name=value` into the pair `app.commandLine.appendSwitch` wants. */
export function splitChromiumFlag(flag: string): { name: string; value?: string } {
  const body = flag.replace(/^--/, '')
  const eq = body.indexOf('=')

  return eq === -1 ? { name: body } : { name: body.slice(0, eq), value: body.slice(eq + 1) }
}

/** True when the persisted flags (or the live argv) already turn the GPU off. */
export function hasSoftwareRenderingFlag(flags: readonly string[]): boolean {
  return flags.some(flag => splitChromiumFlag(flag).name === 'disable-gpu')
}

/** Persisted flags plus the software-rendering pair, without duplicates. */
export function withSoftwareRenderingFlags(flags: readonly string[]): string[] {
  return sanitizeChromiumFlags([...flags, ...SOFTWARE_RENDERING_FLAGS])
}
