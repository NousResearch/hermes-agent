// msix-shared.mjs — the shared MSIX-distribution building blocks used by
// BOTH the out-of-store feed generator (apps/desktop/scripts/gen-appinstaller.mjs)
// and the release job that stages the feed (scripts/stage-msixbundle.mjs).
//
// The two call sites must agree on every name/URL that Windows keys on — the
// .appinstaller's MainBundle identity and the bundle URI — so the XML
// builder and the version/filename derivations live here, once.

import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

// The out-of-store MSIX publisher — the ATS signing cert subject, which is
// what Windows compares against the package manifest publisher at install.
// Mirrored from electron-builder.config.cjs so the .appinstaller and the
// manifest can never drift.
export const OUT_OF_STORE_PUBLISHER =
  'CN=Nous Research Inc., O=Nous Research Inc., L=Austin, S=Texas, C=US'

// Content-Type for MSIX / App Installer artifacts. Without the right MIME the
// browser cannot hand a clicked .appinstaller / .msixbundle to the OS App
// Installer (it would download as octet-stream instead). Everything else
// stays octet-stream (R2's default) unchanged. Keys match by filename suffix,
// case-insensitively.
const CONTENT_TYPES = require('./release-content-types.json')

/**
 * The Content-Type to store for a staged release artifact, if any.
 *
 * Keys starting with '.' (or containing one, like 'release.gpg') match by
 * filename suffix. Extensionless keys (inrelease/release/packages — the apt
 * repo metadata) match by exact basename only, so 'foo-release' or
 * 'xrelease' never collide with the apt 'Release' file.
 * @param {string} filename
 * @returns {string | undefined}
 */
export function contentTypeFor(filename) {
  const lower = String(filename).toLowerCase()
  const base = lower.slice(lower.lastIndexOf('/') + 1)
  for (const [key, mime] of Object.entries(CONTENT_TYPES)) {
    if (key.includes('.')) {
      if (lower.endsWith(key)) return mime
    } else if (base === key) {
      return mime
    }
  }
  return undefined
}

/**
 * @param {unknown} value any value to XML-escape
 * @returns {string}
 */
function escapeAttr(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

/**
 * Build an .appinstaller document for a channel.
 *
 * @param {{
 *   baseUrl: string             // feed host root (no trailing slash)
 *   variantChannelPath: string  // e.g. "win32/", "win32/light/", "win32/canary/"
 *   identityName: string        // package Identity Name (e.g. "NousResearch.HermesBundled")
 *   version: string             // 4-part MSIX version, e.g. "1.2.3.0"
 *   bundleFilename: string      // the universal .msixbundle filename in the feed dir
 *   descriptorFilename?: string // defaults to the channel's .appinstaller name
 * }} o
 * @returns {string} the .appinstaller XML
 */
export function buildAppInstaller(o) {
  const directory = [o.baseUrl.replace(/\/+$/, ''), o.variantChannelPath.replace(/^\/+|\/+$/g, '')].filter(Boolean).join('/')
  const bundleUrl = `${directory}/${o.bundleFilename}`
  const descriptor = o.descriptorFilename || `${o.variantChannelPath.replace(/\/+$/, '').split('/').pop()}.appinstaller`
  const appinstallerUri = `${directory}/${descriptor}`

  return [
    '<?xml version="1.0" encoding="utf-8"?>',
    '<AppInstaller',
    `  Uri="${escapeAttr(appinstallerUri)}"`,
    `  Version="${escapeAttr(o.version)}"`,
    '  xmlns="http://schemas.microsoft.com/appx/appinstaller/2017/2">',
    '  <MainBundle',
    `    Name="${escapeAttr(o.identityName)}"`,
    `    Publisher="${escapeAttr(OUT_OF_STORE_PUBLISHER)}"`,
    `    Version="${escapeAttr(o.version)}"`,
    `    Uri="${escapeAttr(bundleUrl)}" />`,
    '  <UpdateSettings>',
    '    <OnLaunch HoursBetweenUpdateChecks="12" />',
    '  </UpdateSettings>',
    '</AppInstaller>',
    ''
  ].join('\n')
}

// The canary tag base + embedded UTC stamp: v0.27.2-canary.20260829034013
// (8- or 14-digit; the shorter legacy form is midnight of that day). The
// base is the next PATCH over the newest stable, so the first three MSIX
// components come from it (0.27.2) and outversion the stable line
// structurally — cross-line monotonicity is free.
const CANARY_TAG_RE = /^v(\d+\.\d+\.\d+)-canary\.(20\d{6}(?:\d{6})?)$/
const STABLE_TAG_RE = /^v\d+\.\d+\.\d+$/

// MSIX version components are 16-bit (makeappx rejects >65535). Minutes
// since the last stable cross it at 45.5 days; a canary cut more than 45
// days after its stable is a process failure worth surfacing loudly, not a
// number to clamp (a clamped number would break monotonicity).
const MAX_BUILD_MINUTES = 45 * 24 * 60

/**
 * @param {string} stamp YYYYMMDD[HHMMSS] UTC stamp
 * @returns {number} epoch seconds
 */
function stampToEpoch(stamp) {
  const parts = /^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?$/.exec(stamp)
  if (!parts) return 0
  const [, y, mo, d, h, mi, s] = parts
  return Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h ?? 0), Number(mi ?? 0), Number(s ?? 0)) / 1000
}

/**
 * List git tags matching `pattern`, newest-first (git's -v:refname sort).
 * @param {string} gitRoot the repo root
 * @param {string} pattern git tag glob, e.g. "v0.27.*"
 * @returns {string[]}
 */
export function listGitTags(gitRoot, pattern) {
  return execFileSync('git', ['tag', '--list', pattern, '--sort=-v:refname'], { cwd: gitRoot, encoding: 'utf8' })
    .split('\n').filter(Boolean)
}

/**
 * The commit time (epoch seconds) of `tag`, for minutes-since-stable math.
 * @param {string} gitRoot the repo root
 * @param {string} tag a git tag
 * @returns {number}
 */
export function gitTagCommitTime(gitRoot, tag) {
  return Number(execFileSync('git', ['log', '-1', '--format=%ct', tag], { cwd: gitRoot, encoding: 'utf8' }).trim())
}

/**
 * Minutes between a canary tag's UTC stamp and the given stable epoch —
 * the MSIX 4th version component. Null for a stable tag; throws when the
 * stable base is older than 45 days (16-bit component would overflow).
 * @param {string} tag the release tag
 * @param {number} stableEpoch stable tag commit time, epoch seconds
 * @returns {number | null}
 */
export function canaryBuildMinutesFor(tag, stableEpoch) {
  const m = CANARY_TAG_RE.exec(String(tag || ''))
  if (!m) return null
  const minutes = Math.floor((stampToEpoch(m[2]) - stableEpoch) / 60)
  if (minutes < 0) return 0
  if (minutes > MAX_BUILD_MINUTES) {
    throw new Error(
      `canary ${tag} is ${Math.floor(minutes / 1440)} days past its stable base — ` +
      `MSIX versions cap at 16 bits (45 days); cut a stable first`
    )
  }
  return minutes
}

/**
 * Minutes-since-stable for a canary tag, resolving the stable base from
 * the repo's tags on the same major.minor line.
 * @param {string} tag the release tag
 * @param {string} gitRoot the repo root
 * @returns {number | null}
 */
export function canaryBuildMinutes(tag, gitRoot) {
  const m = CANARY_TAG_RE.exec(String(tag || ''))
  if (!m) return null
  const majorMinor = m[1].split('.').slice(0, 2).join('.')
  const stable = listGitTags(gitRoot, `v${majorMinor}.*`).find(t => STABLE_TAG_RE.test(t))
  if (!stable) return 0 // degenerate: no stable on this line; build number restarts
  return canaryBuildMinutesFor(tag, gitTagCommitTime(gitRoot, stable))
}

/** Store reserves revision for itself. Keep its package sequence separate
 * from the app's displayed semver and the sideload update sequence.
 * Calendar fields retain second precision without an epoch offset.
 * @param {number} epochSeconds immutable release time in UTC
 * @returns {string}
 */
export function storePackageVersionAt(epochSeconds) {
  const date = new Date(epochSeconds * 1000)
  const year = date.getUTCFullYear()
  if (!Number.isInteger(epochSeconds) || !Number.isFinite(date.getTime()) || year < 1000 || year > 65535) {
    throw new Error('Store package version needs a valid immutable release timestamp')
  }
  const hourOfYear = Math.floor((date.getTime() - Date.UTC(year, 0, 1)) / 3_600_000)
  const secondOfHour = date.getUTCMinutes() * 60 + date.getUTCSeconds()
  return `${year}.${hourOfYear}.${secondOfHour}.0`
}

/** @param {string} tag @param {string} gitRoot */
export function storePackageVersion(tag, gitRoot) {
  const canary = CANARY_TAG_RE.exec(tag)
  if (canary) {
    const epoch = stampToEpoch(canary[2])
    const roundtrip = new Date(epoch * 1000).toISOString().replace(/[-:T]/g, '').slice(0, canary[2].length)
    if (roundtrip !== canary[2]) throw new Error('Invalid canary calendar timestamp')
    return storePackageVersionAt(epoch)
  }
  if (!STABLE_TAG_RE.test(tag)) throw new Error('A Store build requires an exact release tag')
  const timestamp = execFileSync('git', ['for-each-ref', '--format=%(creatordate:unix)', `refs/tags/${tag}`], {
    cwd: gitRoot, encoding: 'utf8'
  }).trim()
  if (!/^\d+$/.test(timestamp)) throw new Error(`No immutable release timestamp for ${tag}`)
  return storePackageVersionAt(Number(timestamp))
}

/** The custom template controls package identity, not executable VERSIONINFO.
 * @param {string} template @param {string} version
 */
export function storeManifestTemplate(template, version) {
  if (!/^[1-9]\d*\.\d+\.\d+\.0$/.test(version) || version.split('.').some(part => Number(part) > 65535)) {
    throw new Error('Store package version must have a nonzero major, 16-bit fields and zero revision')
  }
  if (template.split('${version}').length !== 2) throw new Error('MSIX template must have one version macro')
  return template.replace('${version}', version)
}

/**
 * Resolve the app identity for a desktop build from the app dir: the product
 * identity + package version. Pure-ish (reads product-identity.cjs and
 * package.json from the app dir) so callers on any runner can derive the
 * exact feed filename/identity without duplicating the derivation.
 *
 * @param {string} desktopDir absolute apps/desktop path
 * @param {string} [tag] the release tag (defaults to HERMES_PAYLOAD_TAG)
 * @returns {{ identity: object, version: string, name: string, fileVersion: string }}
 */
export function appIdentity(desktopDir, tag = process.env.HERMES_PAYLOAD_TAG || '') {
  const identity = require(path.join(desktopDir, 'product-identity.cjs'))
  const pkg = JSON.parse(fs.readFileSync(path.join(desktopDir, 'package.json'), 'utf8'))
  const repoRoot = path.resolve(desktopDir, '..', '..')
  // Commit artifacts retain app semver but do not advance an update channel.
  if (process.env.HERMES_BUILD_COMMIT) {
    if (tag) throw new Error('Commit-only builds must not set HERMES_PAYLOAD_TAG')
    const commit = process.env.HERMES_BUILD_COMMIT
    if (!/^[a-f0-9]{40}$/.test(commit)) throw new Error('Commit builds require an exact full SHA')
    const version = String(process.env.HERMES_PAYLOAD_VERSION || '')
    if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version)
        || version.split('.').some(part => Number(part) > 65535)) {
      throw new Error('Commit builds require HERMES_PAYLOAD_VERSION=X.Y.Z with 16-bit fields')
    }
    if (identity.store) throw new Error('Store packaging requires a stable release tag')
    return { identity, version: `${version}.0`, fileVersion: version, name: identity.artifactNamePascal }
  }
  if (identity.store) {
    return { identity, version: storePackageVersion(String(tag), repoRoot),
      fileVersion: String(tag).slice(1), name: identity.artifactNamePascal }
  }
  const canary = CANARY_TAG_RE.exec(String(tag))
  if (canary) {
    // Manifest + feed version: tag base (0.27.2) + minutes-since-stable.
    // The artifact FILENAME carries electron-builder's appInfo.version — the
    // full canary string (HermesBundled-0.27.2-canary.X-win-x64.msix) — so
    // callers that look files up by name need that string separately.
    return {
      identity,
      version: `${canary[1]}.${canaryBuildMinutes(String(tag), repoRoot)}`,
      fileVersion: String(tag).slice(1),
      name: identity.artifactNamePascal,
    }
  }
  if (tag && !STABLE_TAG_RE.test(tag)) throw new Error(`Invalid release tag: ${tag}`)
  // Release builds override Electron's version without rewriting package.json.
  const version = tag ? tag.slice(1) : pkg.version
  return {
    identity,
    version: `${version}.0`,
    fileVersion: version,
    name: identity.artifactNamePascal,
  }
}
