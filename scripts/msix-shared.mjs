// Native MSIX identity/version derivation and artifact content types.
// App Installer XML and feed publication belong to scripts.bundles.release_artifacts.

import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)

/** Build-only structured bridge; never read by the installed runtime.
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {import('../apps/desktop/electron/install-stamp.js').ChannelBuildRequest | null}
 */
export function channelBuildRequest(env = process.env) {
  if (!env._HERMES_CHANNEL_REQUEST_JSON) return null
  const value = JSON.parse(env._HERMES_CHANNEL_REQUEST_JSON)
  if (env.HERMES_DESKTOP_VARIANT !== 'bundled') throw new Error('Channel builds support only bundled packaging')
  if (env.HERMES_BUILD_COMMIT || env.HERMES_PAYLOAD_TAG) throw new Error('Channel request conflicts with commit or tag identity')
  // The bundled toolchain already supplies Python. Reuse the authoritative
  // validator rather than maintaining a third protocol decoder for packaging.
  const validator = [
    'import sys',
    'sys.path.insert(0, sys.argv[1])',
    'from hermes_cli.release_channels import decode_json',
    'from scripts.bundles.desktop_prepare import validate_channel_request',
    'validate_channel_request(decode_json(sys.stdin.buffer.read()))'
  ].join('; ')
  execFileSync(env.HERMES_PYTHON || 'python', ['-I', '-S', '-c', validator, path.resolve(import.meta.dirname, '..')], {
    env, input: env._HERMES_CHANNEL_REQUEST_JSON, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 30_000
  })
  if (env.HERMES_PAYLOAD_VERSION && env.HERMES_PAYLOAD_VERSION !== value.version) throw new Error('Channel package version conflicts with prepared request')
  Object.freeze(value.identity)
  Object.freeze(value.bundleEnv)
  return Object.freeze(value)
}


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

const VERSION_CORE = '(?:0|[1-9]\\d{0,2})\\.(?:0|[1-9]\\d*)\\.(?:0|[1-9]\\d*)'
const CANARY_TAG_RE = new RegExp(`^v(${VERSION_CORE})\\+canary\\.(20\\d{6}T\\d{6}Z)$`)
const STABLE_TAG_RE = new RegExp(`^v${VERSION_CORE}$`)

/**
 * @param {string} stamp YYYYMMDDTHHMMSSZ
 * @returns {number} epoch seconds
 */
function stampToEpoch(stamp) {
  const compact = stamp.replace('T', '').replace(/Z$/, '')
  const parts = /^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?$/.exec(compact)
  if (!parts) return 0
  const [, y, mo, d, h, mi, s] = parts
  return Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h ?? 0), Number(mi ?? 0), Number(s ?? 0)) / 1000
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

/** Canary package quad: ``yy.mmdd.hh.mmss`` in UTC.
 * @param {number} epochSeconds immutable build time in UTC
 * @returns {string}
 */
export function canaryPackageVersionAt(epochSeconds) {
  const date = new Date(epochSeconds * 1000)
  if (!Number.isInteger(epochSeconds) || !Number.isFinite(date.getTime())) {
    throw new Error('Canary package version needs a valid immutable build timestamp')
  }
  const yy = date.getUTCFullYear() % 100
  const mmdd = (date.getUTCMonth() + 1) * 100 + date.getUTCDate()
  const hh = date.getUTCHours()
  const mmss = date.getUTCMinutes() * 100 + date.getUTCSeconds()
  return `${yy}.${mmdd}.${hh}.${mmss}`
}

/** The native quad for a release.

Stable Store packages retain ``year.hourOfYear.secondOfHour.0``. Canary uses
``yy.mmdd.hh.mmss`` so its four numeric fields show the UTC build time directly
while remaining monotonic across month boundaries.
@param {string} ref the release ref whose native package is being stamped
@param {number} epochSeconds the build time, UTC
@returns {string}
*/
export function nativeQuad(ref, epochSeconds) {
  if (CANARY_TAG_RE.test(ref)) return canaryPackageVersionAt(epochSeconds)
  if (STABLE_TAG_RE.test(ref)) return storePackageVersionAt(epochSeconds)
  throw new Error('Native package version requires a current release tag')
}

/** The build time of a release tag: the stamp embedded in a canary tag, or
 * the tag's own creation time for a stable tag.
 * @param {string} tag @param {string} gitRoot @returns {number} epoch seconds
 */
function releaseEpoch(tag, gitRoot) {
  const canary = CANARY_TAG_RE.exec(tag)
  if (canary) {
    const stamp = canary[2]
    const epoch = stampToEpoch(stamp)
    const compact = new Date(epoch * 1000).toISOString().replace(/[-:T]/g, '').replace(/\.000Z$/, '')
    const expected = `${compact.slice(0, 8)}T${compact.slice(8)}Z`
    if (expected !== stamp) throw new Error('Invalid canary calendar timestamp')
    const supplied = process.env.HERMES_RELEASE_EPOCH
    if (supplied !== undefined && Number(supplied) !== epoch) throw new Error('Canary release epoch differs from its tag')
    return epoch
  }
  const supplied = process.env.HERMES_RELEASE_EPOCH
  if (supplied !== undefined) {
    if (!/^\d+$/.test(supplied)) throw new Error('Invalid immutable release epoch')
    return Number(supplied)
  }
  const claim = process.env.RELEASE_CLAIM_TAG
  if (claim && claim !== `${tag}-rc`) throw new Error('Stable claim tag differs from its payload tag')
  const claimObject = process.env.RELEASE_CLAIM_OBJECT
  let timestamp
  let timestampTag = tag
  if (claim) {
    timestampTag = claim
    if (!/^[a-f0-9]{40}$/.test(claimObject || '')) throw new Error('Stable claim object is missing')
    const actual = execFileSync('git', ['rev-parse', `refs/tags/${claim}`], { cwd: gitRoot, encoding: 'utf8' }).trim()
    if (actual !== claimObject) throw new Error('Stable claim object differs from its tag')
    const metadata = execFileSync('git', ['cat-file', '-p', claimObject], { cwd: gitRoot, encoding: 'utf8' })
    timestamp = /^tagger .* (\d+) [+-]\d{4}$/m.exec(metadata)?.[1]
  } else {
    timestamp = execFileSync('git', ['for-each-ref', '--format=%(creatordate:unix)', `refs/tags/${tag}`], {
      cwd: gitRoot, encoding: 'utf8'
    }).trim()
  }
  if (!/^\d+$/.test(timestamp || '')) throw new Error(`No immutable release timestamp for ${timestampTag}`)
  return Number(timestamp)
}

/** @param {string} tag @param {string} gitRoot */
export function storePackageVersion(tag, gitRoot) {
  if (!CANARY_TAG_RE.test(tag) && !STABLE_TAG_RE.test(tag)) throw new Error('A Store build requires an exact release tag')
  return nativeQuad(tag, releaseEpoch(tag, gitRoot))
}

/** The custom template controls package identity, not executable VERSIONINFO.
 * @param {string} template @param {string} version
 */
export function storeManifestTemplate(template, version) {
  if (!/^[1-9]\d*\.\d+\.\d+\.0$/.test(version) || version.split('.').some(part => Number(part) > 65535)) {
    throw new Error('Store package version must have a nonzero major, 16-bit fields and zero revision')
  }
  return nativeManifestTemplate(template, version)
}

/** @param {string} template @param {string} version */
export function nativeManifestTemplate(template, version) {
  if (!/^[1-9]\d*\.\d+\.\d+\.\d+$/.test(version) || version.split('.').some(part => Number(part) > 65535)) {
    throw new Error('Native package version must contain four 16-bit numeric fields')
  }
  if (template.split('${version}').length !== 2) throw new Error('MSIX template must contain exactly one version macro')
  return template.replace('${version}', version)
}

/** MsixTarget derives its quad from app semver even when shortVersion is set.
 * Bake the admitted quad into the already staged nonstable template instead.
 * @param {string} desktopDir
 * @param {import('../apps/desktop/electron/install-stamp.js').ChannelBuildRequest} request
 */
export function stageChannelManifest(desktopDir, request) {
  const file = path.join(desktopDir, 'build/msix-manifest.xml')
  const template = fs.readFileSync(path.join(desktopDir, 'assets/msix-manifest.xml'), 'utf8')
  const payload = JSON.parse(fs.readFileSync(path.join(desktopDir, 'build/agent-payload/manifest.json'), 'utf8'))
  const { appExecutionAliasApplications } = require(path.join(desktopDir, 'scripts/before-build.mjs'))
  const applications = appExecutionAliasApplications(payload.launchers, request.identity)
  if (template.split('${version}').length !== 2) throw new Error('Channel MSIX template must have one version macro')
  fs.writeFileSync(file, template.replace('${version}', request.windowsVersion)
    .replace('</Applications>', `${applications}\n  </Applications>`), 'utf8')
  // The legacy hook emits one multi-alias extension when artifact/app names agree.
  // Channels always use separate applications for distinct CLI entrypoints.
  const extensions = path.join(desktopDir, 'build/msix-extensions.xml')
  const xml = fs.readFileSync(extensions, 'utf8')
  fs.writeFileSync(extensions, xml.replace(/<uap5:Extension\b[^>]*Category="windows\.appExecutionAlias"[^>]*>[\s\S]*?<\/uap5:Extension>/g, ''), 'utf8')
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
  const request = channelBuildRequest()
  if (request) {
    if (tag) throw new Error('Channel builds must not select a release tag')
    return { identity, version: request.windowsVersion, fileVersion: request.version, name: identity.artifactNamePascal }
  }
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
    return { identity, version: nativeQuad(String(tag), releaseEpoch(String(tag), repoRoot)),
      fileVersion: String(tag).slice(1), name: identity.artifactNamePascal }
  }
  if (CANARY_TAG_RE.test(String(tag)) || (tag && STABLE_TAG_RE.test(tag))) {
    return {
      identity,
      version: nativeQuad(String(tag), releaseEpoch(String(tag), repoRoot)),
      fileVersion: String(tag).slice(1),
      name: identity.artifactNamePascal,
    }
  }
  if (tag) throw new Error(`Invalid release tag: ${tag}`)
  // Release builds override Electron's version without rewriting package.json.
  const version = tag ? tag.slice(1) : pkg.version
  return {
    identity,
    version: `${version}.0`,
    fileVersion: version,
    name: identity.artifactNamePascal,
  }
}
