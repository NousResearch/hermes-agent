// Shared admission for remote inputs and resumed native acceptance phases.
const fs = require('node:fs')
const semver = require('semver')

const TAG = /^v\d+\.\d+\.\d+(?:\+canary\.20\d{6}T\d{6}Z)?$/
const COMMIT = /^[0-9a-f]{40}$/
const SHA256 = /^[0-9a-f]{64}$/
const FORMATS = { windows: '.msixbundle', macos: '.zip' }

function artifactUrl(value) {
  // SAFETY: this is the URL input boundary; reject JSON values before URL coercion.
  // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
  if (typeof value !== 'string' || !value) throw new Error('artifact.url must be a non-empty string')
  const url = new URL(value)
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
  if (url.username || url.password || !(url.protocol === 'https:' || (url.protocol === 'http:' && local))) {
    throw new Error('Bundle URLs must use HTTPS or loopback HTTP, without credentials')
  }
  return url
}

function windowsVersion(version) {
  // SAFETY: version is untrusted manifest data, not an already typed domain value.
  // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
  if (typeof version !== 'string' || !/^\d+\.\d+\.\d+\.\d+$/.test(version)) {
    throw new Error('Windows package version must have four numeric components')
  }
  const parts = version.split('.').map(Number)
  if (parts.some(n => n > 65535)) throw new Error('Windows package version exceeds 16 bits')
  return parts
}

function validateSide(item, slot, platform) {
  // SAFETY: validate JSON fields before regexes can coerce objects or numbers.
  // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
  if (!item || ['tag', 'version', 'commit', 'identity'].some(key => typeof item[key] !== 'string' || !item[key]) ||
      !TAG.test(item.tag) || !COMMIT.test(item.commit)) {
    throw new Error(`${slot}: exact release tag, full commit and package identity are required`)
  }
  // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode the SHA-256 JSON string before regex coercion.
  if (!item.artifact || typeof item.artifact.sha256 !== 'string' || !SHA256.test(item.artifact.sha256)) throw new Error(`${slot}: SHA-256 is required`)
  const url = artifactUrl(item.artifact.url)
  if (!url.pathname.endsWith(FORMATS[platform])) throw new Error(`${slot}: expected ${FORMATS[platform]} artifact`)
  if (platform === 'windows') {
    windowsVersion(item.version)
    // SAFETY: publisher and applicationId are required string fields at admission.
    // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
    if (['publisher', 'applicationId'].some(key => typeof item[key] !== 'string' || !item[key])) throw new Error(`${slot}: publisher and applicationId are required`)
  } else {
    if (!/^[A-Za-z][A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$/.test(item.identity)) throw new Error(`${slot}: identity must be a CFBundleIdentifier`)
    if (!semver.valid(item.version) || item.version !== item.tag.slice(1)) throw new Error(`${slot}: macOS version must match its release tag`)
    // SAFETY: teamId must be a JSON string, not an array coerced by RegExp.test.
    // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
    if (typeof item.teamId !== 'string' || !/^[A-Z0-9]{10}$/.test(item.teamId)) throw new Error(`${slot}: macOS signing teamId is required`)
  }
}

function validateBundleInputs(value, platform, arch) {
  if (!Object.hasOwn(FORMATS, platform) || !['x64', 'arm64'].includes(arch)) {
    throw new Error('Unsupported bundled-update platform or architecture')
  }
  if (value?.schema !== 1 || value.platform !== platform || value.arch !== arch) {
    throw new Error('Bundle manifest schema, platform or architecture mismatch')
  }
  for (const slot of ['old', 'new']) validateSide(value[slot], slot, platform)
  if (value.old.identity !== value.new.identity || value.old.commit === value.new.commit) {
    throw new Error('Update must preserve package identity and change the build commit')
  }
  if (value.old.artifact.sha256 === value.new.artifact.sha256) throw new Error('Update artifacts must differ')
  if (platform === 'windows') {
    if (value.old.publisher !== value.new.publisher || value.old.applicationId !== value.new.applicationId) {
      throw new Error('Update must preserve publisher and applicationId')
    }
    const old = windowsVersion(value.old.version)
    const newer = windowsVersion(value.new.version)
    const first = newer.findIndex((n, i) => n !== old[i])
    if (first < 0 || newer[first] <= old[first]) throw new Error('New package version must increase')
  } else {
    if (value.new.teamId !== value.old.teamId) throw new Error('Update must preserve signing team')
    if (!semver.gt(value.new.version, value.old.version)) throw new Error('New package version must increase')
    if (value.old.tag.includes('+canary.') !== value.new.tag.includes('+canary.')) throw new Error('Bundle transition must stay on one update channel')
  }
  return value
}

// Revalidate normalized JSON on every phase; staging is not a durable trust boundary.
function validateDownloadedBundle(value, platform, arch) {
  validateBundleInputs(value, platform, arch)
  for (const slot of ['old', 'new']) {
    const filename = value[slot].artifact.path
    // SAFETY: normalized JSON is re-read between phases; validate before filesystem access.
    // eslint-disable-next-line anti-slop/no-runtime-typeof -- Decode untrusted manifest JSON at its input boundary.
    if (typeof filename !== 'string' || !filename || !fs.statSync(filename, { throwIfNoEntry: false })?.isFile()) {
      throw new Error(`${slot}.artifact.path must name a downloaded file`)
    }
  }
  return value
}

module.exports = { FORMATS, SHA256, artifactUrl, validateBundleInputs, validateDownloadedBundle }
