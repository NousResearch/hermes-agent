#!/usr/bin/env node
// Pin real release packages before a destructive update leg starts.
import { createHash } from 'node:crypto'
import { createWriteStream } from 'node:fs'
import { mkdir, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { Transform, Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { parseArgs } from 'node:util'
import { pathToFileURL } from 'node:url'
import semver from 'semver'

const TAG = /^v\d+\.\d+\.\d+(?:-canary\.\d{14})?$/
const COMMIT = /^[0-9a-f]{40}$/
const SHA256 = /^[0-9a-f]{64}$/
const FORMATS = { windows: '.msixbundle', macos: '.zip' }

function artifactUrl(value) {
  const url = new URL(value)
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
  if (url.username || url.password || !(url.protocol === 'https:' || (url.protocol === 'http:' && local))) {
    throw new Error('Bundle URLs must use HTTPS or loopback HTTP, without credentials')
  }
  return url
}

function windowsVersion(version) {
  if (typeof version !== 'string' || !/^\d+\.\d+\.\d+\.\d+$/.test(version)) {
    throw new Error('Windows package version must have four numeric components')
  }
  const parts = version.split('.').map(Number)
  if (parts.some(n => n > 65535)) throw new Error('Windows package version exceeds 16 bits')
  return parts
}

export function validateBundleInputs(value, platform, arch) {
  if (!Object.hasOwn(FORMATS, platform) || !['x64', 'arm64'].includes(arch)) {
    throw new Error('Unsupported bundled-update platform or architecture')
  }
  if (value?.schema !== 1 || value.platform !== platform || value.arch !== arch) {
    throw new Error('Bundle manifest schema, platform or architecture mismatch')
  }
  for (const slot of ['old', 'new']) {
    const item = value[slot]
    if (!item || !TAG.test(item.tag) || !COMMIT.test(item.commit) || !item.identity) {
      throw new Error(`${slot}: exact release tag, full commit and package identity are required`)
    }
    if (!item.artifact || !SHA256.test(item.artifact.sha256)) throw new Error(`${slot}: SHA-256 is required`)
    const url = artifactUrl(item.artifact.url)
    if (!url.pathname.endsWith(FORMATS[platform])) throw new Error(`${slot}: expected ${FORMATS[platform]} artifact`)
    if (platform === 'windows') {
      windowsVersion(item.version)
      if (!item.publisher || !item.applicationId) throw new Error(`${slot}: publisher and applicationId are required`)
    } else {
      if (!semver.valid(item.version) || item.version !== item.tag.slice(1)) throw new Error(`${slot}: macOS version must match its release tag`)
      if (!/^[A-Z0-9]{10}$/.test(item.teamId)) throw new Error(`${slot}: macOS signing teamId is required`)
    }
  }
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
    if (value.old.tag.includes('-canary.') !== value.new.tag.includes('-canary.')) throw new Error('Bundle transition must stay on one update channel')
  }
  return value
}

export async function downloadArtifact(artifact, destination) {
  const url = artifactUrl(artifact.url)
  const response = await fetch(url, { signal: AbortSignal.timeout(30 * 60 * 1000) })
  if (!response.ok || !response.body) throw new Error(`Package download returned HTTP ${response.status}`)
  const digest = createHash('sha256')
  const temporary = `${destination}.partial`
  try {
    await pipeline(Readable.fromWeb(response.body), new Transform({
      transform(chunk, encoding, done) { digest.update(chunk); done(null, chunk) }
    }), createWriteStream(temporary, { flags: 'wx' }))
    if (digest.digest('hex') !== artifact.sha256) throw new Error('Package SHA-256 mismatch')
    await rename(temporary, destination)
  } catch (error) {
    await rm(temporary, { force: true })
    throw error
  }
}

export async function stageBundleInputs({ manifestUrl, platform, arch, out, expectedCommit, manifestSha256 }) {
  const response = await fetch(artifactUrl(manifestUrl), { signal: AbortSignal.timeout(60_000) })
  if (!response.ok) throw new Error(`Manifest download returned HTTP ${response.status}`)
  const text = await response.text()
  if (text.length > 1024 * 1024) throw new Error('Bundle input manifest is too large')
  if (manifestSha256 && (!SHA256.test(manifestSha256) || createHash('sha256').update(text).digest('hex') !== manifestSha256)) {
    throw new Error('Transition manifest SHA-256 mismatch')
  }
  const manifest = validateBundleInputs(JSON.parse(text), platform, arch)
  if (expectedCommit && manifest.new.commit !== expectedCommit) {
    throw new Error('Candidate commit must equal the tested workflow SHA')
  }
  await mkdir(out, { recursive: true })
  const staged = { ...manifest }
  for (const slot of ['old', 'new']) {
    const destination = path.resolve(out, `${slot}${FORMATS[platform]}`)
    await downloadArtifact(manifest[slot].artifact, destination)
    staged[slot] = { ...manifest[slot], artifact: { ...manifest[slot].artifact, path: destination } }
  }
  const filename = path.resolve(out, 'bundle-inputs.json')
  await writeFile(filename, JSON.stringify(staged, null, 2) + '\n', { encoding: 'utf8', flag: 'wx' })
  return filename
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: {
    'manifest-url': { type: 'string' }, platform: { type: 'string' },
    arch: { type: 'string' }, out: { type: 'string' }
  } })
  if (!values['manifest-url'] || !values.platform || !values.arch || !values.out) {
    throw new Error('--manifest-url, --platform, --arch and --out are required')
  }
  console.log(await stageBundleInputs({ manifestUrl: values['manifest-url'], platform: values.platform, arch: values.arch, out: values.out,
    expectedCommit: process.env.GITHUB_ACTIONS === 'true' ? process.env.GITHUB_SHA : undefined,
    manifestSha256: process.env.BUNDLE_MANIFEST_SHA256 || undefined }))
}
