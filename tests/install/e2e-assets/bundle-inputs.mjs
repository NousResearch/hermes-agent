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
import { FORMATS, SHA256, artifactUrl, validateBundleInputs } from './bundle-manifest.cjs'

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
