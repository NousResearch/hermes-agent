#!/usr/bin/env node
// End-to-end verification of the IX Agency auto-update feed on S3.
//
// Checks, for every electron-updater channel file (mac / windows / linux):
//   1. the channel file is publicly fetchable (HTTP 200),
//   2. every artifact it references is publicly fetchable (HTTP 200) and the
//      Content-Length matches the size recorded in the channel file,
//   3. all three channels agree on the same version,
//   4. (optional) that version equals EXPECT_VERSION (set by the release CI
//      to the freshly published package.json version).
//
// Usage:  node scripts/verify-update-feed.mjs
//         EXPECT_VERSION=0.17.1 node scripts/verify-update-feed.mjs
//
// Exits non-zero on any failure — run by .github/workflows on every merge
// and at the end of every desktop release.

const FEED = (process.env.UPDATE_FEED_URL || 'https://intelliverse-x-desktop.s3.amazonaws.com/ix-agency').replace(/\/+$/, '')
const EXPECT_VERSION = (process.env.EXPECT_VERSION || '').trim()

const CHANNELS = [
  { os: 'mac', file: 'latest-mac.yml' },
  { os: 'windows', file: 'latest.yml' },
  { os: 'linux', file: 'latest-linux.yml' }
]

let failures = 0
const fail = (msg) => {
  failures++
  console.error(`  FAIL  ${msg}`)
}
const ok = (msg) => console.log(`  ok    ${msg}`)

/** Minimal parse of electron-builder channel yml: version + files[{url,size}]. */
function parseChannel(yml) {
  const version = (yml.match(/^version:\s*(\S+)/m) || [])[1] || ''
  const files = []
  // entries look like:  "  - url: NAME"  followed by "    size: N"
  const re = /-\s*url:\s*(\S+)[\s\S]*?size:\s*(\d+)/g
  for (let m; (m = re.exec(yml)); ) files.push({ url: m[1], size: Number(m[2]) })
  return { version, files }
}

async function head(url) {
  const r = await fetch(url, { method: 'HEAD' })
  return { status: r.status, length: Number(r.headers.get('content-length') || 0) }
}

const versions = new Set()
for (const { os, file } of CHANNELS) {
  const url = `${FEED}/${file}`
  console.log(`── ${os}: ${url}`)
  let body
  try {
    const r = await fetch(url)
    if (r.status !== 200) {
      fail(`${file} -> HTTP ${r.status}`)
      continue
    }
    body = await r.text()
  } catch (e) {
    fail(`${file} -> ${e.message}`)
    continue
  }
  const { version, files } = parseChannel(body)
  if (!version) {
    fail(`${file} has no version field`)
    continue
  }
  versions.add(version)
  ok(`${file} v${version} (${files.length} artifacts)`)
  if (!files.length) fail(`${file} lists no artifacts`)
  for (const f of files) {
    const artifactUrl = `${FEED}/${encodeURIComponent(f.url).replace(/%2F/g, '/')}`
    try {
      const h = await head(artifactUrl)
      if (h.status !== 200) fail(`${f.url} -> HTTP ${h.status}`)
      else if (f.size && h.length !== f.size) fail(`${f.url} size mismatch: feed=${f.size} s3=${h.length}`)
      else ok(`${f.url} (${(f.size / 1e6).toFixed(1)} MB)`)
    } catch (e) {
      fail(`${f.url} -> ${e.message}`)
    }
  }
}

if (versions.size > 1) fail(`channels disagree on version: ${[...versions].join(', ')}`)
if (EXPECT_VERSION && !versions.has(EXPECT_VERSION)) {
  fail(`expected v${EXPECT_VERSION} on the feed, found: ${[...versions].join(', ') || 'none'}`)
}

if (failures) {
  console.error(`\nupdate-feed verification FAILED (${failures} problem${failures > 1 ? 's' : ''})`)
  process.exit(1)
}
console.log(`\nupdate-feed verification passed — v${[...versions][0]} live on all 3 platforms`)
