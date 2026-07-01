#!/usr/bin/env node
// assert-icon.cjs — LOCK the app icon so it can't silently drift between builds.
//
// Repeated past updates kept changing the icon's on-screen size/padding. The
// committed icon is macOS-grid-correct: 1024x1024 canvas, art ~820px (≈80% fill,
// ~10% transparent margin per side) — matching Apple's app-icon standard so it
// renders the SAME size as other Dock apps. This guard fails the build if any
// icon asset changes, forcing an intentional (reviewed) update.
//
// To intentionally change the icon: replace the asset(s), keep 1024x1024 + the
// ~80% art fill (10% margin), then update the sha256 below
// (run: `shasum -a 256 assets/icon.png assets/icon.icns assets/icon.ico`).
const crypto = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')

const ASSETS = path.join(__dirname, '..', 'assets')
const EXPECTED = {
  'icon.png': 'fb28595680bbae35b9357ebb40d4866d4e5353b31269a1e8684d66e505193817',
  'icon.icns': '2d9f13eb9be85e243c7268fc5c83bbe66e6b26dc6e8df277c1c596ca3b724155',
  'icon.ico': 'ae2299f9b34252c7dc3834d142319e745e1624f8bdf0b6f51d4fcd0702050eef'
}

let failed = false

for (const [name, expected] of Object.entries(EXPECTED)) {
  let buf
  try {
    buf = fs.readFileSync(path.join(ASSETS, name))
  } catch {
    console.error(`[assert-icon] missing icon asset: assets/${name}`)
    failed = true
    continue
  }
  const got = crypto.createHash('sha256').update(buf).digest('hex')
  if (got !== expected) {
    console.error(
      `[assert-icon] assets/${name} changed (sha256 ${got.slice(0, 12)}… ≠ pinned ${expected.slice(0, 12)}…).\n` +
        '  The app icon is LOCKED to stop size/padding drift between updates.\n' +
        '  If intentional: keep 1024x1024 + ~80% art fill (macOS grid), then update\n' +
        '  the sha256 in scripts/assert-icon.cjs (shasum -a 256 assets/icon.*).'
    )
    failed = true
  }
}

// Belt-and-suspenders: PNG must be exactly 1024x1024 (IHDR width/height at 16..24).
try {
  const png = fs.readFileSync(path.join(ASSETS, 'icon.png'))
  const w = png.readUInt32BE(16)
  const h = png.readUInt32BE(20)
  if (w !== 1024 || h !== 1024) {
    console.error(`[assert-icon] assets/icon.png must be 1024x1024, got ${w}x${h}`)
    failed = true
  }
} catch {
  // readFileSync failure already reported above.
}

if (failed) {
  process.exit(1)
}
console.log('[assert-icon] icon locked + valid (1024x1024, pinned sha256, ~80% macOS-grid fill)')
