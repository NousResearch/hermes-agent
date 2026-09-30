#!/usr/bin/env node
/**
 * verify-release-gate.mjs — the last check before a build becomes a release.
 *
 * Everything upstream of this script is best-effort by design: signing skips
 * when no certificate is configured, a platform leg can be skipped, a build
 * can emit fewer targets than expected. That flexibility is what makes PR
 * builds and local `npm run dist` usable — and it is exactly what would let a
 * half-built, unsigned release reach users if nothing checked at the end.
 *
 * This is that check. It reads release-manifest.json and refuses to pass when:
 *
 *   • the manifest version disagrees with the tag being published;
 *   • a required {platform, kind} artifact is missing (e.g. no arm64 dmg);
 *   • a signable artifact (Windows installer, macOS disk image) is unsigned;
 *   • a recorded sha256 does not match the file on disk;
 *   • an installer is suspiciously small (a truncated upload still has a
 *     valid checksum of its truncated self) — updater sidecars are exempt,
 *     being small by design.
 *
 * Usage:
 *   node scripts/verify-release-gate.mjs --dir release --version 0.17.2
 *   node scripts/verify-release-gate.mjs --dir release --tag v0.17.2 --platforms win,mac,linux
 *   node scripts/verify-release-gate.mjs --dir release --allow-unsigned   # dry runs only
 */

import fs from 'node:fs'
import path from 'node:path'

import { isMain } from './utils.mjs'
import { MANIFEST_FILENAME, sha256File, SIZE_FLOOR_EXEMPT_KINDS } from './release-manifest.mjs'

/**
 * The support matrix the product promises (design doc §12: "pakiety Linux dla
 * jawnie wspieranych dystrybucji" / Windows + macOS signed installers).
 *
 * `archs: null` means "any arch is acceptable"; a list means every entry must
 * be present. macOS lists both slices explicitly because an Intel-only dmg is
 * a release that silently excludes every Mac sold since 2020.
 */
export const REQUIRED_ARTIFACTS = {
  win: [
    { kind: 'nsis', archs: null },
    { kind: 'msi', archs: null }
  ],
  mac: [
    { kind: 'dmg', archs: ['arm64', 'x64'] },
    { kind: 'zip', archs: ['arm64', 'x64'] }
  ],
  linux: [
    { kind: 'appimage', archs: null },
    { kind: 'deb', archs: null },
    { kind: 'rpm', archs: null }
  ]
}

/**
 * Anything under this is a build that failed without saying so — an installer
 * is tens of megabytes. Checksums cannot catch truncation, because the digest
 * of a truncated file is a perfectly valid digest.
 */
export const MIN_ARTIFACT_BYTES = 1024 * 1024

/** `v0.17.2` and `0.17.2` are the same release; four-part CalVer tags too. */
export function normalizeVersion(value) {
  return String(value || '')
    .trim()
    .replace(/^v/i, '')
}

/**
 * Evaluate the gate.
 *
 * Pure over the manifest plus an optional `statSize`/`digest` pair, so the
 * whole decision table is testable without building installers. Returns every
 * problem found rather than the first: a release engineer should see the
 * complete list in one CI run, not peel them off one at a time.
 */
export function evaluateGate({
  manifest,
  version = '',
  platforms = ['win', 'mac', 'linux'],
  archs = null,
  allowUnsigned = false,
  verifyDigests = false,
  statSize = null,
  digest = null,
  minBytes = MIN_ARTIFACT_BYTES
}) {
  const problems = []

  if (!manifest || !Array.isArray(manifest.artifacts)) {
    return { ok: false, problems: ['manifest is missing or has no artifacts array'] }
  }

  const expected = normalizeVersion(version)
  const actual = normalizeVersion(manifest.version)
  if (expected && actual !== expected) {
    problems.push(`version mismatch: manifest says ${actual || '(none)'}, release expects ${expected}`)
  }

  for (const platform of platforms) {
    const required = REQUIRED_ARTIFACTS[platform]
    if (!required) {
      problems.push(`unknown platform "${platform}"`)
      continue
    }

    const present = manifest.artifacts.filter(artifact => artifact.platform === platform)
    if (present.length === 0) {
      problems.push(`no ${platform} artifacts in the manifest`)
      continue
    }

    for (const requirement of required) {
      const matches = present.filter(artifact => artifact.kind === requirement.kind)
      if (matches.length === 0) {
        problems.push(`missing ${platform} ${requirement.kind} artifact`)
        continue
      }

      // `archs` narrows the requirement for a per-architecture build leg; the
      // aggregate gate omits it and so still demands every slice.
      for (const arch of (requirement.archs ?? []).filter(entry => !archs || archs.includes(entry))) {
        // A universal binary satisfies every arch it fuses.
        const covered = matches.some(
          artifact => artifact.arch === arch || artifact.arch === 'universal'
        )
        if (!covered) {
          problems.push(`missing ${platform} ${requirement.kind} for ${arch}`)
        }
      }
    }
  }

  for (const artifact of manifest.artifacts) {
    if (!platforms.includes(artifact.platform)) {
      continue
    }

    if (artifact.signable && !artifact.signed && !allowUnsigned) {
      problems.push(`${artifact.file} is unsigned (${artifact.platform} artifacts must be signed)`)
    }

    const size = statSize ? statSize(artifact.file) : artifact.size
    // The floor is for truncated payloads. Updater sidecars (a kilobyte-scale
    // `latest-*.yml`, a `.blockmap`) are small by design, and failing a release
    // over them is how a complete build never ships.
    if (typeof size === 'number' && size < minBytes && !SIZE_FLOOR_EXEMPT_KINDS.has(artifact.kind)) {
      problems.push(`${artifact.file} is only ${size} bytes — build or upload was truncated`)
    }

    if (verifyDigests && digest) {
      const onDisk = digest(artifact.file)
      if (onDisk !== artifact.sha256) {
        problems.push(`${artifact.file} checksum mismatch: manifest ${artifact.sha256}, on disk ${onDisk}`)
      }
    }
  }

  return { ok: problems.length === 0, problems }
}

export function renderReport(result, manifest) {
  const lines = []
  lines.push(`Release gate — ${manifest?.product ?? 'unknown product'} v${manifest?.version ?? '?'}`)
  lines.push(`  windows signing: ${manifest?.signing?.windows ?? 'none'}`)
  lines.push(`  macos signing:   ${manifest?.signing?.macos ?? 'none'}`)
  lines.push(`  artifacts:       ${manifest?.artifacts?.length ?? 0}`)
  lines.push('')

  if (result.ok) {
    lines.push('PASS — every required artifact is present, signed and intact.')
  } else {
    lines.push(`FAIL — ${result.problems.length} problem(s):`)
    for (const problem of result.problems) {
      lines.push(`  ✗ ${problem}`)
    }
  }

  return lines.join('\n')
}

function parseArgs(argv) {
  const args = {
    dir: 'release',
    version: '',
    platforms: ['win', 'mac', 'linux'],
    archs: null,
    allowUnsigned: false,
    verifyDigests: true
  }

  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i]
    const value = argv[i + 1]

    if ((flag === '--dir' || flag === '--version' || flag === '--tag' || flag === '--platforms' || flag === '--archs') && value) {
      if (flag === '--dir') args.dir = value
      if (flag === '--version' || flag === '--tag') args.version = normalizeVersion(value)
      if (flag === '--archs') args.archs = value.split(',').map(entry => entry.trim()).filter(Boolean)
      if (flag === '--platforms') args.platforms = value.split(',').map(entry => entry.trim()).filter(Boolean)
      i += 1
    } else if (flag === '--allow-unsigned') {
      args.allowUnsigned = true
    } else if (flag === '--skip-digests') {
      args.verifyDigests = false
    }
  }

  return args
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  const dir = path.resolve(args.dir)
  const manifestPath = path.join(dir, MANIFEST_FILENAME)

  if (!fs.existsSync(manifestPath)) {
    console.error(
      `[release-gate] FAIL — ${MANIFEST_FILENAME} not found in ${dir}.\n` +
        '  Run: node scripts/release-manifest.mjs --dir release'
    )
    process.exit(1)
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  const result = evaluateGate({
    manifest,
    version: args.version,
    platforms: args.platforms,
    archs: args.archs,
    allowUnsigned: args.allowUnsigned,
    verifyDigests: args.verifyDigests,
    statSize: file => {
      try {
        return fs.statSync(path.join(dir, file)).size
      } catch {
        return null
      }
    },
    digest: file => {
      try {
        return sha256File(path.join(dir, file))
      } catch {
        return '(unreadable)'
      }
    }
  })

  console.log(renderReport(result, manifest))

  if (!result.ok) {
    process.exit(1)
  }
}

if (isMain(import.meta.url)) {
  main()
}
