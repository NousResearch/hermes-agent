#!/usr/bin/env node
'use strict'
// mac-bundled-manifest.cjs — pure validation/assertion helpers for the
// macOS packaged-app -> open-app-update E2E arm
// (tests/install/macos-bundled-e2e.sh). No dependencies, no side effects:
// everything here is requireable from vitest (tests-js) and from the
// driver's node invocations on the macOS runner.
//
// The bundle manifest itself is RESOLVED by the PARENT-owned common
// resolver tests/install/e2e-assets/bundle-inputs.mjs (schema/commit/
// sha256/semver validation + artifact download). This module asserts the
// macOS-specific semantics on the resolver's normalized output
// (WORKROOT/bundle-inputs.json): per-side teamId + CFBundleIdentifier,
// exact-identity update rules, and channel consistency.

const ARCHES = ['arm64', 'x64']
const SHA256_RE = /^[0-9a-f]{64}$/
// stable tags: vX.Y.Z; canary: vX.Y.Z-canary.YYYYMMDDHHMMSS
const TAG_RE = /^v\d+\.\d+\.\d+(?:-canary\.20\d{6}(?:\d{6})?)?$/
// CFBundleIdentifier shape (exact match is enforced at verify time; this
// only rejects obvious junk).
const BUNDLE_ID_RE = /^[A-Za-z][A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$/

/** The update channel a release tag publishes to (the production
 * darwinFeed channel). Canary tags feed the canary channel. */
function channelFromTag(tag) {
  return /-canary\./.test(String(tag)) ? 'canary' : 'stable'
}

function validateSide(side, label) {
  if (!side || typeof side !== 'object') {
    throw new Error(`${label}: missing side object`)
  }
  for (const key of ['tag', 'version', 'commit', 'identity', 'teamId']) {
    if (typeof side[key] !== 'string' || !side[key]) {
      throw new Error(`${label}.${key}: expected a non-empty string`)
    }
  }
  if (!TAG_RE.test(side.tag)) {
    throw new Error(`${label}.tag: ${JSON.stringify(side.tag)} is not a release tag`)
  }
  if (side.version !== side.tag.slice(1)) {
    throw new Error(`${label}: version ${side.version} does not match tag ${side.tag}`)
  }
  if (!/^[0-9a-f]{40}$/.test(side.commit)) {
    throw new Error(`${label}.commit: expected a 40-hex SHA, got ${JSON.stringify(side.commit)}`)
  }
  // identity IS the CFBundleIdentifier (not the team): verified exactly,
  // never by namespace prefix.
  if (!BUNDLE_ID_RE.test(side.identity)) {
    throw new Error(`${label}.identity: ${JSON.stringify(side.identity)} is not a CFBundleIdentifier`)
  }
  // Same rule the common resolver enforces (bundle-inputs.mjs): an exact
  // 10-char Apple signing team id.
  if (!/^[A-Z0-9]{10}$/.test(side.teamId)) {
    throw new Error(`${label}.teamId: ${JSON.stringify(side.teamId)} is not a 10-char signing team id`)
  }
  const artifact = side.artifact
  if (!artifact || typeof artifact !== 'object') {
    throw new Error(`${label}.artifact: missing object`)
  }
  if (typeof artifact.url !== 'string' || !artifact.url) {
    throw new Error(`${label}.artifact.url: expected a non-empty string`)
  }
  if (typeof artifact.path !== 'string' || !artifact.path) {
    throw new Error(`${label}.artifact.path: missing; the parent resolver must download the artifact first`)
  }
  if (!SHA256_RE.test(String(artifact.sha256 || '').toLowerCase())) {
    throw new Error(`${label}.artifact.sha256: expected 64 hex chars`)
  }
}

/**
 * Validate the normalized bundle-inputs.json for this arm.
 * @param {{schema?: unknown, platform?: unknown, arch?: unknown,
 *          old?: unknown, new?: unknown}} manifest
 * @param {{platform: 'macos', arch: 'arm64'|'x64'}} want
 */
function validateBundleManifest(manifest, want) {
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('bundle-inputs.json: not an object')
  }
  if (manifest.schema !== 1) {
    throw new Error(`bundle-inputs.json: unsupported schema ${JSON.stringify(manifest.schema)}`)
  }
  if (manifest.platform !== want.platform) {
    throw new Error(`bundle-inputs.json: platform ${JSON.stringify(manifest.platform)} != ${want.platform}`)
  }
  if (!ARCHES.includes(want.arch)) {
    throw new Error(`arch: ${JSON.stringify(want.arch)} is not one of ${ARCHES.join('|')}`)
  }
  if (manifest.arch !== want.arch) {
    throw new Error(`bundle-inputs.json: arch ${JSON.stringify(manifest.arch)} != ${want.arch}`)
  }
  validateSide(manifest.old, 'old')
  validateSide(manifest.new, 'new')
  if (manifest.old.commit === manifest.new.commit) {
    throw new Error('old.commit == new.commit: no update would be available')
  }
  if (manifest.old.identity !== manifest.new.identity) {
    // The updater only replaces a bundle whose CFBundleIdentifier matches.
    throw new Error(
      `old.identity ${manifest.old.identity} != new.identity ${manifest.new.identity}: ` +
      'not the same application bundle')
  }
  if (manifest.old.teamId !== manifest.new.teamId) {
    // Squirrel.Mac refuses an update whose code signature changes team.
    throw new Error(
      `old.teamId ${manifest.old.teamId} != new.teamId ${manifest.new.teamId}: ` +
      'Squirrel.Mac would reject this pair (signature gate)')
  }
  if (channelFromTag(manifest.old.tag) !== channelFromTag(manifest.new.tag)) {
    throw new Error(
      `channel mismatch: old tag ${manifest.old.tag} vs new tag ${manifest.new.tag} — ` +
      'the update must stay on one feed channel')
  }
  return manifest
}

/**
 * Parse the TeamIdentifier out of `codesign -dv` output. codesign prints
 * display output on STDERR, so pass stderr (or combined output) here.
 * Returns null for an unsigned / ad-hoc ("not set") signature.
 */
function codesignTeam(codesignOutput) {
  const match = /^TeamIdentifier=(.*)$/m.exec(String(codesignOutput || ''))
  if (!match) return null
  const value = match[1].trim()
  if (!value || value === 'not set' || value === '-') return null
  return value
}

/** Assertions for an installed bundle's install-stamp.json
 * (Contents/Resources/install-stamp.json) against a manifest side.
 * Mirrors apps/desktop/scripts/write-build-stamp.mjs's bundled shape. */
function stampAssertions(stamp, side) {
  const problems = []
  if (!stamp || typeof stamp !== 'object') {
    return ['install-stamp.json: not an object']
  }
  if (stamp.payload !== 'bundled') {
    problems.push(`stamp.payload ${JSON.stringify(stamp.payload)} != 'bundled'`)
  }
  if (stamp.updateMechanism !== 'electron-updater') {
    problems.push(`stamp.updateMechanism ${JSON.stringify(stamp.updateMechanism)} != 'electron-updater'`)
  }
  if (stamp.commit !== side.commit) {
    problems.push(`stamp.commit ${JSON.stringify(stamp.commit)} != ${side.commit}`)
  }
  if (stamp.tag !== side.tag) {
    problems.push(`stamp.tag ${JSON.stringify(stamp.tag)} != ${side.tag}`)
  }
  return problems
}

module.exports = {
  ARCHES,
  channelFromTag,
  validateBundleManifest,
  codesignTeam,
  stampAssertions,
}
