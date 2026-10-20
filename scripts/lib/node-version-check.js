#!/usr/bin/env node
// node-version-check.js -- dependency-free semver-range gate for the desktop
// Node floor. Read by node-version-check.sh (and, via posix.sh, by the
// desktop rebuild flow). The single source of truth is
// apps/desktop/package.json's `engines.node`: every run reads that file fresh,
// so when the official dependency declaration changes the gate follows
// automatically -- no manual copy to keep in sync.
//
// Usage:
//   node node-version-check.js <node_version> <path-to-apps/desktop/package.json>
// Exits 0 when <node_version> satisfies engines.node, 1 otherwise.
//
// The range grammar handled here is the practical subset npm uses for
// engines.node: `||` (OR), whitespace (AND), and comparators `^`, `>=`, `>`,
// `<=`, `<`, `=`/bare. Prerelease versions are treated as lower than their
// release counterpart (standard semver), matching npm's behavior for the
// floors Hermes ships.
'use strict'

const fs = require('fs')

function parseVersion(raw) {
  const v = String(raw).replace(/^v/, '').trim()
  const m = v.match(/^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?$/)
  if (!m) return null
  return {
    major: parseInt(m[1], 10),
    minor: m[2] === undefined ? 0 : parseInt(m[2], 10),
    patch: m[3] === undefined ? 0 : parseInt(m[3], 10),
    pre: m[4] || '',
  }
}

function compareVersions(a, b) {
  if (a.major !== b.major) return a.major - b.major
  if (a.minor !== b.minor) return a.minor - b.minor
  if (a.patch !== b.patch) return a.patch - b.patch
  if (!a.pre && !b.pre) return 0
  if (!a.pre) return 1 // release > prerelease
  if (!b.pre) return -1
  return a.pre < b.pre ? -1 : a.pre > b.pre ? 1 : 0
}

// Parse one comparator like "^22.22.0", ">=26", "22.22.0".
function parseComparator(part) {
  const m = part.trim().match(/^(\^|>=|>|<=|<|=)?\s*v?(\d+(?:\.\d+)?(?:\.\d+)?)$/)
  if (!m) return null
  const op = m[1] || '='
  const ver = parseVersion(m[2])
  if (!ver) return null
  // Upper bound for caret: ^X.Y.Z => <(X+1).0.0 (X>0) or <X.(Y+1).0 (X==0).
  let upper = null
  if (op === '^') {
    if (ver.major > 0) {
      upper = { major: ver.major + 1, minor: 0, patch: 0, pre: '' }
    } else if (ver.minor > 0) {
      upper = { major: 0, minor: ver.minor + 1, patch: 0, pre: '' }
    } else {
      upper = { major: 0, minor: 0, patch: ver.patch + 1, pre: '' }
    }
  }
  return { op, ver, upper }
}

function comparatorSatisfies(version, cmp) {
  const d = compareVersions(version, cmp.ver)
  switch (cmp.op) {
    case '=': return d === 0
    case '>=': return d >= 0
    case '>': return d > 0
    case '<=': return d <= 0
    case '<': return d < 0
    case '^': return d >= 0 && compareVersions(version, cmp.upper) < 0
    default: return false
  }
}

function satisfies(version, range) {
  const ver = parseVersion(version)
  if (!ver) return false
  for (const orPart of String(range).split('||')) {
    const andParts = orPart.split(/\s+/).filter((s) => s.length > 0)
    let groupOk = true
    for (const part of andParts) {
      const cmp = parseComparator(part)
      if (!cmp || !comparatorSatisfies(ver, cmp)) {
        groupOk = false
        break
      }
    }
    if (groupOk) return true
  }
  return false
}

function main() {
  const version = process.argv[2]
  const pkgPath = process.argv[3]
  if (!version || !pkgPath) {
    process.stderr.write('usage: node-version-check.js <node_version> <package.json>\n')
    process.exit(2)
  }
  let range
  try {
    range = JSON.parse(fs.readFileSync(pkgPath, 'utf8')).engines.node
  } catch (err) {
    process.stderr.write(`node-version-check: cannot read ${pkgPath}: ${err.message}\n`)
    process.exit(2)
  }
  if (!range) {
    process.stderr.write(`node-version-check: ${pkgPath} has no engines.node\n`)
    process.exit(2)
  }
  process.exit(satisfies(version, range) ? 0 : 1)
}

main()
