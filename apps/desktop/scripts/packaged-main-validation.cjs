'use strict'

const { builtinModules } = require('node:module')

const BUILTIN_MODULES = new Set([...builtinModules, ...builtinModules.map(name => `node:${name}`)])
const ALLOWED_BARE_REQUIRES = new Set(['electron', 'node-pty'])
const REQUIRE_SPEC_RE = /require\s*\(\s*['"`]([^'"`]+)['"`]\s*\)/g

function findUnexpectedPackagedMainRequires(source) {
  const unexpected = new Set()
  for (const match of source.matchAll(REQUIRE_SPEC_RE)) {
    const spec = match[1]
    if (spec.startsWith('.')) {
      unexpected.add(spec)
      continue
    }
    if (!BUILTIN_MODULES.has(spec) && !ALLOWED_BARE_REQUIRES.has(spec)) {
      unexpected.add(spec)
    }
  }
  return [...unexpected].sort()
}

module.exports = { findUnexpectedPackagedMainRequires }
