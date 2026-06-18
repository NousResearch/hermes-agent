'use strict'

const path = require('node:path')

function uniqueValues(values) {
  const seen = new Set()
  const result = []
  for (const value of values) {
    if (!value || seen.has(value)) continue
    seen.add(value)
    result.push(value)
  }
  return result
}

function rebuiltMacAppCandidates(updateRoot, arch = process.arch) {
  const releaseDir = path.join(updateRoot, 'apps', 'desktop', 'release')
  const currentArchDir = arch === 'x64' ? 'mac-x64' : arch === 'arm64' ? 'mac-arm64' : null
  return uniqueValues([currentArchDir, 'mac-arm64', 'mac-x64', 'mac']).map(directory =>
    path.join(releaseDir, directory, 'Hermes.app')
  )
}

module.exports = { rebuiltMacAppCandidates }
