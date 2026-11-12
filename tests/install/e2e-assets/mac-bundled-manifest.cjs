// Independent native signature and installed-stamp assertions.
const { isDeepStrictEqual } = require('node:util')

function channelStampAssertions(stamp, request) {
  const expected = { channelBuild: request, commit: request.commit, tag: null, source: 'channel-build',
    branch: null, dirty: false, baseVersion: request.sourceVersion,
    displayVersion: `${request.sourceVersion} (${request.channel} #${request.sequence}, ${request.commit.slice(0, 7)})` }
  if (request.receiverCandidate) Object.assign(expected, {
    channelBuild: undefined, tag: request.releaseTag, source: 'build',
    displayVersion: request.version, receiverProtocol: 1,
  })
  return Object.entries(expected).filter(([key, value]) => !isDeepStrictEqual(stamp?.[key], value))
    .map(([key]) => `stamp.${key} disagrees with admitted channel request`)
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
  if (side.channelRequest) problems.push(...channelStampAssertions(stamp, side.channelRequest))
  else if (stamp.channelBuild != null) problems.push('Unexpected stamp.channelBuild without an admitted request')
  return problems
}

module.exports = { codesignTeam, stampAssertions, channelStampAssertions }
