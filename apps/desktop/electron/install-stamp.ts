const INSTALL_STAMP_SCHEMA_VERSION = 1
const DEFAULT_GITHUB_REPOSITORY = 'NousResearch/hermes-agent'

function optionalString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function normalizeInstallStampPayload(payload, expectedSchemaVersion = INSTALL_STAMP_SCHEMA_VERSION) {
  if (!payload || typeof payload !== 'object') return null
  if (payload.schemaVersion !== expectedSchemaVersion) return null
  if (typeof payload.commit !== 'string' || payload.commit.length < 7) return null

  return Object.freeze({
    schemaVersion: payload.schemaVersion,
    commit: payload.commit,
    branch: optionalString(payload.branch),
    repository: optionalString(payload.repository),
    bootstrapRef: optionalString(payload.bootstrapRef),
    commitPinned: typeof payload.commitPinned === 'boolean' ? payload.commitPinned : null,
    repoUrlHttps: optionalString(payload.repoUrlHttps),
    repoUrlSsh: optionalString(payload.repoUrlSsh),
    builtAt: optionalString(payload.builtAt),
    dirty: Boolean(payload.dirty),
    source: optionalString(payload.source)
  })
}

function isLocalProtectedInstallStamp(installStamp) {
  if (!installStamp || installStamp.source !== 'local') return false
  if (installStamp.commitPinned === true && installStamp.repository === DEFAULT_GITHUB_REPOSITORY) {
    return false
  }
  return true
}

function localBuildUpdateBlock(installStamp, env = process.env) {
  if (!isLocalProtectedInstallStamp(installStamp)) return null
  if (env.HERMES_DESKTOP_ALLOW_LOCAL_UNPINNED_UPDATE === '1') return null

  const branch = installStamp.branch || installStamp.bootstrapRef || 'the reachable branch'
  const commit = installStamp.commit ? installStamp.commit.slice(0, 12) : 'unknown commit'
  const repo = installStamp.repository || DEFAULT_GITHUB_REPOSITORY
  return {
    ok: true,
    manual: true,
    reason: 'local-build-install',
    command: 'hermes update && hermes desktop',
    message:
      `This Hermes Desktop build was made locally from ${repo}@${commit} on ${branch}. ` +
      'Automatic self-update is paused so it does not replace the local desktop bundle with a different repository or branch.'
  }
}

export {
  DEFAULT_GITHUB_REPOSITORY,
  INSTALL_STAMP_SCHEMA_VERSION,
  isLocalProtectedInstallStamp,
  localBuildUpdateBlock,
  normalizeInstallStampPayload
}
