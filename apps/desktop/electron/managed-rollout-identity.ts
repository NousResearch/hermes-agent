import crypto from 'node:crypto'

const INSTALL_ID_RE = /^[0-9a-f]{32}$/
const SHA256_RE = /^[0-9a-f]{64}$/
const SAFE_SEGMENT_RE = /^[A-Za-z0-9._-]+$/

export interface InstallationFingerprintInput {
  installId: string
  codeRoot: string
  repositoryId: string
  platform?: NodeJS.Platform
}

export interface SourceFingerprintInput {
  installationFingerprint: string
  connectionId: string
  connectionConfigRevision: string | number
  verifiedHostKeyFingerprint: string
  remoteUser: string
  port: number
  configuredProfile: string
  configuredCodePath: string
}

function invalid(message: string): never {
  throw new Error(message)
}

function text(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.length === 0 || /[\x00\r\n]/.test(value)) {
    invalid(`invalid-${label}`)
  }

  return value
}

function sha256Tuple(tuple: readonly unknown[]): string {
  return crypto.createHash('sha256').update(JSON.stringify(tuple), 'utf8').digest('hex')
}

function stripRepositorySuffix(value: string): string {
  const withoutSlash = value.replace(/\/+$/, '')
  const withoutGit = withoutSlash.endsWith('.git') ? withoutSlash.slice(0, -4) : withoutSlash

  if (withoutGit.endsWith('.git') || !withoutGit) invalid('invalid-repository-path')

  return withoutGit
}

function repositorySegments(pathname: string): [string, string] {
  let decoded: string

  try {
    decoded = decodeURIComponent(pathname)
  } catch {
    invalid('invalid-repository-path')
  }

  const path = stripRepositorySuffix(decoded)
  const parts = path.split('/').filter(Boolean)

  if (parts.length !== 2 || parts.some(part => !SAFE_SEGMENT_RE.test(part))) invalid('invalid-repository-path')

  return [parts[0], parts[1]]
}

function githubRepository(owner: string, repository: string): string {
  return `github.com/${owner.toLowerCase()}/${repository.toLowerCase()}`
}

/**
 * Canonicalize the supported source spellings without carrying credentials into
 * identity. GitHub has an explicit cross-protocol rule; other hosts retain a
 * conservative scheme/host/port/path identity.
 */
export function canonicalRepositoryId(origin: string): string {
  const raw = text(origin, 'repository-origin')
  if (raw !== raw.trim()) invalid('invalid-repository-origin')
  const value = raw

  if (value.includes('?') || value.includes('#') || /\s/.test(value)) invalid('invalid-repository-origin')

  const scp = /^([^@/:]+)@([^/:]+):(.+)$/.exec(value)

  if (scp) {
    const [, user, host, repositoryPath] = scp

    if (user !== 'git' || host.toLowerCase() !== 'github.com') {
      if (host.includes('.') || host === 'localhost') {
        const [owner, repository] = repositorySegments(`/${repositoryPath}`)
        return `ssh://${host.toLowerCase()}/${owner}/${repository}`
      }

      invalid('unsupported-repository-origin')
    }

    const [owner, repository] = repositorySegments(`/${repositoryPath}`)

    return githubRepository(owner, repository)
  }

  if (!/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(value)) invalid('ambiguous-repository-origin')

  let parsed: URL

  try {
    parsed = new URL(value)
  } catch {
    invalid('invalid-repository-origin')
  }

  const protocol = parsed.protocol.toLowerCase()
  const host = parsed.hostname.toLowerCase()

  if (!host || parsed.password) invalid('repository-origin-contains-credentials')

  if (host === 'github.com') {
    if (protocol !== 'https:' && protocol !== 'ssh:') invalid('unsupported-github-origin')
    if (parsed.port && parsed.port !== (protocol === 'https:' ? '443' : '22')) invalid('unsupported-github-port')
    if (protocol === 'ssh:' && parsed.username !== 'git') invalid('unsupported-github-user')
    if (protocol === 'https:' && parsed.username) invalid('repository-origin-contains-credentials')

    const [owner, repository] = repositorySegments(parsed.pathname)

    return githubRepository(owner, repository)
  }

  if (!['http:', 'https:', 'ssh:', 'git:'].includes(protocol)) invalid('unsupported-repository-origin')
  if ((protocol === 'http:' || protocol === 'https:') && (parsed.username || parsed.password)) {
    invalid('repository-origin-contains-credentials')
  }

  const pathname = stripRepositorySuffix(parsed.pathname)

  if (!pathname.startsWith('/') || pathname === '/' || pathname.includes('//')) invalid('invalid-repository-path')

  return `${protocol}//${host}${parsed.port ? `:${parsed.port}` : ''}${pathname}`
}

/** Normalize the canonical values accepted from an already-inspected probe. */
function normalizeRepositoryId(value: string): string {
  if (/^github\.com\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(value)) {
    const [, owner, repository] = value.split('/')

    return githubRepository(owner, repository)
  }

  if (/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(value)) return canonicalRepositoryId(value)

  invalid('invalid-repository-id')
}

/**
 * Preserve the remote probe's canonical root. Only platform path separators and
 * a Windows drive-letter form are normalized; path case and components remain
 * authoritative data from the probe.
 */
export function canonicalCodeRoot(codeRoot: string, platform: NodeJS.Platform = process.platform): string {
  let value = text(codeRoot, 'code-root')
  if (value !== value.trim()) invalid('invalid-code-root')

  if (platform === 'win32') {
    value = value.replace(/\\/g, '/')
    value = value.replace(/^([a-z]):/, (_, drive: string) => `${drive.toUpperCase()}:`)
    if (value.length > 3) value = value.replace(/\/+$/, '')
  } else if (value.length > 1) {
    value = value.replace(/\/+$/, '')
  }

  return value
}

export function installationFingerprint(input: InstallationFingerprintInput): string {
  const installId = text(input.installId, 'install-id')
  if (!INSTALL_ID_RE.test(installId)) invalid('invalid-install-id')

  const root = canonicalCodeRoot(input.codeRoot, input.platform)
  const repositoryId = normalizeRepositoryId(text(input.repositoryId, 'repository-id'))

  return sha256Tuple([1, installId, root, repositoryId])
}

export function sourceFingerprint(input: SourceFingerprintInput): string {
  if (!SHA256_RE.test(text(input.installationFingerprint, 'installation-fingerprint'))) {
    invalid('invalid-installation-fingerprint')
  }
  const connectionId = text(input.connectionId, 'connection-id')
  const revision = input.connectionConfigRevision

  if (!(
    typeof revision === 'string' ||
    (typeof revision === 'number' && Number.isSafeInteger(revision) && revision >= 0)
  )) {
    invalid('invalid-connection-config-revision')
  }

  const hostKey = text(input.verifiedHostKeyFingerprint, 'host-key-fingerprint')
  const remoteUser = text(input.remoteUser, 'remote-user')
  const configuredProfile = text(input.configuredProfile, 'configured-profile')
  const configuredCodePath = text(input.configuredCodePath, 'configured-code-path')

  if (!Number.isSafeInteger(input.port) || input.port < 1 || input.port > 65_535) invalid('invalid-port')

  return sha256Tuple([
    1,
    input.installationFingerprint,
    connectionId,
    revision,
    hostKey,
    remoteUser,
    input.port,
    configuredProfile,
    configuredCodePath
  ])
}

export function installationFingerprintTuple(
  input: InstallationFingerprintInput
): readonly [1, string, string, string] {
  const installId = text(input.installId, 'install-id')
  if (!INSTALL_ID_RE.test(installId)) invalid('invalid-install-id')

  return [
    1,
    installId,
    canonicalCodeRoot(input.codeRoot, input.platform),
    normalizeRepositoryId(text(input.repositoryId, 'repository-id'))
  ]
}

export function sourceFingerprintTuple(
  input: SourceFingerprintInput
): readonly [1, string, string, string | number, string, string, number, string, string] {
  sourceFingerprint(input)

  return [
    1,
    input.installationFingerprint,
    input.connectionId,
    input.connectionConfigRevision,
    input.verifiedHostKeyFingerprint,
    input.remoteUser,
    input.port,
    input.configuredProfile,
    input.configuredCodePath
  ]
}

export function installationIdentityConflict(
  left: InstallationFingerprintInput,
  right: InstallationFingerprintInput
): boolean {
  if (left.installId !== right.installId) return false

  return (
    canonicalCodeRoot(left.codeRoot, left.platform) !== canonicalCodeRoot(right.codeRoot, right.platform) ||
    normalizeRepositoryId(left.repositoryId) !== normalizeRepositoryId(right.repositoryId)
  )
}

export function sameInstallation(left: InstallationFingerprintInput, right: InstallationFingerprintInput): boolean {
  return !installationIdentityConflict(left, right) && left.installId === right.installId
}

export function sourceBindingChanged(left: SourceFingerprintInput, right: SourceFingerprintInput): boolean {
  return sourceFingerprint(left) !== sourceFingerprint(right)
}
