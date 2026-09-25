/* The protocol decoder is the sole untrusted-JSON boundary. Unknown inputs and
 * primitive checks belong here; only validated domain objects leave this file. */
/* oxlint-disable anti-slop/no-unknown-parameters, anti-slop/no-unknown-returns, anti-slop/no-runtime-typeof, anti-slop/no-reflect-get */
import feedContract from '../../update-feed.cjs'

export interface ChannelIdentity {
  token: string
  displayName: string
  appId: string
  appNamePascal: string
  artifactNamePascal: string
  cliName: string
  windowsExecutableName: string
  msixAppIdWithOrg: string
}

/** Build-stamp facts, independent of the Python payload or stamp module. */
export interface ChannelBuild {
  buildId: string
  channel: string
  sequence: number
  repository: string
  commit: string
  sourceVersion: string
  version: string
  windowsVersion: string
  identity: ChannelIdentity
  bundleEnv: { [name: string]: string | null }
  publicBase: string
}

export interface ChannelRequest extends ChannelBuild {
  schema: 1
  controllerCommit?: string
  releaseTag?: string
  /** Attempt ref naming the immutable archive; only the releases/tag/ prefix reads it. */
  archiveRef?: string
}
export interface ChannelHead {
  buildId: string
  sequence: number
  manifestKey: string
  sha256: string
}
interface ChannelRecordBase {
  schema: 1
  name: string
  repository: string
  policy: 'preview' | 'stable-release' | 'canary-release'
  revision: number
  nextSequence: number
  identity: ChannelIdentity
  head: ChannelHead | null
}
export interface ActiveChannel extends ChannelRecordBase {
  state: 'active'
}
export type ReceiverKind = 'in-place' | 'discontinued'
export interface RetiredChannel extends ChannelRecordBase {
  state: 'retired'
  destination: string
  minimumVersion: string
  destinationHead: ChannelHead
  receiverProtocol: 1
  receiver: { kind: ReceiverKind }
  lastHead: ChannelHead | null
}
export type ChannelRecord = ActiveChannel | RetiredChannel
export interface ChannelPackage {
  platform: 'darwin' | 'win32'
  arch: 'x64' | 'arm64'
  variant: 'bundled'
  version: string
  identity: string
  artifact: { key: string; sha256: string; size: number }
  publisher?: string
  teamId?: string
  feed: { key: string; channel: string }
}
export interface ChannelManifest {
  schema: 1
  request: ChannelRequest
  packages: ChannelPackage[]
  receiverProtocol?: number
}

const SHA256 = /^[a-f0-9]{64}$/
const COMMIT = /^[a-f0-9]{40}$/
const BUILD_ID = /^[a-f0-9]{32}$/
const VERSION =
  /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$/
// rc.<N>-vX.Y.Z with N >= 1 without leading zeros and a release version whose
// major stays within three digits — one regex matching the Python grammar.
const ARCHIVE_REF = /^rc\.(?:[1-9]\d*)-v(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/

function parseChannelJson(body: string): unknown {
  const parsed: unknown = JSON.parse(body)
  // JSON.parse discards duplicate members; scan validated JSON tokens to keep
  // the same reject-duplicates contract as the publisher's object_pairs_hook.
  const objects: Set<string>[] = []
  const tokens = /"(?:[^"\\]|\\[\s\S])*"\s*:|"(?:[^"\\]|\\[\s\S])*"|[{}]/g

  for (const match of body.matchAll(tokens)) {
    const token = match[0]

    if (token === '{') {
      objects.push(new Set<string>())

      continue
    }

    if (token === '}') {
      objects.pop()

      continue
    }

    if (!token.endsWith(':')) {
      continue
    }

    const key: string = JSON.parse(token.slice(0, -1))
    const keys = objects[objects.length - 1]

    if (keys.has(key)) {
      throw new Error('Duplicate channel JSON member')
    }

    keys.add(key)
  }

  return parsed
}

/** Untrusted JSON is confined to these network decoders. */
class Fields {
  private readonly input: object
  constructor(input: unknown) {
    if (!input || typeof input !== 'object' || Array.isArray(input)) {
      throw new Error('Expected channel object')
    }

    this.input = input
  }
  get(key: string): unknown {
    return Reflect.get(this.input, key)
  }
  keys(): string[] {
    return Object.keys(this.input)
  }
  text(key: string, pattern: RegExp = /^[\s\S]{1,2048}$/): string {
    const value: unknown = this.get(key)

    if (
      typeof value !== 'string' ||
      [...value].some((character: string): boolean => character.charCodeAt(0) < 32) ||
      !pattern.test(value)
    ) {
      throw new Error(`Invalid channel ${key}`)
    }

    return value
  }
  integer(key: string, maximum: number = Number.MAX_SAFE_INTEGER): number {
    const value: unknown = this.get(key)

    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > maximum) {
      throw new Error(`Invalid channel ${key}`)
    }

    return value
  }
  object(key: string): Fields {
    return new Fields(this.get(key))
  }
  optional(key: string, pattern?: RegExp): string | undefined {
    return this.get(key) === undefined ? undefined : this.text(key, pattern)
  }
  schema(): void {
    if (this.get('schema') !== 1) {
      throw new Error('Unsupported channel schema')
    }
  }
}

export function validateChannelName(name: string): string {
  // This validates syntax, not whether R2 has registered the name.
  feedContract.darwinFeed(name)

  return name
}

export function channelPublicBase(value: string): string {
  const url = new URL(value)
  const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)

  if (
    (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    /[%\\;\s]/.test(value) ||
    value.split('/').some((part: string): boolean => part === '.' || part === '..')
  ) {
    throw new Error('Channel archive must use HTTPS or loopback HTTP without credentials or traversal')
  }

  return value.replace(/\/+$/, '')
}

export function channelKey(value: string): string {
  if (
    !/^releases\/[A-Za-z0-9_./+-]+$/.test(value) ||
    value
      .split('/')
      .some(
        (part: string): boolean =>
          !part ||
          part === '.' ||
          part === '..' ||
          part.endsWith('.') ||
          /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part)
      )
  ) {
    throw new Error('Invalid channel object key')
  }

  return value
}

export function buildPrefix(buildId: string): string {
  if (!BUILD_ID.test(buildId)) {
    throw new Error('Invalid channel buildId')
  }

  return `releases/channel-builds/${buildId}/`
}

function identity(fields: Fields): ChannelIdentity {
  return {
    token: fields.text('token', /^[a-f0-9]{16}$/),
    displayName: fields.text('displayName', /^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$/),
    appId: fields.text('appId', /^[a-z][a-z0-9.-]{2,127}$/),
    appNamePascal: fields.text('appNamePascal', /^[A-Za-z][A-Za-z0-9]{0,63}$/),
    artifactNamePascal: fields.text('artifactNamePascal', /^[A-Za-z][A-Za-z0-9]{0,63}$/),
    cliName: fields.text('cliName', /^[a-z][a-z0-9-]{0,63}$/),
    windowsExecutableName: fields.text('windowsExecutableName', /^[A-Za-z][A-Za-z0-9 ._-]{0,79}$/),
    msixAppIdWithOrg: fields.text('msixAppIdWithOrg', /^[A-Za-z][A-Za-z0-9.-]{2,49}$/)
  }
}

export function sameChannelIdentity(left: ChannelIdentity, right: ChannelIdentity): boolean {
  return (
    left.token === right.token &&
    left.displayName === right.displayName &&
    left.appId === right.appId &&
    left.appNamePascal === right.appNamePascal &&
    left.artifactNamePascal === right.artifactNamePascal &&
    left.cliName === right.cliName &&
    left.windowsExecutableName === right.windowsExecutableName &&
    left.msixAppIdWithOrg === right.msixAppIdWithOrg
  )
}

function head(value: unknown): ChannelHead | null {
  if (value === null) {
    return null
  }

  const fields = new Fields(value)
  const buildId = fields.text('buildId', BUILD_ID)
  const manifestKey = fields.text('manifestKey')

  if (manifestKey !== `${buildPrefix(buildId)}build.json`) {
    throw new Error('Invalid channel manifestKey')
  }

  return {
    buildId,
    manifestKey,
    sequence: fields.integer('sequence', 0xffffffff),
    sha256: fields.text('sha256', SHA256)
  }
}

function requiredHead(value: unknown): ChannelHead {
  const result = head(value)

  if (!result) {
    throw new Error('Missing retirement destination head')
  }

  return result
}

export function decodeChannelRecord(body: string): ChannelRecord {
  const fields = new Fields(parseChannelJson(body))
  fields.schema()
  const policy = fields.text('policy')

  if (policy !== 'preview' && policy !== 'stable-release' && policy !== 'canary-release') {
    throw new Error('Channel has no supported desktop delivery')
  }

  const common: ChannelRecordBase = {
    schema: 1,
    name: validateChannelName(fields.text('name')),
    repository: fields.text('repository', /^[A-Za-z0-9][A-Za-z0-9_-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/),
    policy,
    identity: identity(fields.object('identity')),
    revision: fields.integer('revision'),
    nextSequence: fields.integer('nextSequence', 0x100000000),
    head: head(fields.get('head'))
  }

  if (common.head && common.head.sequence >= common.nextSequence) {
    throw new Error('Channel head exceeds allocation')
  }

  const state = fields.text('state')

  if (state === 'active') {
    return { ...common, state }
  }

  if (state !== 'retired') {
    throw new Error('Invalid channel state')
  }

  const lastHead = head(fields.get('lastHead'))

  if (JSON.stringify(lastHead) !== JSON.stringify(common.head)) {
    throw new Error('Retirement lastHead mismatch')
  }

  if (fields.get('receiverProtocol') !== 1) {
    throw new Error('Unsupported retirement receiver protocol')
  }

  const receiverFields = fields.object('receiver')
  const kind = receiverFields.text('kind')

  if (kind !== 'in-place' && kind !== 'discontinued') {
    throw new Error('Invalid retirement receiver kind')
  }

  return {
    ...common,
    state,
    lastHead,
    destination: validateChannelName(fields.text('destination')),
    minimumVersion: fields.text('minimumVersion', VERSION),
    destinationHead: requiredHead(fields.get('destinationHead')),
    receiverProtocol: 1,
    receiver: { kind }
  }
}

function request(fields: Fields): ChannelRequest {
  fields.schema()
  const environment = fields.object('bundleEnv')
  const bundleEnv: ChannelBuild['bundleEnv'] = {}

  // Match the build-time allowlist: a channel request cannot inject process flags.
  const allowed = new Set([
    'HERMES_HOME',
    'HERMES_DATA_DIR_SUFFIX',
    'HERMES_DESKTOP_USER_DATA_DIR',
    'HERMES_SHARED_AUTH_DIR',
    'HERMES_GUEST_ONBOARDING',
    'HERMES_SKIP_INTRO'
  ])

  for (const key of environment.keys()) {
    if (!allowed.has(key)) {
      throw new Error('Invalid bundle environment name')
    }

    const value: unknown = environment.get(key)

    if (value !== null && (typeof value !== 'string' || value.includes('\0'))) {
      throw new Error('Invalid bundle environment value')
    }

    Object.defineProperty(bundleEnv, key, { value, enumerable: true })
  }

  const publicBase = fields.text('publicBase')

  if (channelPublicBase(publicBase) !== publicBase) {
    throw new Error('Noncanonical request publicBase')
  }

  const releaseTag = fields.optional('releaseTag', /^v\d+\.\d+\.\d+(?:\+canary\.20\d{6}T\d{6}Z)?$/)

  const canaryRelease: boolean = /\+canary\./.test(releaseTag || '')

  const windowsVersion = fields.text('windowsVersion', canaryRelease ? /^\d+\.\d+\.\d+\.\d+$/ : /^\d+\.\d+\.\d+\.0$/)

  if (windowsVersion.split('.').some((part: string): boolean => Number(part) > 65535)) {
    throw new Error('Invalid Windows version')
  }

  return {
    schema: 1,
    buildId: fields.text('buildId', BUILD_ID),
    channel: validateChannelName(fields.text('channel')),
    sequence: fields.integer('sequence', 0xffffffff),
    repository: fields.text('repository', /^[A-Za-z0-9][A-Za-z0-9_-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/),
    commit: fields.text('commit', COMMIT),
    sourceVersion: fields.text('sourceVersion', VERSION),
    version: fields.text('version', VERSION),
    windowsVersion,
    publicBase,
    bundleEnv,
    identity: identity(fields.object('identity')),
    controllerCommit: fields.optional('controllerCommit', COMMIT),
    releaseTag,
    archiveRef: fields.optional('archiveRef', ARCHIVE_REF)
  }
}

function packageEntry(value: unknown): ChannelPackage {
  const fields = new Fields(value)
  const platform = fields.text('platform')
  const arch = fields.text('arch')

  if (
    (platform !== 'darwin' && platform !== 'win32') ||
    (arch !== 'x64' && arch !== 'arm64') ||
    fields.get('variant') !== 'bundled'
  ) {
    throw new Error('Unsupported channel package')
  }

  const artifact = fields.object('artifact')
  const feed = fields.object('feed')

  return {
    platform,
    arch,
    variant: 'bundled',
    version: fields.text('version', platform === 'darwin' ? VERSION : /^\d+\.\d+\.\d+\.\d+$/),
    identity: fields.text('identity'),
    artifact: {
      key: channelKey(artifact.text('key')),
      sha256: artifact.text('sha256', SHA256),
      size: artifact.integer('size')
    },
    publisher: fields.optional('publisher'),
    teamId: fields.optional('teamId', /^[A-Z0-9]{10}$/),
    feed: { key: channelKey(feed.text('key')), channel: feed.text('channel', /^[a-z][a-z0-9-]{0,31}$/) }
  }
}

export function decodeChannelManifest(body: string): ChannelManifest {
  const fields = new Fields(parseChannelJson(body))
  fields.schema()
  const entries: unknown = fields.get('packages')

  if (!Array.isArray(entries) || !entries.length || entries.length > 4) {
    throw new Error('Invalid channel packages')
  }

  const packages: ChannelPackage[] = entries.map(packageEntry)
  const keys = new Set(packages.map((entry: ChannelPackage): string => `${entry.platform}/${entry.arch}`))

  if (keys.size !== packages.length) {
    throw new Error('Duplicate channel package')
  }

  const manifest: ChannelManifest = { schema: 1, request: request(fields.object('request')), packages }

  if (fields.get('receiverProtocol') !== undefined) {
    manifest.receiverProtocol = fields.integer('receiverProtocol')
  }

  return manifest
}
