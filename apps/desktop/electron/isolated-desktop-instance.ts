/**
 * Isolated Desktop instance helpers — electron-free so they can be tested
 * with vitest without booting Electron.
 *
 * The Python CLI (`hermes_cli.desktop_instances`) is the Windows/macOS/Linux
 * installer. This module is the in-app twin: map a Connections SSH entry to
 * an isolated-shell spec, parse instance deep links, and decide whether this
 * process should own the global protocol handler / quick-entry hotkey / AUMID.
 */

export const INSTANCE_AUMID_PREFIX = 'com.nousresearch.hermes.instance.'
export const DEFAULT_AUMID = 'com.nousresearch.hermes'

export interface IsolatedSshConnection {
  kind?: string
  label?: string
  host?: string
  user?: string
  port?: number
  keyPath?: string
  remoteHermesPath?: string
  remoteProfile?: string
  connectionId?: string
  id?: string
}

export interface IsolatedInstanceSpec {
  name: string
  displayName: string
  connectionId: string
  sshHost: string
  sshUser: string
  sshPort: number
  sshKeyPath: string
  remoteHermesPath: string
  remoteProfile: string
  aumid: string
  dialIdentity: string
}

export interface InstanceDeepLink {
  instanceName: string
  remainder: string
}

const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

const RESERVED_NAMES = new Set([
  'connections',
  'default',
  'desktop',
  'gui',
  'hermes',
  'instance',
  'local',
  'root',
  'sudo',
  'test',
  'tmp'
])

export function instanceAumid(name: string): string {
  return `${INSTANCE_AUMID_PREFIX}${name}`
}

export function slugFromLabel(label: string): string {
  let text = String(label || '').trim()

  if (text.toLowerCase().startsWith('hermes ')) {
    text = text.slice(6).trim()
  }

  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  if (!NAME_RE.test(slug)) {
    throw new Error(`Cannot derive an instance name from ${JSON.stringify(label)}.`)
  }

  if (RESERVED_NAMES.has(slug)) {
    throw new Error(`Instance name ${JSON.stringify(slug)} is reserved.`)
  }

  return slug
}

export function isolatedInstanceSpecFromSsh(connection: IsolatedSshConnection): IsolatedInstanceSpec {
  if (String(connection.kind || '').trim().toLowerCase() !== 'ssh') {
    throw new Error('Only SSH Connections can open as an isolated Desktop.')
  }

  const host = String(connection.host || '').trim()

  if (!host) {
    throw new Error('SSH host is required.')
  }

  const remoteHermesPath = String(connection.remoteHermesPath || '').trim()
  const posixAbs = remoteHermesPath.startsWith('/')
  const windowsAbs = /^[A-Za-z]:[\\/]/.test(remoteHermesPath) || remoteHermesPath.startsWith('\\\\')

  if (!posixAbs && !windowsAbs) {
    throw new Error(
      'Set an absolute Remote Hermes path on this SSH connection before opening it as an isolated Desktop.'
    )
  }

  const connectionId = String(connection.connectionId || connection.id || '').trim()

  if (!connectionId) {
    throw new Error('A Connections registry id is required so the isolated shell keeps the exact SSH row.')
  }

  const label = String(connection.label || '').trim()
  const name = slugFromLabel(label || host)
  const displayName = label.toLowerCase().startsWith('hermes ') ? label : `Hermes ${name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`
  const remoteProfile = String(connection.remoteProfile || 'default').trim() || 'default'
  const sshUser = String(connection.user || '').trim()
  const rawPort = Number(connection.port)
  const sshPort = Number.isInteger(rawPort) && rawPort > 0 && rawPort <= 65535 ? rawPort : 22
  const sshKeyPath = String(connection.keyPath || '').trim()

  const dialIdentity = JSON.stringify({
    host,
    keyPath: sshKeyPath,
    port: sshPort,
    remoteHermesPath,
    remoteProfile,
    user: sshUser
  })

  return {
    name,
    displayName,
    connectionId,
    sshHost: host,
    sshUser,
    sshPort,
    sshKeyPath,
    remoteHermesPath,
    remoteProfile,
    aumid: instanceAumid(name),
    dialIdentity
  }
}

export function assertIsolatedManifestMatches(
  existing: { connectionId?: string; dialIdentity?: string },
  spec: IsolatedInstanceSpec
): void {
  if (existing.connectionId && existing.connectionId !== spec.connectionId) {
    throw new Error(
      `Isolated Desktop instance belongs to connection ${JSON.stringify(existing.connectionId)}, not ${JSON.stringify(spec.connectionId)}.`
    )
  }

  if (existing.dialIdentity && existing.dialIdentity !== spec.dialIdentity) {
    throw new Error(
      `Isolated Desktop instance no longer matches the selected Connection ${JSON.stringify(spec.connectionId)}.`
    )
  }
}

export function parseInstanceDeepLink(url: string): InstanceDeepLink | null {
  const raw = String(url || '').trim()
  const prefix = 'hermes://instance/'

  if (!raw.startsWith(prefix)) {
    return null
  }

  const rest = raw.slice(prefix.length)
  const slash = rest.indexOf('/')
  const slug = slash === -1 ? rest : rest.slice(0, slash)
  const tail = slash === -1 ? '' : rest.slice(slash + 1)

  if (!NAME_RE.test(slug) || RESERVED_NAMES.has(slug)) {
    return null
  }

  return {
    instanceName: slug,
    remainder: tail ? `hermes://${tail}` : 'hermes://'
  }
}

export function isolatedDesktopLaunchArguments(userData: string, deepLink?: string): string[] {
  const args = [`--user-data-dir=${userData}`]

  if (deepLink) {
    args.push(deepLink)
  }

  return args
}

export function isolatedDesktopLaunchEnv(
  spec: IsolatedInstanceSpec,
  paths: { hermesHome: string; userData: string; runtimeRoot: string; cwd: string }
): Record<string, string> {
  return {
    HERMES_HOME: paths.hermesHome,
    HERMES_DESKTOP_USER_DATA_DIR: paths.userData,
    HERMES_DESKTOP_HERMES_ROOT: paths.runtimeRoot,
    HERMES_DESKTOP_APP_NAME: spec.displayName,
    HERMES_DESKTOP_CWD: paths.cwd,
    HERMES_DESKTOP_INSTANCE: spec.name,
    HERMES_DESKTOP_AUMID: spec.aumid,
    HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS: '1',
    HERMES_DESKTOP_SKIP_PROTOCOL_REGISTER: '1'
  }
}

export function shouldRegisterProtocolClient(env: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env): boolean {
  return env.HERMES_DESKTOP_SKIP_PROTOCOL_REGISTER !== '1'
}

export function shouldRegisterGlobalShortcuts(env: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env): boolean {
  return env.HERMES_DESKTOP_DISABLE_GLOBAL_SHORTCUTS !== '1'
}

export function resolveAppUserModelId(env: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env): string {
  const override = String(env.HERMES_DESKTOP_AUMID || '').trim()

  return override || DEFAULT_AUMID
}
