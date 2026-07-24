import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const DESKTOP_FILE_ID = 'hermes.desktop'
const HERMES_SCHEME_MIME = 'x-scheme-handler/hermes'

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

function isLinuxUnpackedExecutable(execPath) {
  const parts = String(execPath || '')
    .replaceAll('\\', '/')
    .split('/')
    .filter(Boolean)

  return parts.length >= 2 && parts.at(-2) === 'linux-unpacked'
}

function quoteDesktopExec(execPath) {
  const value = String(execPath || '')

  if (!path.posix.isAbsolute(value) || /[\0\r\n]/.test(value)) {
    throw new Error('the Linux Desktop executable path must be an absolute single-line path')
  }

  const escaped = value.replace(/%/g, '%%').replace(/([\\`"$])/g, '\\$1')

  return `"${escaped}"`
}

function linuxDesktopEntry(execPath) {
  return [
    '[Desktop Entry]',
    'Type=Application',
    'Name=Hermes',
    `Exec=${quoteDesktopExec(execPath)} %U`,
    'Terminal=false',
    `MimeType=${HERMES_SCHEME_MIME};`,
    'Categories=Development;',
    ''
  ].join('\n')
}

function writeDesktopEntry(filePath, content) {
  try {
    const stat = fs.lstatSync(filePath)

    if (stat.isFile() && !stat.isSymbolicLink() && fs.readFileSync(filePath, 'utf8') === content) {
      return false
    }
  } catch (error: any) {
    if (error?.code !== 'ENOENT') {
      throw error
    }
  }

  const directory = path.dirname(filePath)
  const temporary = path.join(directory, `.${DESKTOP_FILE_ID}.${process.pid}.${Date.now()}.tmp`)
  fs.mkdirSync(directory, { recursive: true })

  try {
    fs.writeFileSync(temporary, content, { encoding: 'utf8', flag: 'wx', mode: 0o644 })
    fs.renameSync(temporary, filePath)
  } finally {
    fs.rmSync(temporary, { force: true })
  }

  return true
}

function runCommand(command, args) {
  return execFileSync(command, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    timeout: 5_000
  })
}

function ensureLinuxUnpackedDeepLink({ execPath, env = process.env, homeDir = os.homedir(), run = runCommand }) {
  const dataHome = String(env.XDG_DATA_HOME || '').trim() || path.join(homeDir, '.local', 'share')
  const applicationsDirectory = path.join(dataHome, 'applications')
  const desktopFile = path.join(applicationsDirectory, DESKTOP_FILE_ID)
  let changed = false

  try {
    changed = writeDesktopEntry(desktopFile, linuxDesktopEntry(execPath))

    if (changed) {
      try {
        run('update-desktop-database', [applicationsDirectory])
      } catch {
        // Optional cache refresh. xdg-mime below is the authoritative registration.
      }
    }

    let registered = String(run('xdg-mime', ['query', 'default', HERMES_SCHEME_MIME]) || '').trim()

    if (registered !== DESKTOP_FILE_ID) {
      run('xdg-mime', ['default', DESKTOP_FILE_ID, HERMES_SCHEME_MIME])
      registered = String(run('xdg-mime', ['query', 'default', HERMES_SCHEME_MIME]) || '').trim()
    }

    if (registered !== DESKTOP_FILE_ID) {
      throw new Error(`xdg-mime reported ${registered || 'no default handler'} after registration`)
    }

    return { ok: true, changed, desktopFile }
  } catch (error) {
    return {
      ok: false,
      changed,
      desktopFile,
      error: `could not register ${HERMES_SCHEME_MIME} with ${DESKTOP_FILE_ID}: ${errorMessage(error)}`
    }
  }
}

function registerDeepLinkProtocol({
  protocol,
  platform,
  execPath,
  defaultApp,
  argv,
  setAsDefaultProtocolClient,
  log,
  linux = {}
}) {
  try {
    if (platform === 'linux' && isLinuxUnpackedExecutable(execPath)) {
      const result = ensureLinuxUnpackedDeepLink({ execPath, ...linux })

      if (!result.ok) {
        log(`[deeplink] protocol registration failed: ${result.error}`)
      }

      return result.ok
    }

    const registered =
      defaultApp && argv.length >= 2
        ? setAsDefaultProtocolClient(protocol, execPath, [path.resolve(argv[1])])
        : setAsDefaultProtocolClient(protocol)

    if (!registered) {
      log('[deeplink] protocol registration failed: setAsDefaultProtocolClient returned false')
    }

    return registered
  } catch (error) {
    log(`[deeplink] protocol registration failed: ${errorMessage(error)}`)

    return false
  }
}

export {
  ensureLinuxUnpackedDeepLink,
  isLinuxUnpackedExecutable,
  registerDeepLinkProtocol
}
