const { execFileSync } = require('node:child_process')

const WIN_DRIVE_RE = /^([A-Za-z]):[\\/](.*)$/
const WSL_MOUNT_RE = /^\/mnt\/([a-z])\/(.*)$/i
const WSL_UNC_RE = /^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$/i

let cachedDefaultDistro = null

function resolveDefaultWslDistro() {
  if (cachedDefaultDistro) {
    return cachedDefaultDistro
  }

  if (process.platform !== 'win32') {
    cachedDefaultDistro = 'Ubuntu'
    return cachedDefaultDistro
  }

  try {
    const output = execFileSync('wsl.exe', ['-l', '-q'], {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 2000
    })
    const distro = String(output)
      .split(/\r?\n/)
      .map(line => line.replace(/^\*?\s*/, '').trim())
      .find(Boolean)

    cachedDefaultDistro = distro || 'Ubuntu'
  } catch {
    cachedDefaultDistro = 'Ubuntu'
  }

  return cachedDefaultDistro
}

function windowsPathToWslMount(winPath) {
  const normalized = String(winPath || '').trim()
  const match = normalized.match(WIN_DRIVE_RE)

  if (!match) {
    return null
  }

  const drive = match[1].toLowerCase()
  const tail = match[2].replace(/\\/g, '/')

  return `/mnt/${drive}/${tail}`
}

function wslUncToPosix(uncPath) {
  const normalized = String(uncPath || '').trim().replace(/\//g, '\\')
  const match = normalized.match(WSL_UNC_RE)

  if (!match) {
    return null
  }

  const tail = match[2].replace(/\\/g, '/')
  return tail ? `/${tail}` : '/'
}

function wslPosixToWindowsAccessible(posixPath, distro = resolveDefaultWslDistro()) {
  const normalized = String(posixPath || '').trim().replace(/\\/g, '/')

  if (!normalized.startsWith('/')) {
    return normalized
  }

  const mountMatch = normalized.match(WSL_MOUNT_RE)

  if (mountMatch) {
    const letter = mountMatch[1].toUpperCase()
    const tail = mountMatch[2].replace(/\//g, '\\')
    return `${letter}:\\${tail}`
  }

  const relative = normalized.replace(/^\/+/, '').replace(/\//g, '\\')
  return `\\\\wsl.localhost\\${distro}\\${relative}`
}

function resolvePickerDefaultPath(defaultPath, distro = resolveDefaultWslDistro()) {
  if (!defaultPath) {
    return undefined
  }

  const value = String(defaultPath).trim()

  if (value.startsWith('/') && !value.match(WIN_DRIVE_RE)) {
    return wslPosixToWindowsAccessible(value, distro)
  }

  return defaultPath
}

function resolvePickerResultForRemoteBackend(selectedPath) {
  if (!selectedPath) {
    return selectedPath
  }

  const value = String(selectedPath).trim()
  const unc = wslUncToPosix(value)

  if (unc) {
    return unc
  }

  const mount = windowsPathToWslMount(value)

  if (mount) {
    return mount
  }

  return value
}

function resolveLocalReadPath(dirPath, distro = resolveDefaultWslDistro()) {
  const value = String(dirPath || '').trim()

  if (process.platform === 'win32' && value.startsWith('/') && !value.match(WIN_DRIVE_RE)) {
    return wslPosixToWindowsAccessible(value, distro)
  }

  return value
}

module.exports = {
  resolveDefaultWslDistro,
  resolveLocalReadPath,
  resolvePickerDefaultPath,
  resolvePickerResultForRemoteBackend,
  windowsPathToWslMount,
  wslPosixToWindowsAccessible,
  wslUncToPosix
}