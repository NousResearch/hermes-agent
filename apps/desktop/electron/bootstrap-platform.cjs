const fs = require('node:fs')

function isWslEnvironment(env = process.env, platform = process.platform, kernelRelease = null) {
  if (platform !== 'linux') return false
  if (env.WSL_DISTRO_NAME || env.WSL_INTEROP) return true

  try {
    const release = kernelRelease ?? fs.readFileSync('/proc/sys/kernel/osrelease', 'utf8')
    return /microsoft|wsl/i.test(release)
  } catch {
    return false
  }
}

function isWindowsBinaryPathInWsl(filePath, options = {}) {
  const isWsl = options.isWsl ?? isWslEnvironment(options.env, options.platform)
  if (!isWsl) return false

  const normalized = String(filePath || '')
    .replace(/\\/g, '/')
    .toLowerCase()

  return (
    normalized.endsWith('.exe') ||
    normalized.endsWith('.cmd') ||
    normalized.endsWith('.bat') ||
    normalized.endsWith('.ps1')
  )
}

function bundledRuntimeImportCheck(platform = process.platform) {
  return platform === 'win32' ? 'import fastapi, uvicorn, winpty' : 'import fastapi, uvicorn, ptyprocess'
}

module.exports = {
  bundledRuntimeImportCheck,
  isWindowsBinaryPathInWsl,
  isWslEnvironment
}
