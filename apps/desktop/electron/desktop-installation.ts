import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

const INSTALLATION_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function parseInstallationId(raw) {
  try {
    const value = JSON.parse(String(raw || ''))?.installationId

    return INSTALLATION_ID_RE.test(value) ? value.toLowerCase() : ''
  } catch {
    return ''
  }
}

function readInstallationId(filePath) {
  try {
    const stat = fs.lstatSync(filePath)

    if (!stat.isFile() || stat.isSymbolicLink()) {
      return ''
    }

    if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
      return ''
    }

    if (process.platform !== 'win32' && (stat.mode & 0o777) !== 0o600) {
      fs.chmodSync(filePath, 0o600)
    }

    return parseInstallationId(fs.readFileSync(filePath, 'utf8'))
  } catch {
    return ''
  }
}

function waitForRepair() {
  const buffer = new SharedArrayBuffer(4)
  Atomics.wait(new Int32Array(buffer), 0, 0, 25)
}

function loadOrCreateInstallationId(filePath, randomUUID = crypto.randomUUID) {
  const existing = readInstallationId(filePath)

  if (existing) {
    return existing
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  const installationId = randomUUID().toLowerCase()

  if (!INSTALLATION_ID_RE.test(installationId)) {
    throw new Error('Could not generate a valid desktop installation ID.')
  }

  const repairPath = `${filePath}.repair.lock`

  for (let attempt = 0; attempt < 40; attempt++) {
    let repairFd

    try {
      repairFd = fs.openSync(repairPath, 'wx', 0o600)
    } catch (error: any) {
      if (error?.code !== 'EEXIST') {
        throw error
      }

      const winner = readInstallationId(filePath)

      if (winner) {
        return winner
      }

      waitForRepair()

      continue
    }

    try {
      const winner = readInstallationId(filePath)

      if (winner) {
        return winner
      }

      try {
        const stat = fs.lstatSync(filePath)

        if (!stat.isFile() && !stat.isSymbolicLink()) {
          throw new Error('Desktop installation ID path is not a regular file.')
        }

        if (!stat.isSymbolicLink() && typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
          throw new Error('Desktop installation ID is owned by another user.')
        }

        fs.unlinkSync(filePath)
      } catch (error: any) {
        if (error?.code !== 'ENOENT') {
          throw error
        }
      }

      fs.writeFileSync(filePath, JSON.stringify({ installationId }), { encoding: 'utf8', flag: 'wx', mode: 0o600 })

      return installationId
    } finally {
      if (repairFd !== undefined) {
        fs.closeSync(repairFd)
      }

      try {
        fs.unlinkSync(repairPath)
      } catch {
        void 0
      }
    }
  }

  throw new Error('Could not repair the desktop installation ID.')
}

function sshOwnershipId(installationId, scope) {
  if (!INSTALLATION_ID_RE.test(String(installationId || ''))) {
    throw new Error('Desktop installation ID is invalid.')
  }

  return crypto
    .createHash('sha256')
    .update(`${installationId}\0${String(scope || '')}`)
    .digest('hex')
    .slice(0, 32)
}

export { INSTALLATION_ID_RE, loadOrCreateInstallationId, parseInstallationId, readInstallationId, sshOwnershipId }
