import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { readRecovery, restoreRecoveryFiles } from './update-recovery'

const [root, parent, expectedInstallRoot] = process.argv.slice(2)

if (!root || !expectedInstallRoot || !parent || !Number.isSafeInteger(Number(parent)) || Number(parent) <= 0) {
  throw new Error('Nieprawidłowe argumenty przywracania.')
}

try {
  const record = readRecovery(root)

  if (!record || path.resolve(record.installRoot) !== path.resolve(expectedInstallRoot)) {
    throw new Error('Nieprawidłowa kopia aplikacji.')
  }

  let exited = false

  for (let i = 0; i < 120; i++) {
    try {
      process.kill(Number(parent), 0)
    } catch {
      exited = true

      break
    }

    await new Promise(resolve => setTimeout(resolve, 500))
  }

  if (!exited) {
    throw new Error('Aplikacja nie została zamknięta; nie zmieniono plików.')
  }

  restoreRecoveryFiles(record, root)

  const child = spawn(path.join(record.installRoot, record.executable), [], {
    detached: true,
    stdio: 'ignore',
    windowsHide: true
  })

  child.on('error', error => fs.writeFileSync(path.join(root, 'restore-error.txt'), String(error)))
  child.unref()
} catch (error) {
  fs.writeFileSync(path.join(root, 'restore-error.txt'), String(error))
  process.exitCode = 1
}
