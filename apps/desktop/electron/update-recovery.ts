import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const DATA = [
  'config.yaml',
  '.env',
  'SOUL.md',
  'USER.md',
  'MEMORY.md',
  'memory',
  'memories',
  'skills',
  'profiles',
  'state.db',
  'state.db-wal',
  'state.db-shm'
]

export interface RecoveryRecord {
  schema: 1
  createdAt: string
  version: string
  home: string
  installRoot: string
  snapshot: string
  executable: string
  status: 'prepared' | 'healthy' | 'restored'
  files: string[]
}

export function readRecovery(root: string): RecoveryRecord | null {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(root, 'latest.json'), 'utf8')) as RecoveryRecord

    if (
      data.schema !== 1 ||
      path.dirname(data.snapshot) !== path.resolve(root) ||
      !fs.existsSync(path.join(data.snapshot, 'app', data.executable))
    ) {
      return null
    }

    return data
  } catch {
    return null
  }
}

/** Called only AFTER all owned backends have stopped, so SQLite/WAL and memory are quiescent. */
export async function prepareRecovery({
  home,
  installRoot,
  root,
  version,
  executable
}: {
  home: string
  installRoot: string
  root: string
  version: string
  executable: string
}): Promise<RecoveryRecord> {
  if (path.resolve(root).startsWith(path.resolve(installRoot) + path.sep)) {
    throw new Error('Kopia musi być poza katalogiem aplikacji.')
  }

  const snapshot = path.join(path.resolve(root), `snapshot-${Date.now()}`)

  const record: RecoveryRecord = {
    schema: 1,
    createdAt: new Date().toISOString(),
    version,
    home: path.resolve(home),
    installRoot: path.resolve(installRoot),
    snapshot,
    executable: path.basename(executable),
    status: 'prepared',
    files: []
  }

  await fs.promises.mkdir(snapshot, { recursive: true })

  const filter = async (source: string) => {
    if ((await fs.promises.lstat(source)).isSymbolicLink()) {
      throw new Error('Kopia przerwana: katalog zawiera dowiązanie. Dane nie zostały zmienione.')
    }

    return true
  }

  await fs.promises.cp(installRoot, path.join(snapshot, 'app'), { recursive: true, filter })

  for (const name of DATA) {
    if (!fs.existsSync(path.join(home, name))) {
      continue
    }

    await fs.promises.cp(path.join(home, name), path.join(snapshot, 'home', name), { recursive: true, filter })
    record.files.push(name)
  }

  const runtimeChoice = path.join(root, '..', 'runtime-collaborator.json')

  if (fs.existsSync(runtimeChoice)) {
    await fs.promises.copyFile(runtimeChoice, path.join(snapshot, 'runtime-collaborator.json'))
  }

  await fs.promises.writeFile(path.join(root, 'latest.json.tmp'), JSON.stringify(record), { mode: 0o600 })
  await fs.promises.rename(path.join(root, 'latest.json.tmp'), path.join(root, 'latest.json'))

  return record
}

export function markRecoveryHealthy(root: string) {
  const record = readRecovery(root)

  if (!record || record.status !== 'prepared') {
    return
  }

  fs.writeFileSync(path.join(root, 'latest.json'), JSON.stringify({ ...record, status: 'healthy' }))
}

/** Also executed by the detached worker after Electron exits. Keep this function self-contained. */
export function restoreRecoveryFiles(record: RecoveryRecord, root: string) {
  const target = path.resolve(record.installRoot)
  const snapshot = path.resolve(record.snapshot)
  const home = path.resolve(record.home)

  if (
    path.dirname(snapshot) !== path.resolve(root) ||
    target === path.parse(target).root ||
    home === path.parse(home).root ||
    path.basename(record.executable) !== record.executable
  ) {
    throw new Error('Nieprawidłowe ścieżki przywracania.')
  }

  for (const name of record.files) {
    if (!DATA.includes(name) || path.basename(name) !== name || name === '..') {
      throw new Error('Nieprawidłowy plik kopii.')
    }
  }

  if (!fs.existsSync(path.join(snapshot, 'app', record.executable))) {
    throw new Error('Niepełna kopia aplikacji.')
  }

  const suffix = Date.now()
  const preserved = path.join(path.dirname(home), `czesiek-before-restore-${suffix}`)
  const previousApp = `${target}.before-restore-${suffix}`
  const stagedApp = `${target}.restore-stage-${suffix}`
  fs.mkdirSync(preserved, { recursive: true })
  const choiceTarget = path.join(root, '..', 'runtime-collaborator.json')
  const previousChoice = fs.existsSync(choiceTarget) ? fs.readFileSync(choiceTarget) : null
  let choiceChanged = false
  // Finish copying before touching the current app. Renames stay on its volume.
  fs.cpSync(path.join(snapshot, 'app'), stagedApp, { recursive: true })
  fs.renameSync(target, previousApp)
  const changed: string[] = []

  try {
    fs.renameSync(stagedApp, target)

    for (const name of DATA) {
      const current = path.join(home, name)
      const old = path.join(preserved, 'home', name)

      if (fs.existsSync(current)) {
        fs.mkdirSync(path.dirname(old), { recursive: true })
        fs.renameSync(current, old)
      }

      changed.push(name)

      if (record.files.includes(name)) {
        fs.cpSync(path.join(snapshot, 'home', name), current, { recursive: true })
      }
    }

    const choice = path.join(snapshot, 'runtime-collaborator.json')

    if (fs.existsSync(choice)) {
      choiceChanged = true
      fs.copyFileSync(choice, choiceTarget)
    }

    fs.writeFileSync(path.join(root, 'latest.json'), JSON.stringify({ ...record, status: 'restored' }))
  } catch (error) {
    if (choiceChanged) {
      if (previousChoice) {
        fs.writeFileSync(choiceTarget, previousChoice)
      } else {
        fs.unlinkSync(choiceTarget)
      }
    }

    // Move partial restored data aside and put the pre-restore state back.
    for (const name of changed.reverse()) {
      const current = path.join(home, name)

      if (fs.existsSync(current)) {
        fs.renameSync(current, path.join(preserved, `partial-${name}`))
      }

      const old = path.join(preserved, 'home', name)

      if (fs.existsSync(old)) {
        fs.renameSync(old, current)
      }
    }

    if (fs.existsSync(target)) {
      fs.renameSync(target, `${target}.partial-restore-${suffix}`)
    }

    fs.renameSync(previousApp, target)
    throw new Error(`Przywracanie przerwane; zachowano poprzedni stan. ${String(error)}`)
  }
}

export async function launchRecovery(root: string, parentPid: number, expectedInstallRoot: string) {
  const record = readRecovery(root)

  if (!record || path.resolve(record.installRoot) !== path.resolve(expectedInstallRoot)) {
    throw new Error('Brak zgodnej kopii do przywrócenia.')
  }

  const node = path.join(root, 'recovery-node.exe')
  fs.copyFileSync(path.join(record.snapshot, 'app', 'resources', 'runtime', 'node', 'node.exe'), node)
  const script = path.join(root, 'restore.mjs')
  fs.writeFileSync(script, fs.readFileSync(path.join(import.meta.dirname, 'recovery-worker.mjs')))
  await new Promise<void>((resolve, reject) => {
    const child = spawn(node, [script, root, String(parentPid), expectedInstallRoot], {
      detached: true,
      stdio: 'ignore',
      windowsHide: true,
      cwd: root
    })

    child.once('error', reject)
    child.once('spawn', () => {
      child.unref()
      resolve()
    })
  })
}
