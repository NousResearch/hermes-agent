import assert from 'node:assert/strict'
import crypto from 'node:crypto'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptPath = fileURLToPath(import.meta.url)
const desktopRoot = path.resolve(path.dirname(scriptPath), '..')
const packageJson = JSON.parse(fs.readFileSync(path.join(desktopRoot, 'package.json'), 'utf8'))

function readOption(name, fallback) {
  const index = process.argv.indexOf(name)
  if (index === -1) return fallback

  const value = process.argv[index + 1]
  if (!value || value.startsWith('--')) {
    throw new Error(`${name} requires a value.`)
  }

  return value
}

function sha512(filePath) {
  const hash = crypto.createHash('sha512')
  hash.update(fs.readFileSync(filePath))
  return hash.digest('hex')
}

function assertDisposableInstallPath(installPath) {
  const temporaryRoot = path.resolve(process.env.RUNNER_TEMP || os.tmpdir())
  const relative = path.relative(temporaryRoot, installPath)
  const escapesTemporaryRoot =
    relative === '' || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)

  assert.equal(
    escapesTemporaryRoot,
    false,
    `Disposable install directory must be a child of the temporary directory: ${temporaryRoot}`
  )
}

function snapshotFiles(paths) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-retention-snapshot-'))
  const snapshots = []

  for (const filePath of paths) {
    if (!fs.existsSync(filePath)) continue

    const snapshotPath = path.join(root, path.basename(filePath))
    fs.copyFileSync(filePath, snapshotPath)
    snapshots.push({ filePath, snapshotPath })
  }

  return { root, snapshots }
}

function restoreFiles(paths, snapshot) {
  for (const filePath of paths) {
    fs.rmSync(filePath, { force: true, recursive: true })
  }

  for (const { filePath, snapshotPath } of snapshot.snapshots) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true })
    fs.copyFileSync(snapshotPath, filePath)
  }

  fs.rmSync(snapshot.root, { force: true, recursive: true })
}

export function verifyInstallerRetention({ installer, version, appPackageName, installDirectory }) {
  if (process.platform !== 'win32') {
    throw new Error('NSIS installer retention can only be exercised on Windows.')
  }
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(version)) {
    throw new Error(`Invalid installer version: ${JSON.stringify(version)}.`)
  }
  if (!/^[a-z0-9][a-z0-9._-]*$/i.test(appPackageName)) {
    throw new Error(`Invalid app package name: ${JSON.stringify(appPackageName)}.`)
  }
  if (!process.env.GITHUB_ACTIONS && appPackageName === packageJson.name) {
    throw new Error('Refusing to exercise the production Hermes installer outside GitHub Actions.')
  }

  const installerPath = path.resolve(installer)
  const installPath = path.resolve(installDirectory)
  assertDisposableInstallPath(installPath)
  const localAppData = process.env.LOCALAPPDATA
  assert.ok(localAppData, 'LOCALAPPDATA must be defined for an NSIS retention test.')
  assert.ok(fs.statSync(installerPath).isFile(), `Installer does not exist: ${installerPath}`)
  assert.ok(!fs.existsSync(installPath), `Disposable install directory already exists: ${installPath}`)

  const retentionDirectory = path.join(localAppData, `${appPackageName}-rollback`, 'installers')
  const retainedInstaller = path.join(retentionDirectory, `${version}.exe`)
  const retainedTemp = path.join(retentionDirectory, `${version}.tmp`)
  const retainedBackup = path.join(retentionDirectory, `${version}.bak`)
  const retentionPaths = [retainedInstaller, retainedTemp, retainedBackup]
  const snapshot = snapshotFiles(retentionPaths)

  try {
    fs.mkdirSync(retentionDirectory, { recursive: true })
    for (const filePath of retentionPaths) fs.rmSync(filePath, { force: true, recursive: true })

    const previousInstaller = Buffer.from('pre-existing-retained-installer\n', 'utf8')
    fs.writeFileSync(retainedInstaller, previousInstaller)

    const result = spawnSync(installerPath, ['/S', `/D=${installPath}`], {
      encoding: 'utf8',
      windowsHide: true
    })
    if (result.error) throw result.error
    if (result.status !== 0) {
      throw new Error(
        `NSIS installer exited ${result.status}. stdout=${JSON.stringify(result.stdout)} stderr=${JSON.stringify(result.stderr)}`
      )
    }

    assert.ok(fs.statSync(retainedInstaller).isFile(), 'Installer did not create the retained rollback copy.')
    assert.notDeepEqual(
      fs.readFileSync(retainedInstaller),
      previousInstaller,
      'Installer left the pre-existing retained copy unchanged.'
    )
    assert.equal(
      sha512(retainedInstaller),
      sha512(installerPath),
      'Retained installer does not match the executed NSIS installer.'
    )
    assert.equal(fs.existsSync(retainedTemp), false, 'Installer left a temporary retention file behind.')
    assert.equal(
      fs.existsSync(retainedBackup),
      false,
      'Installer left a backup retention file behind after successful promotion.'
    )

    return {
      installer: installerPath,
      retainedInstaller,
      sha512: sha512(retainedInstaller),
      version
    }
  } finally {
    try {
      fs.rmSync(installPath, { force: true, recursive: true })
    } finally {
      restoreFiles(retentionPaths, snapshot)
    }
  }
}

function main() {
  const installer = readOption('--installer')
  const version = readOption('--version')
  const appPackageName = readOption('--app-package-name', packageJson.name)
  const installDirectory = readOption('--install-dir')

  if (!installer || !version || !installDirectory) {
    throw new Error(
      'Usage: node verify-installer-retention.mjs --installer <path> --version <semver> --install-dir <disposable-path> [--app-package-name <name>]'
    )
  }

  console.log(
    JSON.stringify(verifyInstallerRetention({ installer, version, appPackageName, installDirectory }), null, 2)
  )
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main()
  } catch (error) {
    console.error(`[installer-retention] ${error instanceof Error ? error.message : String(error)}`)
    process.exitCode = 1
  }
}
