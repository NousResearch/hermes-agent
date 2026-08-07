import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'

const require = createRequire(import.meta.url)
const {
  verifySignature: verifyRuntimeUpdateSignature
} = require('electron-updater/out/windowsExecutableCodeSignatureVerifier.js')

function yamlScalar(value) {
  const trimmed = value.trim()

  if (
    trimmed.length >= 2 &&
    ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith('"') && trimmed.endsWith('"')))
  ) {
    return trimmed.slice(1, -1)
  }

  return trimmed
}

function requiredMatch(text, pattern, label) {
  const match = text.match(pattern)

  if (!match) {
    throw new Error(`latest.yml is missing ${label}`)
  }

  return yamlScalar(match[1])
}

function parseLatestYml(text) {
  const version = requiredMatch(text, /^version:\s*(.+)$/mu, 'top-level version')
  const installerPath = requiredMatch(text, /^path:\s*(.+)$/mu, 'top-level path')
  const topLevelSha512 = requiredMatch(text, /^sha512:\s*(.+)$/mu, 'top-level sha512')
  const fileUrl = requiredMatch(text, /^\s{2}-\s+url:\s*(.+)$/mu, 'files[0].url')
  const fileSha512 = requiredMatch(text, /^\s{4}sha512:\s*(.+)$/mu, 'files[0].sha512')
  const sizeText = requiredMatch(text, /^\s{4}size:\s*(\d+)\s*$/mu, 'files[0].size')
  const size = Number(sizeText)

  if (!Number.isSafeInteger(size) || size < 1) {
    throw new Error(`latest.yml contains an invalid installer size: ${sizeText}`)
  }

  if (installerPath !== fileUrl) {
    throw new Error(`latest.yml path ${installerPath} does not match files[0].url ${fileUrl}`)
  }

  if (topLevelSha512 !== fileSha512) {
    throw new Error('latest.yml top-level SHA-512 does not match files[0].sha512')
  }

  return { installerPath, sha512: topLevelSha512, size, version }
}

export function readPackagedPublisherNames(configPath) {
  const resolvedConfigPath = path.resolve(configPath)

  if (!fs.existsSync(resolvedConfigPath)) {
    throw new Error(`missing packaged updater configuration: ${resolvedConfigPath}`)
  }

  const lines = fs.readFileSync(resolvedConfigPath, 'utf8').split(/\r?\n/u)
  const publisherLine = lines.findIndex(line => /^publisherName\s*:/u.test(line))

  if (publisherLine < 0) {
    throw new Error(`packaged updater configuration does not contain a non-empty publisherName: ${resolvedConfigPath}`)
  }

  const inlineValue = lines[publisherLine].replace(/^publisherName\s*:\s*/u, '').trim()
  const publisherNames = []

  if (inlineValue) {
    if (!['[]', "''", '""', 'null', '~'].includes(inlineValue)) {
      publisherNames.push(yamlScalar(inlineValue))
    }
  } else {
    for (const line of lines.slice(publisherLine + 1)) {
      if (!line.trim()) {
        continue
      }

      const listItem = line.match(/^\s+-\s+(.+)$/u)

      if (!listItem) {
        break
      }

      const publisherName = yamlScalar(listItem[1])

      if (publisherName) {
        publisherNames.push(publisherName)
      }
    }
  }

  if (publisherNames.length === 0) {
    throw new Error(`packaged updater configuration does not contain a non-empty publisherName: ${resolvedConfigPath}`)
  }

  return publisherNames
}

function sha512Base64(filePath) {
  const hash = crypto.createHash('sha512')
  hash.update(fs.readFileSync(filePath))

  return hash.digest('base64')
}

function verifyAuthenticode(filePath, label) {
  if (process.platform !== 'win32') {
    throw new Error('--require-signature can only be verified on Windows')
  }

  const escapedPath = filePath.replaceAll("'", "''")
  const command = `(Get-AuthenticodeSignature -LiteralPath '${escapedPath}').Status.ToString()`
  const result = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], {
    encoding: 'utf8',
    windowsHide: true
  })

  if (result.error) {
    throw new Error(`Authenticode verification could not start: ${result.error.message}`)
  }

  const status = result.stdout.trim()

  if (result.status !== 0 || status !== 'Valid') {
    const detail = result.stderr.trim() || status || `exit ${result.status}`
    throw new Error(`${label} Authenticode signature is not valid: ${detail}`)
  }

  return status
}

export async function verifyUpdateRelease({
  expectedVersion,
  packagedExecutable,
  packagedUpdateConfig,
  releaseDir,
  requireSignature = false
}) {
  const resolvedReleaseDir = path.resolve(releaseDir)
  const latestPath = path.join(resolvedReleaseDir, 'latest.yml')

  if (!fs.existsSync(latestPath)) {
    throw new Error(`missing update metadata: ${latestPath}`)
  }

  const metadata = parseLatestYml(fs.readFileSync(latestPath, 'utf8'))

  if (metadata.version !== expectedVersion) {
    throw new Error(`latest.yml version ${metadata.version} does not match expected ${expectedVersion}`)
  }

  if (path.basename(metadata.installerPath) !== metadata.installerPath) {
    throw new Error(`latest.yml installer path must be a plain filename: ${metadata.installerPath}`)
  }

  const installerPath = path.join(resolvedReleaseDir, metadata.installerPath)
  const blockmapPath = `${installerPath}.blockmap`

  if (!fs.existsSync(installerPath)) {
    throw new Error(`missing installer: ${installerPath}`)
  }

  if (!fs.existsSync(blockmapPath)) {
    throw new Error(`missing blockmap: ${blockmapPath}`)
  }

  const actualSize = fs.statSync(installerPath).size

  if (actualSize !== metadata.size) {
    throw new Error(`installer size mismatch: latest.yml=${metadata.size}, actual=${actualSize}`)
  }

  const actualSha512 = sha512Base64(installerPath)

  if (actualSha512 !== metadata.sha512) {
    throw new Error('installer SHA-512 mismatch')
  }

  if (requireSignature && !packagedExecutable) {
    throw new Error('--require-signature requires a packaged executable')
  }

  if (requireSignature && !packagedUpdateConfig) {
    throw new Error('--require-signature requires a packaged updater configuration')
  }

  const publisherNames = packagedUpdateConfig ? readPackagedPublisherNames(packagedUpdateConfig) : []
  const signature = requireSignature ? verifyAuthenticode(installerPath, 'installer') : 'not-required'
  let runtimePublisherSignature = 'not-required'

  if (requireSignature) {
    const runtimeSignatureError = await verifyRuntimeUpdateSignature(publisherNames, installerPath, {
      info() {},
      warn() {}
    })

    if (runtimeSignatureError !== null) {
      throw new Error(`installer signature does not match packaged publisherName: ${publisherNames.join(' | ')}`)
    }

    runtimePublisherSignature = 'valid'
  }

  let packagedExecutableName = null
  let packagedExecutableSignature = 'not-requested'

  if (packagedExecutable) {
    const resolvedPackagedExecutable = path.resolve(packagedExecutable)

    if (!fs.existsSync(resolvedPackagedExecutable)) {
      throw new Error(`missing packaged executable: ${resolvedPackagedExecutable}`)
    }

    packagedExecutableName = path.basename(resolvedPackagedExecutable)
    packagedExecutableSignature = requireSignature
      ? verifyAuthenticode(resolvedPackagedExecutable, 'packaged executable')
      : 'not-required'
  }

  return {
    blockmap: path.basename(blockmapPath),
    installer: metadata.installerPath,
    installerBytes: actualSize,
    latest: path.basename(latestPath),
    packagedExecutable: packagedExecutableName,
    packagedExecutableSignature,
    packagedUpdateConfig: packagedUpdateConfig ? path.basename(packagedUpdateConfig) : null,
    publisherNames,
    runtimePublisherSignature,
    sha512: actualSha512,
    signature,
    version: metadata.version
  }
}

async function main() {
  const [releaseDir, expectedVersion, ...flags] = process.argv.slice(2)

  if (!releaseDir || !expectedVersion) {
    throw new Error(
      'usage: node scripts/verify-update-release.mjs <release-dir> <expected-version> [--require-signature] [--packaged-executable <path>] [--packaged-update-config <path>]'
    )
  }

  const packagedExecutableFlag = flags.indexOf('--packaged-executable')
  const packagedExecutable = packagedExecutableFlag >= 0 ? flags[packagedExecutableFlag + 1] : undefined

  if (packagedExecutableFlag >= 0 && !packagedExecutable) {
    throw new Error('--packaged-executable requires a path')
  }

  const packagedUpdateConfigFlag = flags.indexOf('--packaged-update-config')
  const packagedUpdateConfig = packagedUpdateConfigFlag >= 0 ? flags[packagedUpdateConfigFlag + 1] : undefined

  if (packagedUpdateConfigFlag >= 0 && !packagedUpdateConfig) {
    throw new Error('--packaged-update-config requires a path')
  }

  const result = await verifyUpdateRelease({
    expectedVersion,
    packagedExecutable,
    packagedUpdateConfig,
    releaseDir,
    requireSignature: flags.includes('--require-signature')
  })

  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(error => {
    console.error(`[verify-update-release] ${error instanceof Error ? error.message : String(error)}`)
    process.exit(1)
  })
}
