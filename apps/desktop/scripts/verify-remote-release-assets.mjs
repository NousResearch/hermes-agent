import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SHA256_DIGEST = /^sha256:([a-f0-9]{64})$/i

function sha256File(filePath) {
  const hash = crypto.createHash('sha256')
  const descriptor = fs.openSync(filePath, 'r')
  const buffer = Buffer.allocUnsafe(1024 * 1024)

  try {
    let bytesRead
    do {
      bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, null)
      if (bytesRead > 0) hash.update(buffer.subarray(0, bytesRead))
    } while (bytesRead > 0)
  } finally {
    fs.closeSync(descriptor)
  }

  return hash.digest('hex')
}

export function verifyRemoteReleaseAssets({ excludeNames = [], remoteRelease, stagingDir }) {
  if (!remoteRelease || !Array.isArray(remoteRelease.assets)) {
    throw new Error('Remote GitHub release JSON does not contain an assets array.')
  }

  const excluded = new Set(excludeNames)
  const localAssets = fs
    .readdirSync(stagingDir, { withFileTypes: true })
    .filter(entry => entry.isFile() && !excluded.has(entry.name))
    .map(entry => {
      const filePath = path.join(stagingDir, entry.name)
      const stat = fs.statSync(filePath)

      return { digest: sha256File(filePath), name: entry.name, size: stat.size }
    })
    .sort((left, right) => left.name.localeCompare(right.name))
  const remoteAssets = remoteRelease.assets
    .filter(asset => !excluded.has(asset?.name))
    .sort((left, right) => String(left?.name).localeCompare(String(right?.name)))

  if (remoteAssets.length !== localAssets.length) {
    throw new Error(
      `Remote release asset count ${remoteAssets.length} does not match verified staging count ${localAssets.length}.`
    )
  }

  for (const local of localAssets) {
    const matches = remoteAssets.filter(remote => remote?.name === local.name)
    if (matches.length !== 1) {
      throw new Error(`Remote release must contain exactly one ${local.name} asset.`)
    }

    const remote = matches[0]
    if (Number(remote.size) !== local.size) {
      throw new Error(`Remote size mismatch for ${local.name}.`)
    }

    const digestMatch = typeof remote.digest === 'string' ? SHA256_DIGEST.exec(remote.digest) : null
    if (!digestMatch) {
      throw new Error(`Remote asset ${local.name} is missing GitHub SHA-256 digest.`)
    }

    const remoteDigest = Buffer.from(digestMatch[1].toLowerCase(), 'hex')
    const localDigest = Buffer.from(local.digest, 'hex')
    if (!crypto.timingSafeEqual(remoteDigest, localDigest)) {
      throw new Error(`SHA-256 digest mismatch for ${local.name}.`)
    }
  }

  return { assetCount: localAssets.length, names: localAssets.map(asset => asset.name) }
}

function parseArguments(argv) {
  const options = { excludeNames: [] }

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    const value = argv[index + 1]

    if (argument === '--exclude') {
      if (!value) throw new Error('--exclude requires an asset name.')
      options.excludeNames.push(value)
      index += 1
    } else if (argument === '--remote-json') {
      if (!value) throw new Error('--remote-json requires a file path.')
      options.remoteJsonPath = value
      index += 1
    } else if (argument === '--staging') {
      if (!value) throw new Error('--staging requires a directory path.')
      options.stagingDir = value
      index += 1
    } else {
      throw new Error(`Unknown argument: ${argument}`)
    }
  }

  if (!options.remoteJsonPath || !options.stagingDir) {
    throw new Error('Usage: verify-remote-release-assets --staging <dir> --remote-json <file> [--exclude <name>]')
  }

  return options
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  try {
    const { excludeNames, remoteJsonPath, stagingDir } = parseArguments(process.argv.slice(2))
    const remoteRelease = JSON.parse(fs.readFileSync(remoteJsonPath, 'utf8'))
    const result = verifyRemoteReleaseAssets({ excludeNames, remoteRelease, stagingDir })
    process.stdout.write(`${JSON.stringify(result)}\n`)
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  }
}
