import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptPath = fileURLToPath(import.meta.url)
const desktopRoot = path.resolve(path.dirname(scriptPath), '..')
const packageJson = JSON.parse(fs.readFileSync(path.join(desktopRoot, 'package.json'), 'utf8'))

function assertE2EVersion(version) {
  if (!/^\d+\.\d+\.\d+$/.test(version)) {
    throw new Error(`E2E version must be a numeric semver (received ${JSON.stringify(version)}).`)
  }
}

function normalizeLoopbackFeed(feedUrl) {
  let parsed

  try {
    parsed = new URL(feedUrl)
  } catch {
    throw new Error('E2E update feed must be a loopback HTTP URL.')
  }

  const loopbackHosts = new Set(['127.0.0.1', '[::1]', 'localhost'])

  if (parsed.protocol !== 'http:' || !loopbackHosts.has(parsed.hostname) || parsed.username || parsed.password) {
    throw new Error('E2E update feed must be a loopback HTTP URL.')
  }

  parsed.search = ''
  parsed.hash = ''

  return parsed.href
}

export function makeE2EBuilderConfig(version, feedUrl, { failHealth = false } = {}) {
  assertE2EVersion(version)
  const normalizedFeed = normalizeLoopbackFeed(feedUrl)
  const base = packageJson.build

  return {
    ...base,
    appId: 'com.nousresearch.hermes.update-e2e',
    productName: 'Hermes Update E2E',
    executableName: 'HermesUpdateE2E',
    protocols: [],
    artifactName: 'Hermes-Update-E2E-${version}-${arch}.${ext}',
    directories: {
      ...base.directories,
      output: path.join('release', 'update-e2e', version)
    },
    extraMetadata: {
      name: 'hermes-update-e2e',
      productName: 'Hermes Update E2E',
      version,
      ...(failHealth
        ? {
            hermesUpdateE2EFailHealth: true,
            hermesUpdateHealthTimeoutMs: 3_000
          }
        : {})
    },
    publish: [
      {
        provider: 'generic',
        url: normalizedFeed
      }
    ],
    win: {
      ...base.win,
      target: ['nsis']
    },
    nsis: {
      ...base.nsis,
      createDesktopShortcut: false,
      createStartMenuShortcut: false,
      shortcutName: 'Hermes Update E2E',
      uninstallDisplayName: 'Hermes Update E2E'
    }
  }
}

function runNpm(args) {
  const command = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const result = spawnSync(command, args, {
    cwd: desktopRoot,
    env: process.env,
    shell: process.platform === 'win32',
    stdio: 'inherit'
  })

  if (result.error) {
    throw result.error
  }

  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${result.status}.`)
  }
}

function main() {
  const version = process.argv[2]
  const feedUrl = process.argv[3] ?? 'http://127.0.0.1:47892/'
  const failHealth = process.argv.includes('--fail-health')
  const config = makeE2EBuilderConfig(version, feedUrl, { failHealth })
  const configPath = path.join(desktopRoot, 'build', `update-e2e-${version}.json`)

  fs.mkdirSync(path.dirname(configPath), { recursive: true })
  fs.writeFileSync(configPath, `${JSON.stringify(config, null, 2)}\n`, 'utf8')

  runNpm(['run', 'build'])
  runNpm(['run', 'builder', '--', '--config', configPath, '--win', 'nsis', '--publish', 'never'])

  console.log(`[update-e2e] built ${version} -> ${path.join(desktopRoot, config.directories.output)}`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  try {
    main()
  } catch (error) {
    console.error(`[update-e2e] ${error instanceof Error ? error.message : String(error)}`)
    process.exitCode = 1
  }
}
