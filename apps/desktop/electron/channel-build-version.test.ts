import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'

import type { Configuration } from 'app-builder-lib'
import { afterEach, test } from 'vitest'

import type { ChannelBuildRequest, InstallStamp } from './install-stamp'
import { applyDesktopIdentity, type ProductIdentity } from './product-identity'

const require: NodeJS.Require = createRequire(import.meta.url)
const desktop: string = path.resolve(import.meta.dirname, '..')
const repo: string = path.resolve(desktop, '../..')

function request(sequence: number = 65536, token: string = 'ab12cd34ef56ab78'): ChannelBuildRequest {
  return {
    schema: 1,
    buildId: 'a'.repeat(32),
    channel: 'no-registry-needed',
    sequence,
    repository: 'fixture/project',
    commit: 'b'.repeat(40),
    sourceVersion: '1.2.3',
    version: `0.0.${sequence}`,
    windowsVersion: `0.${Math.floor(sequence / 65536)}.${sequence % 65536}.0`,
    identity: {
      token,
      displayName: 'Hermes no-registry-needed',
      appId: `com.nousresearch.hermes-channel-${token}`,
      appNamePascal: `HermesChannel${token}`,
      artifactNamePascal: `HermesChannel${token}`,
      cliName: 'hermes-no-registry-needed',
      windowsExecutableName: 'hermes-no-registry-needed',
      msixAppIdWithOrg: `NousResearch.HermesChannel${token}`
    },
    bundleEnv: { HERMES_GUEST_ONBOARDING: '1' },
    publicBase: 'https://builds.example.test'
  }
}

interface PackagingFacts {
  identity: ProductIdentity
  config: Configuration
}
interface StampPayload {
  runtime: { repoDir: string; commands: { hermes: string } }
}

function load(build: ChannelBuildRequest): PackagingFacts {
  process.env.HERMES_DESKTOP_VARIANT = 'bundled'
  process.env._HERMES_CHANNEL_REQUEST_JSON = JSON.stringify(build)

  for (const file of ['../product-identity.cjs', '../electron-builder.config.cjs']) {
    delete require.cache[require.resolve(file)]
  }

  return { identity: require('../product-identity.cjs'), config: require('../electron-builder.config.cjs') }
}

afterEach((): void => {
  for (const key of [
    '_HERMES_CHANNEL_REQUEST_JSON',
    'HERMES_DESKTOP_VARIANT',
    'HERMES_BUILD_COMMIT',
    'HERMES_PAYLOAD_TAG'
  ]) {
    delete process.env[key]
  }

  for (const file of ['../product-identity.cjs', '../electron-builder.config.cjs']) {
    delete require.cache[require.resolve(file)]
  }
})

test('channel packaging reuses admitted identity and rejects unsupported or unsafe identities', (): void => {
  const first: ChannelBuildRequest = request()
  const a: ReturnType<typeof load> = load(first)

  const b: ReturnType<typeof load> = load({
    ...first,
    ...request(65537),
    commit: 'c'.repeat(40),
    sourceVersion: '0.1.0'
  })

  for (const field of [
    'displayName',
    'appId',
    'appNamePascal',
    'cliName',
    'windowsExecutableName',
    'msixAppIdWithOrg'
  ] as const) {
    assert.equal(a.identity[field], first.identity[field])
    assert.equal(a.identity[field], b.identity[field])
  }

  const other: ReturnType<typeof load> = load(request(1, '1234567890abcdef'))
  assert.notEqual(other.identity.appId, a.identity.appId)
  assert.equal(a.config.extraMetadata?.version, first.version)
  assert.equal(a.config.extraMetadata?.shortVersionWindows, first.windowsVersion)
  assert.equal(a.config.msix?.identityName, first.identity.msixAppIdWithOrg)
  const appData: string = fs.mkdtempSync(path.join(os.tmpdir(), 'channel-runtime-identity-'))
  let userData: string = appData
  let displayName: string = ''

  try {
    assert.equal(
      applyDesktopIdentity(
        {
          getPath: (): string => appData,
          setPath: (_name: 'userData', value: string): void => {
            userData = value
          },
          setName: (value: string): void => {
            displayName = value
          }
        },
        a.identity
      ),
      first.identity.displayName
    )
    assert.equal(userData, path.join(appData, first.identity.appNamePascal))
    assert.equal(displayName, first.identity.displayName)
  } finally {
    fs.rmSync(appData, { recursive: true, force: true })
  }

  assert.deepEqual(a.config.mac?.publish, [
    {
      provider: 'generic',
      url: `${first.publicBase}/releases/channel-builds/${first.buildId}/darwin/`,
      channel: 'latest'
    }
  ])
  process.env.HERMES_DESKTOP_VARIANT = 'light'
  delete require.cache[require.resolve('../product-identity.cjs')]
  assert.throws((): void => {
    require('../product-identity.cjs')
  }, /bundled/)
  const invalid: ChannelBuildRequest = request()
  invalid.identity.cliName = 'con'
  assert.throws((): void => {
    load(invalid)
  }, /identity/)
  invalid.identity.cliName = first.identity.cliName
  invalid.identity.windowsExecutableName = 'trailing.'
  assert.throws((): void => {
    load(invalid)
  }, /identity/)
})

test('channel stamps verify the real checkout and retain source version and native ownership', async (): Promise<void> => {
  const {
    resolveStamp,
    buildStampPayload,
    writeDesktopStamp
  }: {
    resolveStamp: (options: { env: NodeJS.ProcessEnv; repoRoot: string }) => InstallStamp
    buildStampPayload: (
      stamp: InstallStamp,
      env: NodeJS.ProcessEnv,
      platform: string,
      payload: StampPayload
    ) => InstallStamp
    writeDesktopStamp: (directory: string, stamp: InstallStamp) => void
  } = await import('../scripts/write-build-stamp.mjs')

  const dir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'channel-stamp-'))

  try {
    execFileSync('git', ['init', '-q', dir])
    execFileSync(
      'git',
      [
        '-c',
        'user.name=Fixture',
        '-c',
        'user.email=fixture@example.test',
        '-c',
        'commit.gpgsign=false',
        'commit',
        '--allow-empty',
        '-qm',
        'fixture'
      ],
      { cwd: dir }
    )

    const build: ChannelBuildRequest = {
      ...request(),
      commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: dir, encoding: 'utf8' }).trim()
    }

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      HERMES_DESKTOP_VARIANT: 'bundled',
      _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(build),
      GITHUB_SHA: 'd'.repeat(40)
    }

    const provenance: InstallStamp = resolveStamp({ env, repoRoot: dir })
    const payload: StampPayload = { runtime: { repoDir: 'repo', commands: { hermes: 'bin/hermes' } } }
    const built: InstallStamp = buildStampPayload(provenance, env, 'darwin', payload)
    assert.equal(built.source, 'channel-build')
    assert.equal(built.updateMechanism, 'electron-updater')
    assert.equal(built.baseVersion, build.sourceVersion)
    assert.equal(built.tag, null)
    assert.equal(built.receiverProtocol, 1)
    assert.deepEqual(built.channelBuild, build)
    assert.ok(Object.isFrozen(built.channelBuild?.identity))
    fs.mkdirSync(path.join(dir, 'out/agent-payload/repo'), { recursive: true })
    writeDesktopStamp(path.join(dir, 'out'), built)
    assert.deepEqual(
      JSON.parse(fs.readFileSync(path.join(dir, 'out/install-stamp.json'), 'utf8')),
      JSON.parse(fs.readFileSync(path.join(dir, 'out/agent-payload/repo/install-stamp.json'), 'utf8'))
    )
    assert.throws((): void => {
      resolveStamp({ env: { ...env, _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(request()) }, repoRoot: dir })
    }, /checkout/)
    delete process.env._HERMES_CHANNEL_REQUEST_JSON
    process.env.HERMES_PAYLOAD_TAG = 'v0.0.1'
    delete require.cache[require.resolve('../product-identity.cjs')]
    const official: ProductIdentity = require('../product-identity.cjs')
    delete process.env.HERMES_PAYLOAD_TAG

    const receiver: ChannelBuildRequest = {
      ...build,
      channel: 'stable',
      sequence: 1,
      version: '0.0.1',
      windowsVersion: '0.0.1.0',
      releaseTag: 'v0.0.1',
      receiverCandidate: true,
      bundleEnv: {},
      publicBase: 'https://builds.example.test/ci-disposable/1/2',
      identity: { ...official, token: build.identity.token }
    }

    const receiverEnv: NodeJS.ProcessEnv = { ...env, _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(receiver) }

    const stable: InstallStamp = buildStampPayload(
      resolveStamp({ env: receiverEnv, repoRoot: dir }),
      receiverEnv,
      'darwin',
      payload
    )

    assert.equal(stable.channelBuild, undefined)
    assert.equal(stable.source, 'build')
    assert.equal(stable.tag, receiver.releaseTag)
    assert.equal(stable.displayVersion, receiver.version)
    assert.equal(stable.updateMechanism, 'electron-updater')
    const packaged: PackagingFacts = load(receiver)
    assert.deepEqual(packaged.config.mac?.publish, [
      { provider: 'generic', url: `${receiver.publicBase}/releases/darwin/stable/`, channel: 'stable' }
    ])

    const {
      verifyBundleStamp
    }: {
      verifyBundleStamp: (
        stamp: InstallStamp,
        options: {
          commit: string
          platform: string
          channelRequest: ChannelBuildRequest
        }
      ) => string
    } = await import('../../../tests/install/e2e-assets/bundle-smoke-metadata.mjs')

    assert.equal(
      verifyBundleStamp(stable, { commit: receiver.commit, platform: 'darwin', channelRequest: receiver }),
      receiver.version
    )
    assert.throws((): void => {
      verifyBundleStamp(built, { commit: receiver.commit, platform: 'darwin', channelRequest: receiver })
    })
    execFileSync(
      process.env.HERMES_PYTHON || 'python3',
      [
        '-c',
        `
import json, sys
from scripts.bundles.release_artifacts import stamp_matches
stamp, request = json.loads(sys.argv[1]), json.loads(sys.argv[2])
stamp_matches(stamp, '', request['commit'], channel_request=request)
stamp['channelBuild'] = request
try:
    stamp_matches(stamp, '', request['commit'], channel_request=request)
except ValueError:
    pass
else:
    raise AssertionError('receiver accepted preview updater ownership')
`,
        JSON.stringify(stable),
        JSON.stringify(receiver)
      ],
      { cwd: repo }
    )
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

// Four real manifest/recording pairs cross Node and Python process boundaries.
test('actual MSIX manifest writer consumes the channel quad across rollover instead of semver patch', (): void => {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'channel-manifest-'))
  const app: string = path.join(root, 'apps/desktop')
  interface ManifestFacts {
    version: string
    semver: string
    appId: string
    xml: string
  }
  interface RecordedMetadata {
    applicationId: string
    version: string
  }

  try {
    for (const file of [
      'apps/desktop/product-identity.cjs',
      'apps/desktop/electron-builder.config.cjs',
      'apps/desktop/package.json',
      'apps/desktop/update-feed.cjs',
      'apps/desktop/assets/msix-manifest.xml',
      'apps/desktop/scripts/before-build.mjs',
      'apps/desktop/scripts/mac-sign.mjs',
      'apps/desktop/scripts/payload-digests.mjs',
      'apps/desktop/scripts/utils.mjs',
      'scripts/msix-shared.mjs',
      'scripts/release-content-types.json',
      'scripts/build/python.mjs',
      'scripts/bundles/desktop_prepare.py',
      'hermes_cli/update_channel.py',
      'hermes_cli/release_channels.py',
      'hermes_cli/__init__.py',
      'hermes_constants.py'
    ]) {
      const destination: string = path.join(root, file)
      fs.mkdirSync(path.dirname(destination), { recursive: true })
      fs.copyFileSync(path.join(repo, file), destination)
    }

    // The version parser imports the PM package, and desktop_prepare imports
    // release helpers that import each other (versioning -> semver). Copy both
    // complete trees so a new intra-package import cannot break the fixture.
    for (const tree of ['pm', 'scripts/releases']) {
      fs.cpSync(path.join(repo, tree), path.join(root, tree), { recursive: true })
    }

    fs.symlinkSync(path.join(repo, 'node_modules'), path.join(root, 'node_modules'), 'junction')
    const assets: string = path.join(app, 'assets/appx')
    fs.mkdirSync(assets)

    for (const name of ['Square44x44Logo.png', 'Square150x150Logo.png', 'StoreLogo.png', 'Wide310x150Logo.png']) {
      fs.writeFileSync(path.join(assets, name), 'artwork is not inspected by manifest generation')
    }

    fs.mkdirSync(path.join(app, 'build/agent-payload'), { recursive: true })
    fs.writeFileSync(
      path.join(app, 'build/agent-payload/manifest.json'),
      JSON.stringify({ launchers: ['hermes-no-registry-needed', 'hermes-no-registry-needed-acp'] })
    )
    const facts: ManifestFacts[] = []

    for (const sequence of [65535, 65536, 65537, 0xffffffff]) {
      const build: ChannelBuildRequest = request(sequence)

      const script: string = `
        const { AppInfo } = require('./node_modules/app-builder-lib/dist/appInfo.js');
        const Msix = require('./node_modules/app-builder-lib/dist/targets/win/MsixTarget.js').default;
        const config = require('./apps/desktop/electron-builder.config.cjs');
        const pkg = require('./apps/desktop/package.json');
        const fs = require('node:fs');
        const path = require('node:path');
        const appDir = path.resolve('apps/desktop');
        (async () => {
          await config.beforeBuild();
          const metadata = { ...pkg, ...config.extraMetadata };
          const appInfo = new AppInfo({ config, metadata }, null, config.win);
          if (require('semver').valid(appInfo.version) !== appInfo.version) throw Error('Invalid packager semver');
          const packager = { appInfo, config, metadata, appDir, platformOptions: config.win,
            getResource: async (file) => path.resolve(appDir, file) };
          const target = { packager, options: config.msix,
            getCapabilities: Msix.prototype.getCapabilities, getExtensions: Msix.prototype.getExtensions };
          const output = path.resolve('manifest.xml');
          await Msix.prototype.writeManifest.call(target, output, 1, config.msix.publisher, []);
          console.log(JSON.stringify({version: appInfo.shortVersionWindows, semver: appInfo.version,
            appId: appInfo.id, xml: fs.readFileSync(output, 'utf8')}));
        })().catch(e => {console.error(e);process.exitCode=1});
      `

      const output: string = execFileSync(process.execPath, ['-e', script], {
        cwd: root,
        encoding: 'utf8',
        env: { ...process.env, HERMES_DESKTOP_VARIANT: 'bundled', _HERMES_CHANNEL_REQUEST_JSON: JSON.stringify(build) }
      })

      const row: ManifestFacts = JSON.parse(output.trim().split('\n').at(-1)!)
      assert.equal(row.version, build.windowsVersion)
      assert.equal(row.semver, build.version)
      assert.match(row.xml, new RegExp(`Version="${build.windowsVersion.replaceAll('.', '\\.')}"`))
      assert.equal((row.xml.match(/Category="windows.appExecutionAlias"/g) || []).length, 2)
      assert.match(row.xml, /HermesChannelab12cd34ef56ab78Cli1/)

      const recordScript: string = `
import json, pathlib, sys, zipfile, xml.etree.ElementTree as ET
sys.path.insert(0, sys.argv[1])
from scripts.bundles.release_artifacts import record, desktop_application
root = pathlib.Path(sys.argv[2])
request = json.loads(sys.argv[3])
package = root / 'Recorder-win-x64.msix'
with zipfile.ZipFile(package, 'w') as archive:
    archive.writestr('AppxManifest.xml', (root / 'manifest.xml').read_bytes())
    archive.writestr('app/resources/install-stamp.json', json.dumps({
        'source': 'channel-build', 'channelBuild': request, 'commit': request['commit'], 'tag': None}))
out = root / 'metadata.json'
record('windows', 'x64', root, '', request['commit'], out, channel_request=request)
manifest = ET.fromstring((root / 'manifest.xml').read_bytes())
desktop = desktop_application(manifest, request)
desktop.set('Id', 'WrongDesktop')
try:
    desktop_application(manifest, request)
except ValueError:
    pass
else:
    raise AssertionError('Recorder accepted the wrong desktop application ID')
print(out.read_text(encoding='utf-8'))
`

      const recorded: string = execFileSync(
        process.env.HERMES_PYTHON || 'python3',
        ['-c', recordScript, repo, root, JSON.stringify(build)],
        { encoding: 'utf8' }
      )

      const metadata: RecordedMetadata = JSON.parse(recorded)
      assert.equal(metadata.applicationId, build.identity.appNamePascal)
      assert.equal(metadata.version, build.windowsVersion)
      facts.push(row)
    }

    for (let index: number = 1; index < facts.length; index++) {
      assert.equal(facts[index].appId, facts[0].appId)
      assert.ok(facts[index - 1].version.localeCompare(facts[index].version, undefined, { numeric: true }) < 0)
    }

    assert.throws((): void => {
      load(request(0x100000000))
    }, /sequence/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
}, 20_000)
