import { execFile, execFileSync, type SpawnOptions } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'
import { promisify } from 'node:util'

import { expect, it, vi } from 'vitest'

import * as updaterProcess from '../updater-process'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import { readSourceUpdate, type SourceUpdate } from './checkout-source'

const execute: typeof execFile.__promisify__ = promisify(execFile)
const repository: string = path.resolve(import.meta.dirname, '../../../..')
const python: string = process.env.HERMES_PYTHON || 'python3'

function buildId(channel: 'stable' | 'canary'): string {
  return (channel === 'stable' ? 'a' : 'b').repeat(32)
}

function manifestKey(channel: 'stable' | 'canary'): string {
  return `releases/channel-builds/${buildId(channel)}/build.json`
}

const crypto = await import('node:crypto')

interface FixtureIdentity {
  token: string
  displayName: string
  appNamePascal: string
  artifactNamePascal: string
  appId: string
  msixAppIdWithOrg: string
  cliName: string
  windowsExecutableName: string
}

interface FixtureHead {
  buildId: string
  sequence: number
  manifestKey: string
  sha256: string
}

interface FixtureRecord {
  schema: number
  name: string
  repository: string
  policy: 'stable-release' | 'canary-release'
  state: 'active'
  revision: number
  nextSequence: number
  identity: FixtureIdentity
  head: FixtureHead
}

interface FixtureArtifact {
  key: string
  sha256: string
  size: number
  format: string
}

interface FixturePackage {
  platform: 'darwin'
  arch: 'arm64'
  variant: 'bundled'
  identity: string
  version: string
  teamId: string
  artifact: FixtureArtifact
  feed: { channel: 'stable'; key: string }
}

interface FixtureRequest {
  schema: number
  buildId: string
  channel: 'stable' | 'canary'
  repository: string
  commit: string
  sourceVersion: string
  releaseTag: string
  sequence: number
  version: string
  windowsVersion: string
  identity: FixtureIdentity
  bundleEnv: Record<string, string>
  publicBase: string
}

interface FixtureManifest {
  schema: number
  request: FixtureRequest
  packages: FixturePackage[]
}

/** Python canonical_json: sort_keys, compact separators, trailing newline. */
function canonicalJson(value: FixtureManifest): string {
  // SAFETY: FixtureManifest is the exact wire shape; JSON.stringify emits
  // member order as written, matching the fixture builder's field order.
  return JSON.stringify(value)
}

/** R2 channel record per hermes_cli.release_channels.validate_record. */
function channelRecord(channel: 'stable' | 'canary', sequence: number): FixtureRecord {
  return {
    schema: 1,
    name: channel,
    repository: 'NousResearch/hermes-agent',
    policy: channel === 'stable' ? 'stable-release' : 'canary-release',
    state: 'active',
    revision: 1,
    nextSequence: sequence + 1,
    identity: {
      token: 'b'.repeat(16),
      displayName: channel === 'stable' ? 'Hermes Stable' : 'Hermes Canary',
      appNamePascal: 'Hermes',
      artifactNamePascal: 'Hermes',
      appId: 'chat.nous.hermes',
      msixAppIdWithOrg: 'NousResearch.Hermes',
      cliName: 'hermes',
      windowsExecutableName: 'hermes'
    },
    head: {
      buildId: 'a'.repeat(32),
      sequence,
      manifestKey: `releases/channel-builds/${'a'.repeat(32)}/build.json`,
      sha256: 'c'.repeat(64)
    }
  }
}

/** Build manifest per hermes_cli.release_channels.validate_manifest. */
function buildManifest(
  channel: 'stable' | 'canary',
  sha: string,
  tag: string,
  id: string = buildId(channel)
): FixtureManifest {
  const identity: FixtureIdentity = {
    token: 'b'.repeat(16),
    displayName: channel === 'stable' ? 'Hermes Stable' : 'Hermes Canary',
    appNamePascal: 'Hermes',
    artifactNamePascal: 'Hermes',
    appId: 'chat.nous.hermes',
    msixAppIdWithOrg: 'NousResearch.Hermes',
    cliName: 'hermes',
    windowsExecutableName: 'hermes'
  }

  const sequence: number = channel === 'stable' ? 1 : 2

  return {
    schema: 1,
    request: {
      schema: 1,
      buildId: id,
      channel,
      repository: 'NousResearch/hermes-agent',
      commit: sha,
      sourceVersion: tag.replace(/^v/, '').split('+')[0],
      releaseTag: tag,
      sequence,
      version: tag.replace(/^v/, ''),
      windowsVersion: `0.0.${sequence}.0`,
      identity,
      bundleEnv: {},
      publicBase: 'https://hermes-assets.nousresearch.com'
    },
    packages: [
      {
        platform: 'darwin',
        arch: 'arm64',
        variant: 'bundled',
        identity: 'chat.nous.hermes',
        version: tag.replace(/^v/, ''),
        teamId: 'TESTTEAM12',
        artifact: {
          key: `releases/channel-builds/${id}/darwin-arm64.zip`,
          sha256: 'd'.repeat(64),
          size: 1,
          format: 'zip'
        },
        feed: { channel: 'stable', key: `releases/channel-builds/${id}/darwin-arm64.xml` }
      }
    ]
  }
}

it('carries each install channel from Python publication checks into the source handoff', async (): Promise<void> => {
  const temporary: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-channel-'))
  const origin: string = path.join(temporary, 'origin')
  const root: string = path.join(temporary, 'checkout')
  const home: string = path.join(temporary, 'profile')
  const requests: string[] = []
  const responses: Map<string, string | object> = new Map<string, string | object>()

  const server: http.Server = http.createServer(
    (request: http.IncomingMessage, response: http.ServerResponse): void => {
      const url: string = request.url ?? ''
      requests.push(url)
      const body: string | object | undefined = responses.get(url)
      response.statusCode = body === undefined ? 404 : 200
      response.end(typeof body === 'string' ? body : JSON.stringify(body))
    }
  )

  function git(args: string[], cwd: string = origin): string {
    return execFileSync(
      'git',
      [
        '-c',
        'user.name=Fixture',
        '-c',
        'user.email=fixture@example.invalid',
        '-c',
        'commit.gpgsign=false',
        '-c',
        'tag.gpgsign=false',
        ...args
      ],
      { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
    ).trim()
  }

  try {
    fs.mkdirSync(origin)
    fs.mkdirSync(home)
    git(['init', '-b', 'upstream-build'])
    const commits: string[] = []

    for (const label of ['old', 'stable', 'canary', 'unpublished']) {
      git(['commit', '--allow-empty', '-m', label])
      commits.push(git(['rev-parse', 'HEAD']))
    }

    const tags: Record<'stable' | 'canary', string> = { stable: 'v1.2.3', canary: 'v1.2.3+canary.20260911T123456Z' }

    for (const channel of ['stable', 'canary'] as const) {
      const sha: string = commits[channel === 'stable' ? 1 : 2]
      git(['tag', '-a', tags[channel], sha, '-m', channel])
      const manifest: FixtureManifest = buildManifest(channel, sha, tags[channel], buildId(channel))
      const body: string = canonicalJson(manifest)
      responses.set(
        `/releases/channels/${channel}.json`,
        JSON.stringify({
          ...channelRecord(channel, channel === 'stable' ? 1 : 2),
          head: {
            buildId: buildId(channel),
            sequence: channel === 'stable' ? 1 : 2,
            manifestKey: manifestKey(channel),
            sha256: crypto.createHash('sha256').update(body, 'utf8').digest('hex')
          }
        })
      )
      responses.set(`/${manifestKey(channel)}`, body)
      responses.set(`/repos/NousResearch/hermes-agent/releases/tags/${tags[channel]}`, {
        tag_name: tags[channel],
        draft: false,
        prerelease: channel === 'canary'
      })
      responses.set(`/repos/NousResearch/hermes-agent/commits/${tags[channel]}`, { sha })
    }

    responses.set('/releases/stable/release-candidates.json', { tag: tags.stable, commit: commits[1] })
    // The 'main' subscription is a source-branch channel record under the R2 protocol.
    responses.set(
      '/releases/channels/main.json',
      JSON.stringify({
        schema: 1,
        name: 'main',
        repository: 'NousResearch/hermes-agent',
        policy: 'source-branch',
        state: 'active',
        revision: 1,
        nextSequence: 1,
        identity: null,
        head: null,
        delivery: { kind: 'source-branch', branch: 'main' }
      })
    )
    git(['tag', 'v99.0.0'])
    git(['worktree', 'add', '-b', 'feature/gui', root])
    git(['remote', 'add', 'origin', origin])
    fs.writeFileSync(path.join(origin, 'install-stamp.json'), JSON.stringify({ updateMechanism: 'external' }))
    fs.writeFileSync(path.join(root, 'install-stamp.json'), JSON.stringify({ updateMechanism: 'self' }))
    await new Promise<void>((resolve: () => void): void => {
      server.listen(0, '127.0.0.1', resolve)
    })
    const address: AddressInfo = server.address() as AddressInfo
    // Redirect only network transport. Selection, config, tag validation and Git are real.
    fs.cpSync(path.join(repository, 'hermes_cli'), path.join(root, 'hermes_cli'), { recursive: true })
    fs.writeFileSync(
      path.join(root, 'transport.py'),
      `import sys, os
sys.path.append(${JSON.stringify(repository)})
assert not os.environ.get('HERMES_RUNTIME_DIR')
import urllib.request
from urllib.parse import urlsplit
original_build = urllib.request.build_opener
passthrough = original_build().open
def local(request, *args, **kwargs):
    parsed = urlsplit(request.full_url if isinstance(request, urllib.request.Request) else request)
    assert parsed.hostname in ('hermes-assets.nousresearch.com', 'api.github.com')
    url = 'http://127.0.0.1:${address.port}' + parsed.path + ('?' + parsed.query if parsed.query else '')
    # ChannelReader compares response.geturl() against the ORIGINAL request url:
    # wrap so the redirect detector still sees the un-rewritten authority.
    response = passthrough(url, *args, **kwargs)
    original_url = request.full_url if isinstance(request, urllib.request.Request) else request
    response.geturl = lambda: original_url
    return response
urllib.request.urlopen = local
def local_build(*args, **kwargs):
    # ChannelReader resolves through build_opener().open, not module urlopen.
    opener = original_build(*args, **kwargs)
    opener.open = local
    return opener
urllib.request.build_opener = local_build
`
    )
    vi.stubEnv('HERMES_MANAGED', '')
    vi.stubEnv('HERMES_RUNTIME_DIR', path.join(temporary, 'wrong-runtime'))
    vi.stubEnv('PYTHONPATH', path.join(temporary, 'wrong-checkout'))
    vi.stubEnv('PYTHONHOME', path.join(temporary, 'wrong-python'))
    vi.stubEnv('HERMES_INSTALL_ROOT', origin)

    const environment: NodeJS.ProcessEnv = {
      ...process.env,
      HERMES_HOME: home,
      HERMES_INSTALL_ROOT: root,
      PYTHONPATH: repository,
      PYTHONHOME: '',
      HERMES_RUNTIME_DIR: ''
    }

    async function setChannel(channel: 'stable' | 'canary' | 'main', install: string = root): Promise<void> {
      await execute(
        python,
        [
          '-c',
          'import sys; from pathlib import Path; from hermes_cli.update_channel import set_install_channel; set_install_channel(sys.argv[1], Path(sys.argv[2]))',
          channel,
          install
        ],
        { cwd: root, env: environment }
      )
    }

    const checkerPath: string = path.join(root, 'hermes_cli', 'source_check.py')
    fs.writeFileSync(
      checkerPath,
      fs
        .readFileSync(checkerPath, 'utf8')
        .replace(
          'from __future__ import annotations',
          `from __future__ import annotations\nimport runpy; runpy.run_path(${JSON.stringify(path.join(root, 'transport.py'))})`
        )
    )

    const deps: CheckoutStrategyDeps = {
      hermesHome: home,
      isWindows: process.platform === 'win32',
      isMac: process.platform === 'darwin',
      defaultUpdateBranch: 'main',
      updateHandoffDwellMs: 0,
      resolveUpdateRoot: (): string => root,
      readSourceUpdate: (install: string, opts: { force?: boolean }): Promise<SourceUpdate | null> =>
        readSourceUpdate({
          python,
          git: 'git',
          updateRoot: install,
          hermesHome: home,
          force: opts.force
        }),
      resolveUpdaterBinary: (): null => null,
      remoteGatewayActive: (): boolean => false,

      emitUpdateProgress: vi.fn(),
      rememberLog: vi.fn(),
      startHermes: async (): Promise<void> => {},

      stopBackendsForUpdate: vi.fn(async (): Promise<void> => {}),
      repairMacUpdaterHelper: (): void => {},
      preflightStateDb: (): void => {},
      runningAppBundle: (): null => null,
      markQuittingForHandoff: (): void => {},
      quit: (): void => {}
    }

    const strategy: ReturnType<typeof createCheckoutStrategy> = createCheckoutStrategy(deps)
    const spawned: { command: string; args: string[]; options: SpawnOptions }[] = []
    vi.spyOn(updaterProcess, 'spawnUpdaterProcess').mockImplementation(
      (command: string, args: string[], options: SpawnOptions): updaterProcess.UpdaterChild => {
        spawned.push({ command, args, options })

        return { unref: (): void => {} }
      }
    )

    const scriptDirectory: string = path.join(root, 'scripts', 'desktop-update')
    const script: string = path.join(scriptDirectory, process.platform === 'win32' ? 'windows.ps1' : 'posix.sh')
    fs.mkdirSync(scriptDirectory, { recursive: true })
    fs.writeFileSync(path.join(scriptDirectory, 'runtime.ps1'), '')
    fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })
    fs.writeFileSync(path.join(root, '.hermes', 'bin', 'hermes.exe'), '')

    for (const channel of ['stable', 'canary'] as const) {
      await setChannel(channel)
      const sha: string = commits[channel === 'stable' ? 1 : 2]
      const checked: unknown = await strategy.check()
      expect(checked, JSON.stringify({ checked, requests })).toMatchObject({
        supported: true,
        channel,
        targetSha: sha,
        updateAvailable: true
      })
      fs.rmSync(scriptDirectory, { recursive: true, force: true })
      expect(await strategy.apply()).toMatchObject({ manual: true, command: `hermes update --channel ${channel}` })
      deps.resolveUpdaterBinary = (): string => path.join(temporary, 'frozen-updater')
      expect(await strategy.apply()).toMatchObject({ manual: true, command: `hermes update --channel ${channel}` })
      expect(spawned).toHaveLength(0)
      fs.mkdirSync(scriptDirectory, { recursive: true })
      fs.writeFileSync(script, '')
      expect(await strategy.apply()).toMatchObject({ ok: true, handedOff: true })
      const handoff: (typeof spawned)[number] | undefined = spawned.pop()
      expect(handoff?.args).toContain(script)
      expect(handoff?.args).toContain(channel)
      expect(handoff?.args).toContain(process.platform === 'win32' ? '-Channel' : '--channel')
      expect(handoff?.args).not.toContain('--branch')
      expect(handoff?.args).not.toContain('-Branch')
      expect(handoff?.command).not.toBe(deps.resolveUpdaterBinary())
      expect(handoff?.options.env?.HERMES_HOME).toBe(home)
      expect(handoff?.options.env?.HERMES_INSTALL_ROOT).toBe(root)
      expect(handoff?.options.env?.PYTHONPATH).toBe('')
      expect(handoff?.options.env?.PYTHONHOME).toBe('')
      expect(handoff?.options.env?.HERMES_RUNTIME_DIR).toBeUndefined()
      deps.resolveUpdaterBinary = (): null => null
      expect(git(['rev-parse', 'HEAD'], root)).toBe(commits[3])
      git(['checkout', '--detach', sha], root)
      expect(await strategy.check()).toMatchObject({ targetSha: sha, updateAvailable: false })
      git(['checkout', 'feature/gui'], root)
    }

    // The R2 record, not GitHub metadata, decides availability: retire the
    // canary object and the resolver must fail closed before any handoff.
    responses.delete(`/releases/channels/canary.json`)
    vi.mocked(deps.stopBackendsForUpdate).mockClear()
    expect(await strategy.apply()).toMatchObject({ ok: false, error: 'release-unavailable' })
    expect(deps.stopBackendsForUpdate).not.toHaveBeenCalled()
    expect(spawned).toHaveLength(0)
    await setChannel('main')
    expect(await readSourceUpdate({ python, git: 'git', updateRoot: root, hermesHome: home })).toMatchObject({
      supported: true,
      branch: 'feature/gui',
      targetSha: commits[3],
      updateAvailable: false
    })
    const count: number = requests.length
    expect(await strategy.check()).toMatchObject({
      branch: 'feature/gui',
      targetSha: commits[3],
      updateAvailable: false
    })
    expect(await strategy.apply()).toMatchObject({ ok: true, handedOff: true })
    expect(spawned.pop()?.args).toEqual(
      expect.arrayContaining([process.platform === 'win32' ? '-Branch' : '--branch', 'feature/gui'])
    )
    fs.rmSync(scriptDirectory, { recursive: true, force: true })
    expect(await strategy.apply()).toMatchObject({ manual: true, command: 'hermes update --branch feature/gui' })
    // apply() forces a fresh check; under the R2 protocol that re-resolution
    // touches exactly the channel record — no GitHub or artifact chatter.
    expect(requests.slice(count)).toEqual(['/releases/channels/main.json', '/releases/channels/main.json'])
  } finally {
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    server.closeAllConnections()
    await new Promise<void>((resolve: () => void): void => {
      server.close((): void => resolve())
    })
    fs.rmSync(temporary, { recursive: true, force: true })
  }
}, 30000)
