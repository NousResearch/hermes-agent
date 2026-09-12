import { execFile, execFileSync, type SpawnOptions } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'
import { promisify } from 'node:util'

import { expect, it, vi } from 'vitest'

import * as updaterProcess from '../updater-process'
import * as blockers from '../venv-blocker-scan'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import { readSourceUpdate, type SourceUpdate } from './checkout-source'

const execute: typeof execFile.__promisify__ = promisify(execFile)
const repository: string = path.resolve(import.meta.dirname, '../../../..')
const python: string = process.env.HERMES_PYTHON || 'python3'

it('carries each install channel from Python publication checks into the source handoff', async (): Promise<void> => {
  const temporary: string = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-channel-'))
  const origin: string = path.join(temporary, 'origin')
  const root: string = path.join(temporary, 'checkout')
  const home: string = path.join(temporary, 'profile')
  const requests: string[] = []
  const responses: Map<string, unknown> = new Map<string, unknown>()

  const server: http.Server = http.createServer(
    (request: http.IncomingMessage, response: http.ServerResponse): void => {
      const url: string = request.url ?? ''
      requests.push(url)
      const body: unknown = responses.get(url)
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
    git(['init', '-b', 'feature/gui'])
    const commits: string[] = []

    for (const label of ['old', 'stable', 'canary', 'unpublished']) {
      git(['commit', '--allow-empty', '-m', label])
      commits.push(git(['rev-parse', 'HEAD']))
    }

    const tags: Record<'stable' | 'canary', string> = { stable: 'v1.2.3', canary: 'v1.2.4-canary.20260911123456' }

    for (const channel of ['stable', 'canary'] as const) {
      const sha: string = commits[channel === 'stable' ? 1 : 2]
      git(['tag', '-a', tags[channel], sha, '-m', channel])
      responses.set(`/releases/${channel}/index.html`, `<meta name="hermes-build" content="${tags[channel]}">`)
      responses.set(`/repos/NousResearch/hermes-agent/releases/tags/${tags[channel]}`, {
        tag_name: tags[channel],
        draft: false,
        prerelease: channel === 'canary'
      })
      responses.set(`/repos/NousResearch/hermes-agent/commits/${tags[channel]}`, { sha })
    }

    responses.set('/releases/stable/release-candidates.json', { tag: tags.stable, commit: commits[1] })
    git(['tag', 'v99.0.0'])
    git(['clone', origin, root], temporary)
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
import urllib.request\nfrom urllib.parse import urlsplit\noriginal = urllib.request.urlopen\ndef local(request, *args, **kwargs):\n    parsed = urlsplit(request.full_url if isinstance(request, urllib.request.Request) else request)\n    assert parsed.hostname in ('hermes-assets.nousresearch.com', 'api.github.com')\n    return original('http://127.0.0.1:${address.port}' + parsed.path + ('?' + parsed.query if parsed.query else ''), *args, **kwargs)\nurllib.request.urlopen = local\n`
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

    const deps: CheckoutStrategyDeps = {
      hermesHome: home,
      isWindows: process.platform === 'win32',
      isMac: process.platform === 'darwin',
      defaultUpdateBranch: 'main',
      updateHandoffDwellMs: 0,
      updateCheckCachePath: path.join(home, 'cache.json'),
      writeFileAtomic: (file: string, contents: string): void => fs.writeFileSync(file, contents),
      isGitCheckout: (): boolean => true,
      readCanonicalInstallStamp: (): null => null,
      readDesktopUpdateConfig: (): { branch: string; branchExplicit: boolean } => ({
        branch: 'main',
        branchExplicit: false
      }),
      resolveUpdateRoot: (): string => root,
      readSourceUpdate: async (install: string): Promise<SourceUpdate> => {
        const result: { stdout: string } = await execute(
          python,
          [
            '-c',
            "import runpy,sys; runpy.run_path(sys.argv.pop(1)); runpy.run_module('hermes_cli.source_releases', run_name='__main__')",
            path.join(root, 'transport.py'),
            '--install-root',
            install,
            '--git',
            'git'
          ],
          { cwd: root, env: environment }
        )

        return JSON.parse(result.stdout) as SourceUpdate
      },
      resolveHealedBranch: async (_root: string, branch: string): Promise<string> => branch,
      getOriginUrl: async (): Promise<string> => origin,
      runGit: async (args: string[]): Promise<{ code: number; stdout: string; stderr: string }> => ({
        code: 0,
        stdout: git(args, root),
        stderr: ''
      }),
      fetchGitHubApi: async (): Promise<never> => {
        throw new Error('branch API must not resolve releases')
      },
      directoryExists: fs.existsSync,
      resolveUpdaterBinary: (): null => null,
      firstLine: (text: string): string => text.split('\n')[0],
      pathWithVenvBin: (): string => process.env.PATH ?? '',
      venvHermesShimPath: (): string => '',
      emitUpdateProgress: vi.fn(),
      rememberLog: vi.fn(),
      startHermes: async (): Promise<void> => {},
      startGatewaysAfterUpdateAbort: (): void => {},
      releaseBackendLockForUpdate: vi.fn(async (): Promise<{ unlocked: boolean }> => ({ unlocked: true })),
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
    vi.spyOn(blockers, 'scanVenvBlockers').mockResolvedValue({
      kind: 'clear',
      result: { blocked: false, processes: [] }
    })
    const scriptDirectory: string = path.join(root, 'scripts', 'desktop-update')
    const script: string = path.join(scriptDirectory, process.platform === 'win32' ? 'windows.ps1' : 'posix.sh')
    fs.mkdirSync(path.join(root, 'venv', 'Scripts'), { recursive: true })
    fs.writeFileSync(path.join(root, 'venv', 'Scripts', 'python.exe'), '')

    for (const channel of ['stable', 'canary'] as const) {
      await setChannel(channel)
      await setChannel(channel === 'stable' ? 'canary' : 'stable', origin)
      const sha: string = commits[channel === 'stable' ? 1 : 2]
      const checked: unknown = await strategy.check()
      expect(checked, JSON.stringify({ checked, requests })).toMatchObject({
        supported: true,
        channel,
        latestTag: tags[channel],
        targetSha: sha,
        updateAvailable: true
      })
      fs.rmSync(scriptDirectory, { recursive: true, force: true })
      expect(await strategy.apply({})).toMatchObject({ manual: true, command: `hermes update --channel ${channel}` })
      deps.resolveUpdaterBinary = (): string => path.join(temporary, 'frozen-updater')
      expect(await strategy.apply({})).toMatchObject({ manual: true, command: `hermes update --channel ${channel}` })
      expect(spawned).toHaveLength(0)
      fs.mkdirSync(scriptDirectory, { recursive: true })
      fs.writeFileSync(script, '')
      expect(await strategy.apply({})).toMatchObject({ ok: true, handedOff: true })
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

    responses.set(`/repos/NousResearch/hermes-agent/releases/tags/${tags.canary}`, {
      tag_name: tags.canary,
      draft: true,
      prerelease: true
    })
    vi.mocked(deps.releaseBackendLockForUpdate).mockClear()
    expect(await strategy.apply({})).toMatchObject({ ok: false, error: 'release-unavailable' })
    expect(deps.releaseBackendLockForUpdate).not.toHaveBeenCalled()
    expect(spawned).toHaveLength(0)
    await setChannel('main')
    expect(await readSourceUpdate({ python, git: 'git', updateRoot: root, hermesHome: home })).toEqual({
      channel: 'main'
    })
    const count: number = requests.length
    expect(await strategy.check()).toMatchObject({
      branch: 'feature/gui',
      targetSha: commits[3],
      updateAvailable: false
    })
    expect(await strategy.apply({})).toMatchObject({ ok: true, handedOff: true })
    expect(spawned.pop()?.args).toEqual(
      expect.arrayContaining([process.platform === 'win32' ? '-Branch' : '--branch', 'feature/gui'])
    )
    fs.rmSync(scriptDirectory, { recursive: true, force: true })
    expect(await strategy.apply({})).toMatchObject({ manual: true, command: 'hermes update --branch feature/gui' })
    expect(requests).toHaveLength(count)
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
