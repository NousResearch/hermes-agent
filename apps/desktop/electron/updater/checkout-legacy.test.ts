import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, it, vi } from 'vitest'

import { type CheckoutStrategyDeps, createCheckoutStrategy } from './checkout'
import { readSourceUpdate, type SourceUpdate } from './checkout-source'

it('moves a checkout without a source probe to main, but surfaces a broken probe', async (): Promise<void> => {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'legacy-channel-'))
  const home: string = path.join(root, 'profile')
  const modulePath: string = path.join(root, 'hermes_cli', 'source_check.py')
  fs.mkdirSync(path.dirname(modulePath))
  fs.mkdirSync(home)
  fs.writeFileSync(path.join(root, 'hermes_cli', '__init__.py'), '')

  const probe: () => Promise<SourceUpdate | null> = (): Promise<SourceUpdate | null> =>
    readSourceUpdate({
      python: process.env.HERMES_PYTHON || 'python3',
      git: 'git',
      updateRoot: root,
      hermesHome: home
    })

  const deps: CheckoutStrategyDeps = {
    readSourceUpdate: probe,
    hermesHome: home,
    isWindows: process.platform === 'win32',
    isMac: process.platform === 'darwin',
    defaultUpdateBranch: 'main',
    updateHandoffDwellMs: 0,
    resolveUpdateRoot: (): string => root,
    // No staged updater and no repo hand-off script: apply ends on the manual
    // command card, which exposes the target without spawning anything.
    resolveUpdaterBinary: vi.fn((): null => null),
    remoteGatewayActive: (): boolean => false,
    emitUpdateProgress: vi.fn(),
    rememberLog: vi.fn(),
    startHermes: vi.fn(async (): Promise<void> => {}),
    stopBackendsForUpdate: vi.fn(async (): Promise<void> => {}),
    repairMacUpdaterHelper: vi.fn(),
    preflightStateDb: vi.fn(),
    runningAppBundle: (): null => null,
    markQuittingForHandoff: vi.fn(),
    quit: vi.fn()
  }

  const strategy: ReturnType<typeof createCheckoutStrategy> = createCheckoutStrategy(deps)

  try {
    for (const oldModule of ['def resolve_source_release(channel):\n    return None, None\n', null]) {
      if (oldModule !== null) {
        fs.writeFileSync(modulePath, oldModule)
      } else {
        fs.rmSync(modulePath)
      }

      expect(await strategy.check()).toMatchObject({ supported: true, updateAvailable: true, branch: 'main' })
      expect(await strategy.apply()).toMatchObject({ ok: true, manual: true, command: 'hermes update' })
    }

    fs.writeFileSync(modulePath, 'def main():\n    raise RuntimeError("invalid channel configuration")\n')
    await expect(strategy.apply()).rejects.toThrow('invalid channel configuration')
    fs.writeFileSync(modulePath, 'import missing_probe_dependency\n')
    await expect(probe()).rejects.toThrow('missing_probe_dependency')
    expect(deps.stopBackendsForUpdate).not.toHaveBeenCalled()
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

it.skipIf(process.platform === 'win32')(
  'uses the install-scoped PM launcher rather than a system Python for source checks',
  async (): Promise<void> => {
    const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'pm-source-check-'))
    const home: string = path.join(root, 'profile')
    const launcher: string = path.join(root, '.hermes', 'bin', 'hermes')
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    fs.mkdirSync(path.join(root, 'pm'))
    fs.mkdirSync(home)
    fs.writeFileSync(
      launcher,
      '#!/bin/sh\n[ "$1" = --run-module ] && [ "$2" = hermes_cli.source_check ] || exit 5\nprintf \'%s\\n\' \'{"supported":true,"channel":"stable","behind":-1}\'\n',
      { mode: 0o755 }
    )

    try {
      const probe = {
        python: '/nonexistent/system-python',
        git: 'git',
        updateRoot: root,
        hermesHome: home,
        channel: 'stable' as const
      }
      await expect(readSourceUpdate(probe)).resolves.toMatchObject({ supported: true, channel: 'stable', behind: null })
      fs.rmSync(launcher)
      await expect(readSourceUpdate(probe)).rejects.toThrow('installation launcher is missing')
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)

it.skipIf(process.platform !== 'win32')(
  'runs a PM .cmd source check with quoted paths and refuses a missing launcher',
  async (): Promise<void> => {
    const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'pm source check '))
    const home: string = path.join(root, 'profile with spaces')
    const launcher: string = path.join(root, '.hermes', 'bin', 'hermes.cmd')
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    fs.mkdirSync(path.join(root, 'pm'))
    fs.mkdirSync(home)
    fs.writeFileSync(
      launcher,
      '@echo off\r\nif not "%~1"=="--run-module" exit /b 5\r\nif not "%~2"=="hermes_cli.source_check" exit /b 6\r\necho {"supported":true,"channel":"stable","behind":-1}\r\n'
    )

    try {
      const probe = {
        python: 'nonexistent-system-python',
        git: 'git',
        updateRoot: root,
        hermesHome: home,
        channel: 'stable' as const
      }

      await expect(readSourceUpdate(probe)).resolves.toMatchObject({ supported: true, channel: 'stable', behind: null })
      await expect(readSourceUpdate({ ...probe, branch: 'main&echo INJECTED' })).rejects.toThrow(
        'unsafe Windows command argument'
      )
      fs.rmSync(launcher)
      await expect(readSourceUpdate(probe)).rejects.toThrow('installation launcher is missing')
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  }
)
