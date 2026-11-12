import childProcess from 'node:child_process'
import fs from 'node:fs'
import { createRequire, syncBuiltinESMExports } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { Platform, PlatformPackager } from 'app-builder-lib'
import { getFileMatchers } from 'app-builder-lib/internal'
import { afterEach, expect, it, vi } from 'vitest'

import afterPack from './after-pack.mjs'
import { resolvePackagedAsarPath } from './backend-ready-artifact.mjs'

const require = createRequire(import.meta.url)
// The extraResources copy itself (platformPackager.js calls it); the package exports no public alias.
const { copyFiles } = await import(
  pathToFileURL(path.join(path.dirname(require.resolve('app-builder-lib')), 'fileMatcher.js')).href
)

process.env.HERMES_DESKTOP_VARIANT = 'bundled'
const config = require('../electron-builder.config.cjs')

const roots = []
afterEach(() => {
  vi.restoreAllMocks()
  syncBuiltinESMExports()
  vi.unstubAllEnvs()
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function scratch() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'packaged-payload-'))
  roots.push(root)
  return root
}

function write(file, contents = '') {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, contents)
}

// An appOutDir packed from a prepared payload, ready for the afterPack hook.
function packContext(platform = 'linux') {
  const root = scratch()
  const projectDir = path.join(root, 'project')
  const packager = platform === 'darwin'
    ? Object.assign(Object.create(PlatformPackager.prototype), {
      platform: Platform.MAC,
      appInfo: { productFilename: 'Hermes' },
      info: { framework: { distMacOsAppName: 'Electron.app' }, config, projectDir }
    })
    : { config, projectDir }
  const context = { appOutDir: path.join(root, `${platform}-unpacked`), electronPlatformName: platform, packager }
  const prepared = path.join(projectDir, 'build', 'agent-payload')
  write(path.join(prepared, 'hermes-agent', 'run_agent.py'), 'print("hermes")\n')
  write(path.join(prepared, 'hermes-agent', 'contributors', 'emails', 'someone@example.com'), 'someone\n')
  write(path.join(prepared, 'hermes-agent', 'contributors', 'emails', '.gitkeep'))
  write(path.join(prepared, 'tools', 'pnpm', 'dist', 'llhttp', '.gitkeep'))
  write(path.join(prepared, 'tools', 'pnpm', 'bin', 'pnpm'), '#!/bin/sh\n')
  fs.mkdirSync(path.join(prepared, 'uv-cache', 'builds-v0'), { recursive: true })
  const asar = resolvePackagedAsarPath(context)
  write(asar, 'stub archive')
  write(
    path.join(`${asar}.unpacked`, 'dist', 'electron-main.mjs'),
    'const re = /HERMES_(?:BACKEND|DASHBOARD)_READY[^\\n]*port=(\\d+)/m\n'
  )
  const resources = path.dirname(asar)
  return { context, prepared, resources, packaged: path.join(resources, 'agent-payload') }
}

it('lets an exact payload copy through afterPack and fails a lossy one, naming each difference', async () => {
  const exact = packContext()
  fs.cpSync(exact.prepared, exact.packaged, { recursive: true })
  await expect(afterPack(exact.context)).resolves.toBeUndefined()

  const lossy = packContext()
  const matchers = getFileMatchers(config, 'extraResources', lossy.resources, {
    macroExpander: value => value,
    customBuildOptions: {},
    globalOutDir: path.join(lossy.context.packager.projectDir, 'release'),
    defaultSrc: lossy.context.packager.projectDir
  }).filter(matcher => matcher.from === lossy.prepared)
  expect(matchers).toHaveLength(1)
  await copyFiles(matchers, null, false)
  fs.rmSync(path.join(lossy.packaged, 'hermes-agent', 'run_agent.py'))
  // Under vitest builder-util hard-links its copies; a new inode keeps the prepared file intact.
  fs.rmSync(path.join(lossy.packaged, 'tools', 'pnpm', 'bin', 'pnpm'))
  write(path.join(lossy.packaged, 'tools', 'pnpm', 'bin', 'pnpm'), '#!')
  write(path.join(lossy.packaged, 'tools', 'stray.tmp'), 'left behind')
  const failure = await afterPack(lossy.context).catch(error => error)
  expect(failure).toBeInstanceOf(Error)
  for (const difference of [
    'missing file of 0 bytes: hermes-agent/contributors/emails/.gitkeep',
    'missing file of 0 bytes: tools/pnpm/dist/llhttp/.gitkeep',
    'missing directory: tools/pnpm/dist/llhttp',
    'missing directory: uv-cache/builds-v0',
    'missing file of 16 bytes: hermes-agent/run_agent.py',
    'tools/pnpm/bin/pnpm: file of 10 bytes became file of 2 bytes',
    'unexpected file of 11 bytes: tools/stray.tmp'
  ]) {
    expect(failure.message).toContain(difference)
  }
})

it('fails a bundled pack whose payload never arrived but leaves Hermes Light alone', async () => {
  const missing = packContext()
  const failure = await afterPack(missing.context).catch(error => error)
  expect(failure).toBeInstanceOf(Error)
  expect(failure.message).toContain(`packaged payload ${missing.packaged} is missing`)

  vi.stubEnv('HERMES_DESKTOP_VARIANT', 'light')
  for (const module of ['../electron-builder.config.cjs', '../product-identity.cjs']) {
    delete require.cache[require.resolve(module)]
  }
  const light = packContext()
  light.context.packager.config = require('../electron-builder.config.cjs')
  await expect(afterPack(light.context)).resolves.toBeUndefined()
})

// PM's tree digests hash link text, so a retargeted link is a lost payload entry too.
it.skipIf(process.platform === 'win32')('fails a packaged symlink that points somewhere else', async () => {
  const pack = () => {
    const packed = packContext()
    fs.symlinkSync('pnpm', path.join(packed.prepared, 'tools', 'pnpm', 'bin', 'pnpm-cli'))
    fs.cpSync(packed.prepared, packed.packaged, { recursive: true, verbatimSymlinks: true })
    return packed
  }
  await expect(afterPack(pack().context)).resolves.toBeUndefined()

  const retargeted = pack()
  const link = path.join(retargeted.packaged, 'tools', 'pnpm', 'bin', 'pnpm-cli')
  fs.rmSync(link)
  fs.symlinkSync('missing-pnpm', link)
  const failure = await afterPack(retargeted.context).catch(error => error)
  expect(failure).toBeInstanceOf(Error)
  expect(failure.message).toContain('tools/pnpm/bin/pnpm-cli: symlink to pnpm became symlink to missing-pnpm')
})

it('checks the macOS copy before signing repairs a flattened Chromium framework', async () => {
  const chromium = path.join('tools', 'chromium-1208', 'Chromium.app')
  const framework = path.join(chromium, 'Contents', 'Frameworks', 'F.framework')
  const identity = '0123456789ABCDEF0123456789ABCDEF01234567'
  const signed = []
  vi.spyOn(childProcess, 'execFileSync').mockImplementation((command, args) => {
    if (command === 'security') return `  1) ${identity} "Developer ID Application: Fixture (TEAM)"\n`
    signed.push(args.at(-1))
    return ''
  })
  syncBuiltinESMExports()
  vi.stubEnv('CSC_NAME', undefined)
  vi.stubEnv('CSC_KEYCHAIN', undefined)
  const pack = () => {
    const packed = packContext('darwin')
    write(path.join(packed.prepared, framework, 'Versions', 'A', 'F'), 'mach-o')
    write(path.join(packed.prepared, framework, 'Versions', 'A', 'Resources', 'Info.plist'), '<plist/>')
    write(path.join(packed.prepared, framework, 'F'), 'mach-o')
    write(path.join(packed.prepared, framework, 'Resources', 'Info.plist'), '<plist/>')
    fs.cpSync(packed.prepared, packed.packaged, { recursive: true })
    return packed
  }

  const exact = pack()
  await expect(afterPack(exact.context)).resolves.toBeUndefined()
  expect(fs.lstatSync(path.join(exact.packaged, framework, 'F')).isSymbolicLink()).toBe(true)
  expect(signed).toEqual([path.join(exact.packaged, chromium)])

  signed.length = 0
  const lossy = pack()
  fs.rmSync(path.join(lossy.packaged, 'hermes-agent', 'contributors', 'emails', '.gitkeep'))
  fs.appendFileSync(path.join(lossy.packaged, framework, 'Versions', 'A', 'F'), ' and more')
  const failure = await afterPack(lossy.context).catch(error => error)
  expect(failure).toBeInstanceOf(Error)
  expect(failure.message).toContain('missing file of 0 bytes: hermes-agent/contributors/emails/.gitkeep')
  expect(failure.message).toContain(
    `${framework.split(path.sep).join('/')}/Versions/A/F: file of 6 bytes became file of 15 bytes`
  )
  expect(signed).toEqual([])
})
