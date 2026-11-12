import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { getFileMatchers } from 'app-builder-lib/internal'
import { afterEach, expect, it } from 'vitest'

import afterPack from './after-pack.mjs'
import { resolvePackagedAsarPath } from './backend-ready-artifact.mjs'
import { PACKED_PAYLOAD_CHANGES, payloadDifferences } from './packaged-payload.mjs'

const require = createRequire(import.meta.url)
// The extraResources copy itself (platformPackager.js calls it); the package exports no public alias.
const { copyFiles } = await import(
  pathToFileURL(path.join(path.dirname(require.resolve('app-builder-lib')), 'fileMatcher.js')).href
)

process.env.HERMES_DESKTOP_VARIANT = 'bundled'
const config = require('../electron-builder.config.cjs')

const roots = []
afterEach(() => {
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

// A linux appOutDir packed from a prepared payload, ready for the afterPack hook.
function packContext() {
  const root = scratch()
  const projectDir = path.join(root, 'project')
  const context = {
    appOutDir: path.join(root, 'linux-unpacked'),
    electronPlatformName: 'linux',
    packager: { config, projectDir }
  }
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

it('lets signing rewrite binaries on macOS and Windows but still reports every lost entry', () => {
  const root = scratch()
  const prepared = path.join(root, 'prepared')
  const packaged = path.join(root, 'packaged')
  const app = path.join('tools', 'chromium', 'Chromium.app', 'Contents')
  write(path.join(prepared, app, 'MacOS', 'Chromium'), 'mach-o')
  write(path.join(prepared, 'tools', 'uv', 'uv.exe'), 'pe')
  write(path.join(prepared, 'hermes-agent', 'contributors', 'emails', '.gitkeep'))
  fs.cpSync(prepared, packaged, { recursive: true })
  fs.appendFileSync(path.join(packaged, app, 'MacOS', 'Chromium'), ' and its signature')
  fs.appendFileSync(path.join(packaged, 'tools', 'uv', 'uv.exe'), ' and its signature')
  write(path.join(packaged, app, '_CodeSignature', 'CodeResources'), 'bundle seal')
  fs.rmSync(path.join(packaged, 'hermes-agent', 'contributors', 'emails', '.gitkeep'))
  const lost = 'missing file: hermes-agent/contributors/emails/.gitkeep'
  expect(payloadDifferences(prepared, packaged, PACKED_PAYLOAD_CHANGES.darwin)).toEqual([lost])
  expect(payloadDifferences(prepared, packaged, PACKED_PAYLOAD_CHANGES.win32)).toEqual([
    lost,
    'unexpected directory: tools/chromium/Chromium.app/Contents/_CodeSignature',
    'unexpected file: tools/chromium/Chromium.app/Contents/_CodeSignature/CodeResources'
  ])
})
