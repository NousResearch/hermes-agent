import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it } from 'vitest'

import { recordNativeInputs } from './prepared-native-deps.mjs'
import { publishPackagingInputs } from './prepared-packaging.mjs'
import { runElectronBuilder } from './run-electron-builder.mjs'

const source = path.resolve(import.meta.dirname, '../../..')
const roots = []
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function write(file, contents = '') {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, contents)
}

function listing(root) {
  return fs
    .readdirSync(root, { recursive: true, withFileTypes: true })
    .map(entry => {
      const full = path.join(entry.parentPath, entry.name)
      const name = path.relative(root, full).split(path.sep).join('/')
      return entry.isDirectory() ? `d ${name}` : `f ${name} ${fs.statSync(full).size}`
    })
    .sort()
}

// The node flags run-electron-builder.mjs starts the electron-builder child with.
async function builderPreloads(out) {
  const electron = path.join(out, 'electron.zip')
  fs.writeFileSync(electron, 'fixture archive')
  const toolsets = { sevenZip: path.join(out, 'sevenZip'), icons: path.join(out, 'icons') }
  // Windows admission also requires the signing kit and a .NET root, even for --dir.
  const windows = process.platform === 'win32'
    ? { makeappx: 'makeappx.exe', signtool: 'signtool.exe', dlib: null, dotnetRoot: path.join(out, 'dotnet') }
    : null
  if (windows) toolsets.winCodeSign = path.join(out, 'winCodeSign')
  for (const dir of [...Object.values(toolsets), windows?.dotnetRoot].filter(Boolean)) fs.mkdirSync(dir)
  const target = `${process.platform}-${process.arch}`
  const manifest = await publishPackagingInputs({ source, out, target, formats: ['dir'], electron, toolsets, windows })
  const nativeDeps = path.join(out, 'native')
  write(path.join(nativeDeps, 'node-pty/package.json'), '{}')
  recordNativeInputs({ source, out: nativeDeps, platform: process.platform, arch: process.arch })
  let args = []
  const spawn = (_node, launched) => {
    args = launched
    return { status: 0 }
  }
  expect(runElectronBuilder(['--prepared', manifest, '--native-deps', nativeDeps, '--dir'], { spawn })).toBe(0)
  return args.slice(
    0,
    args.findIndex((arg, index) => index % 2 === 0 && !['--import', '--require'].includes(arg))
  )
}

it("keeps every payload entry through electron-builder's own copies in the builder child", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'builder-copy-'))
  roots.push(root)
  const payload = path.join(root, 'agent-payload')
  write(path.join(payload, 'hermes-agent', 'run_agent.py'), 'print("hermes")\n')
  write(path.join(payload, 'hermes-agent', 'contributors', 'emails', 'someone@example.com'), 'someone\n')
  write(path.join(payload, 'hermes-agent', 'contributors', 'emails', '.gitkeep'), 'keep\n')
  write(path.join(payload, 'hermes-agent', '.DS_Store'), 'finder state')
  write(path.join(payload, 'tools', 'pnpm', 'dist', 'llhttp', '.gitkeep'))
  fs.mkdirSync(path.join(payload, 'tools', 'llamacpp', 'llama-b1'), { recursive: true })
  fs.mkdirSync(path.join(payload, 'uv-cache', 'builds-v0', 'nested'), { recursive: true })
  const probe = path.join(root, 'probe.mjs')
  // copyDir is the extraResources and AppImage stage copy; walk lists the MSIX package files.
  fs.writeFileSync(
    probe,
    `import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
const require = createRequire(${JSON.stringify(path.join(import.meta.dirname, '..', 'package.json'))})
const builderUtil = createRequire(require.resolve('app-builder-lib')).resolve('builder-util')
const { copyDir, walk } = await import(pathToFileURL(builderUtil).href)
const [from, to] = process.argv.slice(2)
await copyDir(from, to)
console.log(JSON.stringify(await walk(from)))
`
  )
  const env = { ...process.env, NODE_OPTIONS: '' }
  delete env.VITEST
  const copy = path.join(root, 'copy')
  const result = spawnSync(process.execPath, [...(await builderPreloads(root)), probe, payload, copy], {
    encoding: 'utf8',
    env,
    timeout: 60000
  })
  expect(result.status, result.stderr).toBe(0)
  expect(listing(copy)).toEqual(listing(payload))
  const walked = JSON.parse(result.stdout).map(file => path.relative(payload, file).split(path.sep).join('/'))
  const files = listing(payload)
    .filter(line => line.startsWith('f '))
    .map(line => line.split(' ')[1])
  expect(walked.sort()).toEqual(files.sort())
})
