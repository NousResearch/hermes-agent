#!/usr/bin/env node
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import { spawnSync } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { parseArgs } from 'node:util'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { readNativeInputs } from './prepared-native-deps.mjs'
import { readPackagingInputs } from './prepared-packaging.mjs'

/** @param {string} nativeDeps @returns {Promise<void>} */
async function probeChild(nativeDeps) {
  if (!process.versions.electron) throw new Error('Electron runtime required')
  /** @type {typeof import('node-pty')} */
  const pty = createRequire(import.meta.url)(path.join(nativeDeps, 'node-pty'))
  const marker = `hermes-pty-${randomUUID()}`
  const windows = process.platform === 'win32'
  const shell = windows ? (process.env.ComSpec || 'cmd.exe') : '/bin/sh'
  const args = windows ? ['/d', '/s', '/c', `echo ${marker}`] : ['-c', `echo ${marker}`]
  const terminal = pty.spawn(shell, args, { cols: 80, rows: 24, cwd: process.cwd(), env: process.env })
  let output = ''
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      terminal.kill()
      reject(new Error('Prepared Electron PTY timed out'))
    }, 10000)
    terminal.onData(data => { output += data })
    terminal.onExit(({ exitCode }) => {
      clearTimeout(timer)
      if (exitCode !== 0 || !output.includes(marker)) reject(new Error(`Prepared Electron PTY failed: exit=${exitCode}, output=${output}`))
      else resolve(undefined)
    })
  })
  fs.writeSync(1, JSON.stringify({ electron: process.versions.electron, node: process.versions.node,
    platform: process.platform, arch: process.arch, marker, exitCode: 0 }) + '\n')
}

/**
 * Run every preparation: this execution receipt is job-local, never a cache hit.
 * @param {{source: string, nativeDeps: string, packaging: string, out: string, nativeToolchain?: string}} options
 * @returns {{electron: string, node: string, platform: string, arch: string, marker: string, exitCode: number, work: string}}
 */
export function probePreparedNative({ source, nativeDeps, packaging, out, nativeToolchain }) {
  const admitted = readNativeInputs({ source, nativeDeps, nativeToolchain })
  const inputs = readPackagingInputs(packaging, source)
  fs.mkdirSync(out, { recursive: true })
  const work = fs.mkdtempSync(path.join(path.resolve(out), 'electron-probe-'))
  const bin = path.join(inputs.toolsets.sevenZip, 'bin')
  const archiveTool = fs.readdirSync(bin).find(name => /^(7zz|7za|7z)(\.exe)?$/.test(name))
  if (!archiveTool) throw new Error('Prepared 7zip executable is missing')
  const extract = spawnSync(path.join(bin, archiveTool), ['x', '-y', `-o${work}`, inputs.electron], {
    encoding: 'utf8', timeout: 60000, stdio: ['ignore', 'pipe', 'pipe'],
  })
  if (extract.error || extract.status !== 0) throw new Error(`Electron extraction failed: ${extract.error?.message || extract.stderr}`)
  const executable = new Map([['win32', 'electron.exe'], ['darwin', 'Electron.app/Contents/MacOS/Electron'], ['linux', 'electron']]).get(process.platform)
  if (!executable) throw new Error(`Unsupported native probe platform: ${process.platform}`)
  const child = spawnSync(path.join(work, executable), [fileURLToPath(import.meta.url), '--child', admitted], {
    cwd: work, encoding: 'utf8', timeout: 15000, killSignal: 'SIGKILL',
    env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' }, stdio: ['ignore', 'pipe', 'pipe'],
  })
  if (child.error || child.status !== 0) throw new Error(`Prepared Electron PTY admission failed: ${child.error?.message || child.stderr}`)
  const result = JSON.parse(child.stdout.trim())
  if (!result.electron || result.platform !== process.platform || result.arch !== process.arch || result.exitCode !== 0) {
    throw new Error('Prepared Electron PTY returned an invalid admission result')
  }
  fs.writeFileSync(path.join(work, 'result.json'), JSON.stringify({ ...result, nativeToolchain, nativeDeps: admitted }, null, 2) + '\n')
  return { ...result, work }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv[2] === '--child') {
    await probeChild(path.resolve(process.argv[3]))
    // ConPTY can retain a worker after shell exit. This isolated probe has
    // completed its contract and synchronously flushed its result.
    process.exit(0)
  } else {
    const { values } = parseArgs({ options: {
      source: { type: 'string' }, 'native-deps': { type: 'string' }, packaging: { type: 'string' },
      out: { type: 'string' }, 'native-toolchain': { type: 'string' },
    } })
    if (!values.source || !values['native-deps'] || !values.packaging || !values.out) {
      throw new Error('--source, --native-deps, --packaging and --out are required')
    }
    console.log(JSON.stringify(probePreparedNative({ source: values.source, nativeDeps: values['native-deps'],
      packaging: values.packaging, out: values.out, nativeToolchain: values['native-toolchain'] })))
  }
}
