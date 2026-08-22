#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const flatpakRoot = path.join(desktopRoot, 'flatpak')
const stageRoot = path.join(flatpakRoot, 'stage')
const unpackedDesktop = path.join(desktopRoot, 'release', 'linux-unpacked')

function run(command, args) {
  execFileSync(command, args, { cwd: desktopRoot, stdio: 'inherit' })
}

// Flatpak receives only the Electron client. The native Hermes installation
// remains outside the sandbox and is started by `hermes desktop` on the host.
rmSync(stageRoot, { recursive: true, force: true })
mkdirSync(stageRoot, { recursive: true })

run('npm', ['run', 'build'])
run('npm', ['run', 'builder', '--', '--linux', '--dir'])

if (!existsSync(unpackedDesktop)) {
  throw new Error(`electron-builder did not create ${unpackedDesktop}`)
}

cpSync(unpackedDesktop, path.join(stageRoot, 'hermes-desktop'), {
  recursive: true,
  dereference: false,
})

console.log('Flatpak payload staged at:', stageRoot)
