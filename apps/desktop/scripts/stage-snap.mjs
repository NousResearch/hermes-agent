#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const flatpakRoot = path.join(desktopRoot, 'flatpak')
const snapRoot = path.join(desktopRoot, 'snap')
const stageRoot = path.join(snapRoot, 'stage')
const unpackedDesktop = path.join(desktopRoot, 'release', 'linux-unpacked')
const snapGuiRoot = path.join(snapRoot, 'gui')

function run(command, args, cwd = desktopRoot) {
  execFileSync(command, args, { cwd, stdio: 'inherit' })
}

// This stage creates the Snapcraft payload from the Electron linux-unpacked build
rmSync(stageRoot, { recursive: true, force: true })
rmSync(path.join(snapRoot, 'prime'), { recursive: true, force: true })
rmSync(path.join(snapRoot, 'parts'), { recursive: true, force: true })
mkdirSync(stageRoot, { recursive: true })

run('npm', ['run', 'build'])
run('npm', ['run', 'builder', '--', '--linux', '--dir'])
if (!existsSync(unpackedDesktop)) {
  throw new Error(`electron-builder did not create ${unpackedDesktop}`)
}
cpSync(unpackedDesktop, path.join(stageRoot, 'hermes-desktop'), { recursive: true, dereference: false })

// Copy the Electron binary into the right place for snapcraft
mkdirSync(path.join(snapRoot, 'prime', 'share', 'hermes-desktop'), { recursive: true })
cpSync(path.join(stageRoot, 'hermes-desktop'), path.join(snapRoot, 'prime', 'share', 'hermes-desktop'), { recursive: true, dereference: false })

// Copy gui files
if (existsSync(snapGuiRoot)) {
  cpSync(snapGuiRoot, path.join(snapRoot, 'prime', 'gui'), { recursive: true })
}

console.log('Snapcraft payload staged at:', snapRoot)