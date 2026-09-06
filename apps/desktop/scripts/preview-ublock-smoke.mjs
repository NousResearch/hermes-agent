import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

if (process.env.HERMES_UBOL_SMOKE !== '1') {
  console.log('Preview uBlock smoke test skipped; set HERMES_UBOL_SMOKE=1 to enable it.')
  process.exit(0)
}

const desktopRoot = path.resolve(new URL('..', import.meta.url).pathname)
const electronBinary = [
  path.join(desktopRoot, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron'),
  path.resolve(desktopRoot, '../../node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
].find(candidate => fs.existsSync(candidate))

if (!electronBinary) throw new Error('Electron binary not found')

const mainBundle = path.join(desktopRoot, 'dist/electron-main.mjs')
if (!fs.existsSync(mainBundle)) throw new Error('Build Desktop before running the Preview uBlock smoke test')

const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-preview-ublock-smoke-'))
try {
  execFileSync(electronBinary, [mainBundle, `--user-data-dir=${userData}`], {
    cwd: desktopRoot,
    env: { ...process.env, HERMES_UBOL_SMOKE: '1', ELECTRON_RUN_AS_NODE: '' },
    stdio: 'inherit'
  })
} finally {
  fs.rmSync(userData, { force: true, recursive: true })
}
