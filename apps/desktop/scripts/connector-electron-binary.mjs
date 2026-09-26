import { createRequire } from 'node:module'
import path from 'node:path'

// Resolve from the desktop package so both hoisted and workspace-local npm layouts work.
export function connectorElectronBinary(desktop) {
  return createRequire(path.resolve(desktop, 'package.json'))('electron')
}
