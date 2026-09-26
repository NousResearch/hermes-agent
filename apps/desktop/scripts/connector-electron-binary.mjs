import fs from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

// Resolve the package without loading its entry point, which can download a missing binary.
export function connectorElectronBinary(desktop) {
  const electron = path.dirname(createRequire(path.resolve(desktop, 'package.json')).resolve('electron/package.json'))
  const pathFile = path.join(electron, 'path.txt')
  if (!fs.existsSync(pathFile)) throw new Error(`Electron path.txt missing: ${pathFile}`)
  const executable = fs.readFileSync(pathFile, 'utf8').trim()
  if (!executable) throw new Error(`Electron path.txt empty: ${pathFile}`)
  const binary = path.join(electron, 'dist', executable)
  if (!fs.existsSync(binary)) throw new Error(`Electron binary missing: ${binary}`)
  return binary
}
