import { spawnSync } from 'node:child_process'
import path from 'node:path'

import { connectorElectronBinary } from './connector-electron-binary.mjs'

const desktop = path.resolve(import.meta.dirname, '..')
const binary = connectorElectronBinary(desktop)
const result = spawnSync(binary, ['--version'], { stdio: 'inherit' })
if (result.error) throw result.error
process.exitCode = result.status ?? 1
