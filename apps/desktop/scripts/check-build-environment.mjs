import path from 'node:path'

import { electronMirror, formatProblems, inspectBuildEnvironment } from './build-environment.mjs'

const repoRoot = path.resolve(import.meta.dirname, '..', '..', '..')
const mirror = electronMirror()
const problems = inspectBuildEnvironment({ repoRoot })

if (problems.length > 0) {
  console.error(formatProblems(problems, mirror))
  process.exit(1)
}

console.log(`[desktop-build] Dependency preflight passed. Electron mirror: ${mirror}`)
