import { resolve } from 'node:path'

import { electronMirror, formatProblems, inspectBuildEnvironment } from './build-environment.mjs'

const root = resolve(import.meta.dirname, '..', '..', '..')
const problems = inspectBuildEnvironment({ repoRoot: root })

if (problems.length > 0) {
  console.error(formatProblems(problems, electronMirror()))
  process.exit(1)
}
