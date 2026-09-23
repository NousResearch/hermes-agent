import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { createDesktopWorkspaceCwdRuntime } from './desktop-workspace-cwd-runtime'

test('workspace cwd keeps sessions out of the packaged app and honors the saved project directory', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-workspace-cwd-'))

  try {
    const appRoot = path.join(root, 'app')
    const project = path.join(root, 'project')

    fs.mkdirSync(appRoot)
    fs.mkdirSync(project)

    const runtime = createDesktopWorkspaceCwdRuntime({
      app: { getPath: name => (name === 'userData' ? path.join(root, 'data') : root) },
      APP_ROOT: appRoot,
      SOURCE_REPO_ROOT: root,
      IS_PACKAGED: true,
      directoryExists: candidate => fs.existsSync(candidate) && fs.statSync(candidate).isDirectory(),
      rememberLog: () => undefined
    })

    runtime.writeDefaultProjectDir(project)

    expect(runtime.readDefaultProjectDir()).toBe(project)
    expect(runtime.resolveHermesCwd()).toBe(project)
    expect(runtime.sanitizeWorkspaceCwd(appRoot)).toEqual({ cwd: project, sanitized: true })
  } finally {
    expect(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep)).toBe(true)

    fs.rmSync(root, { recursive: true, force: true })
  }
})
