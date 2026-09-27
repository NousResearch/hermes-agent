import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

export interface ExistingCollaborator {
  root: string
  python: string
}
export type CollaboratorChoice = { mode: 'bundled' } | { mode: 'existing'; root: string; python: string }
const run = promisify(execFile)

/** Older Hermes installations cannot serve the Czesiek Live voice adapter. */
export function supportsCzesiekLive(root: string): boolean {
  return fs.existsSync(path.join(root, 'hermes_cli', 'web_routers', 'voice_realtime.py'))
}

export function readCollaboratorChoice(file: string): CollaboratorChoice | null {
  let choice: CollaboratorChoice | null

  try {
    choice = JSON.parse(fs.readFileSync(file, 'utf8'))
  } catch {
    // A missing or damaged preference must leave setup available.
    return null
  }

  if (!choice) {
    return null
  }

  if (choice.mode === 'bundled') {
    return { mode: 'bundled' }
  }

  if (
    choice.mode === 'existing' &&
    typeof choice.root === 'string' &&
    typeof choice.python === 'string' &&
    fs.existsSync(choice.python) &&
    fs.existsSync(path.join(choice.root, 'hermes_cli/main.py'))
  ) {
    return choice
  }

  return null
}

export function saveCollaboratorChoice(file: string, choice: CollaboratorChoice) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file + '.tmp', JSON.stringify(choice), 'utf8')
  fs.renameSync(file + '.tmp', file)
}

// Validate imports without starting another gateway, changing its config or
// installing anything. Detection never blocks Electron's event loop.
export async function discoverCollaborators(roots: string[]): Promise<ExistingCollaborator[]> {
  const probeHome = fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-probe-'))

  try {
    const results = await Promise.all(
      [...new Set(roots.map(root => path.resolve(root)))].map(async root => {
        if (!fs.existsSync(path.join(root, 'hermes_cli/main.py')) || !supportsCzesiekLive(root)) {
          return null
        }

        for (const venv of ['venv', '.venv']) {
          const python = path.join(root, venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

          if (!fs.existsSync(python)) {
            continue
          }

          try {
            await run(python, ['-c', 'import hermes_cli.main, hermes_cli.web_routers.voice_realtime, fastapi, uvicorn'], {
              cwd: root,
              windowsHide: true,
              timeout: 30000,
              env: {
                ...process.env,
                HERMES_HOME: probeHome,
                PYTHONPATH: root,
                PYTHONNOUSERSITE: '1',
                PYTHONDONTWRITEBYTECODE: '1',
                PYTHONHOME: ''
              }
            })

            return { root, python }
          } catch {
            // An incomplete environment is not an available collaborator.
          }
        }

        return null
      })
    )

    return results.filter((value): value is ExistingCollaborator => value !== null)
  } finally {
    fs.rmSync(probeHome, { recursive: true, force: true })
  }
}
