import { execFileSync as nodeExecFileSync } from 'node:child_process'
import path from 'node:path'

export type DesktopUpdateRuntimeKind = 'git-checkout' | 'managed-runtime' | 'unknown-runtime'

export interface DesktopUpdateRuntime {
  kind: DesktopUpdateRuntimeKind
  message: string | null
  runtimeRoot: string | null
  supported: boolean
  updateRoot: string
}

interface ProbeDeps {
  execFileSync?: (command: string, args: string[], options: Record<string, unknown>) => string | Buffer
  isWindows?: boolean
}

const MANAGED_RUNTIME_MESSAGE =
  'This Hermes runtime is managed by an immutable release. Update it through its release manager; Desktop will not run the Git updater against a different checkout.'

const UNKNOWN_RUNTIME_MESSAGE =
  'Desktop could not verify which Hermes runtime the updater would modify. Update was disabled to avoid changing the wrong installation.'

export function classifyDesktopUpdateRuntime({
  updateRoot,
  runtimeRoot
}: {
  updateRoot: string
  runtimeRoot: string | null
}): DesktopUpdateRuntime {
  const normalizedUpdateRoot = path.resolve(updateRoot)
  const normalizedRuntimeRoot = runtimeRoot ? path.resolve(runtimeRoot) : null

  if (!normalizedRuntimeRoot) {
    return {
      kind: 'unknown-runtime',
      message: UNKNOWN_RUNTIME_MESSAGE,
      runtimeRoot: null,
      supported: false,
      updateRoot: normalizedUpdateRoot
    }
  }

  if (normalizedRuntimeRoot !== normalizedUpdateRoot) {
    return {
      kind: 'managed-runtime',
      message: MANAGED_RUNTIME_MESSAGE,
      runtimeRoot: normalizedRuntimeRoot,
      supported: false,
      updateRoot: normalizedUpdateRoot
    }
  }

  return {
    kind: 'git-checkout',
    message: null,
    runtimeRoot: normalizedRuntimeRoot,
    supported: true,
    updateRoot: normalizedUpdateRoot
  }
}

export function probeDesktopUpdateRuntime(updateRoot: string, deps: ProbeDeps = {}): DesktopUpdateRuntime {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const python = path.join(updateRoot, 'venv', isWindows ? 'Scripts' : 'bin', isWindows ? 'python.exe' : 'python')
  const execFileSync = deps.execFileSync ?? nodeExecFileSync

  try {
    const output = execFileSync(
      python,
      [
        '-c',
        'import hermes_cli.config as c; print(c.get_project_root())'
      ],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 15_000, windowsHide: true }
    )

    const runtimeRoot = String(output).trim().split(/\r?\n/).filter(Boolean).at(-1) || null

    return classifyDesktopUpdateRuntime({ updateRoot, runtimeRoot })
  } catch {
    return classifyDesktopUpdateRuntime({ updateRoot, runtimeRoot: null })
  }
}
