import { execFile } from 'node:child_process'

export type PreUpdateBackupMode = 'full' | 'off' | 'quick'

export interface PreUpdateBackupPolicy {
  backupKeep: number
  mode: PreUpdateBackupMode
  quickKeep: number
  quickMaxFileSize: number
}

export interface RunPolicyProcessOptions {
  cwd: string
  env: NodeJS.ProcessEnv
  maxBuffer: number
  shell: false
  timeout: number
  windowsHide: true
}

export type RunPolicyProcess = (
  command: string,
  args: string[],
  options: RunPolicyProcessOptions
) => Promise<{ stderr: string; stdout: string }>

const POLICY_TIMEOUT_MS = 15_000
const POLICY_MAX_OUTPUT_BYTES = 64 * 1024
const POLICY_KEYS = ['backup_keep', 'mode', 'quick_keep', 'quick_max_file_size']

function positiveSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0
}

export function parsePreUpdateBackupPolicy(stdout: string): PreUpdateBackupPolicy {
  const lines = stdout.split(/\r?\n/).filter(line => line.length > 0)

  if (lines.length !== 1) {
    throw new Error('pre-update backup policy output must contain exactly one JSON line')
  }

  let payload: unknown

  try {
    payload = JSON.parse(lines[0])
  } catch (error) {
    throw new Error('pre-update backup policy output is not valid JSON', { cause: error })
  }

  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('pre-update backup policy payload must be an object')
  }

  const record = payload as Record<string, unknown>
  const keys = Object.keys(record).sort()

  if (keys.length !== POLICY_KEYS.length || keys.some((key, index) => key !== POLICY_KEYS[index])) {
    throw new Error('pre-update backup policy payload has an unexpected schema')
  }

  if (record.mode !== 'off' && record.mode !== 'quick' && record.mode !== 'full') {
    throw new Error('pre-update backup policy contains an invalid mode')
  }

  if (
    !positiveSafeInteger(record.backup_keep) ||
    !positiveSafeInteger(record.quick_keep) ||
    !positiveSafeInteger(record.quick_max_file_size)
  ) {
    throw new Error('pre-update backup policy contains an invalid positive integer')
  }

  return {
    backupKeep: record.backup_keep,
    mode: record.mode,
    quickKeep: record.quick_keep,
    quickMaxFileSize: record.quick_max_file_size
  }
}

const runPolicyProcess: RunPolicyProcess = (command, args, options) =>
  new Promise((resolve, reject) => {
    execFile(command, args, { ...options, encoding: 'utf8' }, (error, stdout, stderr) => {
      if (error) {
        reject(error)

        return
      }

      resolve({ stderr, stdout })
    })
  })

export async function resolvePreUpdateBackupPolicy(
  paths: { hermesHome: string; pythonPath: string; updateRoot: string },
  runProcess: RunPolicyProcess = runPolicyProcess
): Promise<PreUpdateBackupPolicy> {
  let output: { stderr: string; stdout: string }

  try {
    output = await runProcess(paths.pythonPath, ['-m', 'hermes_cli.update_preflight_policy'], {
      cwd: paths.updateRoot,
      env: {
        ...process.env,
        HERMES_HOME: paths.hermesHome,
        PYTHONUTF8: '1'
      },
      maxBuffer: POLICY_MAX_OUTPUT_BYTES,
      shell: false,
      timeout: POLICY_TIMEOUT_MS,
      windowsHide: true
    })
  } catch (error) {
    throw new Error('could not resolve pre-update backup policy with the managed Python runtime', {
      cause: error
    })
  }

  if (output.stderr.trim()) {
    throw new Error('pre-update backup policy process wrote unexpected stderr output')
  }

  return parsePreUpdateBackupPolicy(output.stdout)
}
