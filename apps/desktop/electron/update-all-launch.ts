import fs from 'node:fs'
import path from 'node:path'

import { isPidAliveWindows } from './backend-release-gate'

const SWITCH = '--hermes-update-all-request='

interface LaunchRequest {
  version: 1
  id: string
  state: 'pending' | 'accepted' | 'committed' | 'coalesced' | 'interrupted'
  launcher_pids: number[]
  expires_at: number
}

/** Consume rather than retain the one-shot switch in app.relaunch() argv. */
export function takeUpdateAllLaunchArg(argv: string[]): string | null {
  const requests = argv.filter(arg => arg.startsWith(SWITCH))

  for (let i = argv.length - 1; i >= 0; i--) {
    if (argv[i].startsWith(SWITCH)) {
      argv.splice(i, 1)
    }
  }

  return requests.length === 1 ? requests[0].slice(SWITCH.length) : null
}

function readRequest(file: string): LaunchRequest {
  if (
    !path.isAbsolute(file) ||
    path.basename(file) !== 'request.json' ||
    !/^hermes-update-all-[\w-]+$/.test(path.basename(path.dirname(file)))
  ) {
    throw new Error('Invalid Desktop update request path')
  }

  const directory = fs.lstatSync(path.dirname(file))
  const stat = fs.lstatSync(file)

  if (
    !directory.isDirectory() ||
    !stat.isFile() ||
    stat.size > 4096 ||
    (process.platform !== 'win32' && (directory.uid !== process.getuid!() || (directory.mode & 0o077) !== 0))
  ) {
    throw new Error('Desktop update request must be a private regular file')
  }

  const value = JSON.parse(fs.readFileSync(file, 'utf8')) as LaunchRequest

  if (
    value.version !== 1 ||
    !/^[a-f0-9]{32}$/.test(value.id) ||
    !Number.isFinite(value.expires_at) ||
    value.expires_at < Date.now() ||
    value.expires_at > Date.now() + 60_000 ||
    !Array.isArray(value.launcher_pids) ||
    !value.launcher_pids.length ||
    value.launcher_pids.length > 2 ||
    !value.launcher_pids.every(pid => Number.isSafeInteger(pid) && pid > 0)
  ) {
    throw new Error('Invalid or expired Desktop update request')
  }

  return value
}

function acknowledge(file: string, request: LaunchRequest, state: 'accepted' | 'coalesced' | 'interrupted'): void {
  const staged = file.replace(/\.json$/, '.tmp')
  fs.writeFileSync(staged, JSON.stringify({ ...request, state }), { mode: 0o600 })
  fs.renameSync(staged, file)
}

interface LaunchDependencies {
  deliver: (id: string) => void
  reportError: (error: unknown) => void
  isPidAlive?: (pid: number) => boolean
}

/** Only queues the local launch intent. The renderer and native updater retain
 * their existing responsibilities; no connection or update state is copied. */
export function createUpdateAllLaunch(deps: LaunchDependencies) {
  const isPidAlive = deps.isPidAlive ?? isPidAliveWindows
  let ready = false
  let phase: 'idle' | 'launcher' | 'queued' | 'running' | 'interrupted' = 'idle'
  let activeId: string | null = null

  const flush = () => {
    if (ready && phase === 'queued' && activeId) {
      phase = 'running'
      deps.deliver(activeId)
    }
  }

  return {
    async request(file: string): Promise<void> {
      let owned = false

      try {
        const request = readRequest(file)

        if (request.state !== 'pending') {
          return
        }

        if (phase === 'interrupted') {
          acknowledge(file, request, 'interrupted')

          return
        }

        if (phase !== 'idle') {
          acknowledge(file, request, 'coalesced')

          return
        }

        activeId = request.id
        phase = 'launcher'
        owned = true
        acknowledge(file, request, 'accepted')

        while (Date.now() < request.expires_at) {
          const current = readRequest(file)

          if (current.id !== request.id) {
            throw new Error('Desktop update request identity changed')
          }

          if (current.state === 'committed' && !request.launcher_pids.some(isPidAlive)) {
            fs.unlinkSync(file)
            fs.rmdirSync(path.dirname(file))
            phase = 'queued'
            flush()

            return
          }

          await new Promise(resolve => setTimeout(resolve, 50))
        }

        throw new Error('The update launcher did not exit; update-all was not started')
      } catch (error) {
        if (owned) {
          phase = 'idle'
          activeId = null
        }

        deps.reportError(error)
      }
    },
    ready() {
      ready = true
      flush()
    },
    rendererGone() {
      // Never replay a delivered request after a renderer reload.
      ready = false

      if (phase === 'running') {
        phase = 'interrupted'
      }
    },
    complete(id: string) {
      if ((phase === 'running' || phase === 'interrupted') && id === activeId) {
        phase = 'idle'
        activeId = null
      }
    }
  }
}
