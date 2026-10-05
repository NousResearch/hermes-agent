import fs from 'node:fs'
import path from 'node:path'

export interface BackgroundWorkItem {
  key: string
  title: string
  worker: string
  profile: string
  model: string
  provider: string
  pid: number
  started_at: number
  state: 'running'
  elapsed_seconds: number
  elapsed: string
}

export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))

  if (total < 60) {
    return `${total}s`
  }

  const minutes = Math.floor(total / 60)
  const secs = total % 60

  if (minutes < 60) {
    return `${minutes}m ${String(secs).padStart(2, '0')}s`
  }

  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

export function activityHomes(hermesHome: string): string[] {
  const resolved = path.resolve(hermesHome)
  const root = path.basename(path.dirname(resolved)) === 'profiles' ? path.dirname(path.dirname(resolved)) : resolved
  const homes = new Set([root, resolved])
  const profiles = path.join(root, 'profiles')

  try {
    for (const entry of fs.readdirSync(profiles, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        homes.add(path.join(profiles, entry.name))
      }
    }
  } catch {
    void 0
  }

  return [...homes]
}

function processAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    process.kill(pid, 0)

    return true
  } catch {
    return false
  }
}

export function readBackgroundWork(hermesHome: string, nowSeconds = Date.now() / 1000): BackgroundWorkItem[] {
  const items = new Map<string, BackgroundWorkItem>()

  for (const home of activityHomes(hermesHome)) {
    const directory = path.join(home, 'cache', 'background-work')
    let names: string[] = []

    try {
      names = fs.readdirSync(directory).filter(name => name.endsWith('.json'))
    } catch {
      continue
    }

    for (const name of names) {
      const leasePath = path.join(directory, name)

      try {
        const raw = JSON.parse(fs.readFileSync(leasePath, 'utf8')) as Partial<BackgroundWorkItem>
        const pid = Number(raw.pid)

        if (raw.state !== 'running' || !processAlive(pid)) {
          fs.rmSync(leasePath, { force: true })

          continue
        }

        const startedAt = Number(raw.started_at) || nowSeconds
        const elapsedSeconds = Math.max(0, Math.floor(nowSeconds - startedAt))

        const item: BackgroundWorkItem = {
          key: String(raw.key || name),
          title: String(raw.title || 'Background worker'),
          worker: String(raw.worker || 'worker'),
          profile: String(raw.profile || 'default'),
          model: String(raw.model || ''),
          provider: String(raw.provider || ''),
          pid,
          started_at: startedAt,
          state: 'running',
          elapsed_seconds: elapsedSeconds,
          elapsed: formatElapsed(elapsedSeconds)
        }

        items.set(`${pid}:${item.key}`, item)
      } catch {
        try {
          fs.rmSync(leasePath, { force: true })
        } catch {
          void 0
        }
      }
    }
  }

  return [...items.values()].sort((a, b) => a.started_at - b.started_at || a.key.localeCompare(b.key))
}
