/**
 * Watch ~/.hermes/prompt-templates-inbox for PE import jobs and apply live.
 * Side-effect module — import once from main window boot.
 */
import { applyPromptImportJob, parsePromptImportJob } from './prompt-import'

const INBOX_DIRNAME = 'prompt-templates-inbox'
const APPLIED_DIRNAME = 'applied'
const POLL_MS = 2000

let started = false
let busy = false
const seen = new Set<string>()

async function resolveInboxDir(): Promise<null | string> {
  const desktop = window.hermesDesktop

  if (!desktop?.desktopPluginsRoot) {
    return null
  }

  try {
    const pluginsRoot = await desktop.desktopPluginsRoot()
    // <HERMES_HOME>/desktop-plugins → parent is HERMES_HOME
    const home = pluginsRoot.replace(/[/\\]desktop-plugins[/\\]?$/, '')

    if (!home || home === pluginsRoot) {
      return null
    }

    return `${home.replace(/[/\\]$/, '')}/${INBOX_DIRNAME}`
  } catch {
    return null
  }
}

async function ensureDir(path: string): Promise<void> {
  // Do NOT use openDir — that mkdir+reveals in Finder. Touch a keep file instead.
  const desktop = window.hermesDesktop
  const keep = `${path.replace(/[/\\]$/, '')}/.keep`

  try {
    if (desktop?.writeTextFile) {
      await desktop.writeTextFile(keep, '')
    }
  } catch {
    // best-effort
  }
}

async function listJsonJobs(dir: string): Promise<string[]> {
  const desktop = window.hermesDesktop

  if (!desktop?.readDir) {
    return []
  }

  try {
    const listing = await desktop.readDir(dir)
    const entries = listing.entries

    if (!Array.isArray(entries)) {
      return []
    }

    return entries
      .filter(e => !e.isDirectory && e.name.endsWith('.json') && !e.name.startsWith('.'))
      .map(e => e.path || `${dir.replace(/[/\\]$/, '')}/${e.name}`)
  } catch {
    return []
  }
}

async function readJob(path: string): Promise<unknown | null> {
  const desktop = window.hermesDesktop

  if (!desktop?.readFileText) {
    return null
  }

  try {
    const result = await desktop.readFileText(path)
    const text = typeof result === 'string' ? result : result?.text

    if (!text || (typeof result === 'object' && result && 'truncated' in result && result.truncated)) {
      return null
    }

    return JSON.parse(text) as unknown
  } catch {
    return null
  }
}

async function archiveJob(path: string, inboxDir: string, ok: boolean): Promise<void> {
  const desktop = window.hermesDesktop
  const base = path.split(/[/\\]/).pop() || `job-${Date.now()}.json`
  const appliedDir = `${inboxDir.replace(/[/\\]$/, '')}/${APPLIED_DIRNAME}`
  await ensureDir(appliedDir)
  const dest = `${appliedDir}/${ok ? '' : 'failed-'}${Date.now()}-${base}`

  try {
    // Prefer rename if available; else write+trash
    if (desktop?.renamePath) {
      // renamePath only renames basename in place — need move. Use read+write+trash.
    }

    const raw = await readJob(path)

    if (raw !== null && desktop?.writeTextFile) {
      await desktop.writeTextFile(dest, JSON.stringify(raw, null, 2))
    }

    if (desktop?.trashPath) {
      await desktop.trashPath(path)
    }
  } catch {
    // best-effort archive
  }
}

async function processInboxOnce(inboxDir: string): Promise<void> {
  if (busy) {
    return
  }

  busy = true

  try {
    const jobs = await listJsonJobs(inboxDir)

    for (const path of jobs) {
      if (seen.has(path)) {
        continue
      }

      seen.add(path)
      const raw = await readJob(path)
      const job = parsePromptImportJob(raw)

      if (!job) {
        await archiveJob(path, inboxDir, false)
        continue
      }

      try {
        const result = applyPromptImportJob(job)
        console.info('[prompt-import-inbox]', path, result)
        await archiveJob(path, inboxDir, true)
      } catch (err) {
        console.warn('[prompt-import-inbox] apply failed', path, err)
        await archiveJob(path, inboxDir, false)
      }
    }
  } finally {
    busy = false
  }
}

/** Start watching the inbox. Safe to call multiple times. */
export function startPromptImportInbox(): void {
  if (started) {
    return
  }

  // Overlay / quick windows skip.
  const winParam = new URLSearchParams(window.location.search).get('win')

  if (winParam && winParam !== '') {
    return
  }

  started = true

  void (async () => {
    const inboxDir = await resolveInboxDir()

    if (!inboxDir) {
      console.warn('[prompt-import-inbox] no HERMES_HOME; inbox disabled')
      return
    }

    await ensureDir(inboxDir)
    await ensureDir(`${inboxDir}/${APPLIED_DIRNAME}`)

    const tick = () => {
      void processInboxOnce(inboxDir)
    }

    tick()

    const desktop = window.hermesDesktop
    let watched = false

    if (desktop?.watchDirectory && desktop.onPreviewFileChanged) {
      try {
        const watch = await desktop.watchDirectory(inboxDir)
        desktop.onPreviewFileChanged(payload => {
          if (payload.id === watch.id) {
            tick()
          }
        })
        watched = true
      } catch {
        watched = false
      }
    }

    if (!watched) {
      window.setInterval(tick, POLL_MS)
    } else {
      // light poll backup in case watch misses atomic replaces
      window.setInterval(tick, POLL_MS * 5)
    }

    console.info('[prompt-import-inbox] watching', inboxDir)
  })()
}
