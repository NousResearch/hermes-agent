/**
 * One-time Desktop notice for plugins that import pre-decomposition module paths (PR #102117).
 *
 * The Python side (hermes_cli/plugin_compat.py) statically scans the user's enabled external plugins on
 * every CLI/gateway/TUI start and writes HERMES_HOME/.plugin-compat-report.json when any plugin imports a
 * path scheduled for removal on 2026-09-14 (deleting the file when none do). Desktop reads that file at
 * boot and shows ONE modal, then records the dismissal in userData so the same set of affected plugins is
 * never shown again. A *different* set (a new affected plugin, or the removal date passing so the plugins
 * are now disabled) is a new message and shows once more.
 *
 * Pure module: no Electron imports, so it is unit-testable; main.ts owns the dialog.
 */

import fs from 'fs'
import path from 'path'

export const REPORT_FILE = '.plugin-compat-report.json'
export const DISMISSED_FILE = 'plugin-compat-dismissed.json'

export interface PluginCompatHit {
  file: string
  line: number
  old: string
  new: string
}

export interface PluginCompatReport {
  removal_date: string
  in_effect: boolean
  written_at?: string
  plugins: Record<string, PluginCompatHit[]>
  lines: [string, string]
}

/** Stable identity of a report: which plugins, how many hits each, and whether removal is in effect. */
export function reportKey(report: PluginCompatReport): string {
  const parts = Object.keys(report.plugins)
    .sort()
    .map(name => `${name}:${report.plugins[name].length}`)

  return `${report.in_effect ? 'disabled' : 'pending'}|${parts.join(',')}`
}

export function readReport(hermesHome: string): PluginCompatReport | null {
  try {
    const raw = fs.readFileSync(path.join(hermesHome, REPORT_FILE), 'utf8')
    const parsed = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || !parsed.plugins || !Array.isArray(parsed.lines)) {
      return null
    }

    if (Object.keys(parsed.plugins).length === 0) {
      return null
    }

    return parsed as PluginCompatReport
  } catch {
    return null
  }
}

export function wasDismissed(userData: string, key: string): boolean {
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(userData, DISMISSED_FILE), 'utf8'))

    return Array.isArray(raw?.keys) && raw.keys.includes(key)
  } catch {
    return false
  }
}

export function recordDismissed(userData: string, key: string): void {
  const file = path.join(userData, DISMISSED_FILE)
  let keys: string[] = []

  try {
    const raw = JSON.parse(fs.readFileSync(file, 'utf8'))

    if (Array.isArray(raw?.keys)) {
      keys = raw.keys.filter((k: unknown) => typeof k === 'string')
    }
  } catch {
    /* first dismissal */
  }

  if (!keys.includes(key)) {
    keys.push(key)
  }

  fs.mkdirSync(userData, { recursive: true })
  fs.writeFileSync(file, JSON.stringify({ keys: keys.slice(-20) }, null, 2))
}

export interface PendingNotice {
  key: string
  title: string
  message: string
  detail: string
}

/** The modal to show this boot, or null (no report, or this exact report already dismissed). */
export function pendingNotice(hermesHome: string, userData: string): PendingNotice | null {
  const report = readReport(hermesHome)

  if (!report) {
    return null
  }

  const key = reportKey(report)

  if (wasDismissed(userData, key)) {
    return null
  }

  const names = Object.keys(report.plugins).sort()

  const list = names
    .map(n => {
      const hits = report.plugins[n]
      const first = hits[0]

      return `• ${n} — ${hits.length} import${hits.length === 1 ? '' : 's'} (e.g. ${first.old} → ${first.new})`
    })
    .join('\n')

  const title = report.in_effect ? 'Some plugins were not loaded' : 'Plugins need an update'

  const message = report.in_effect
    ? `${names.length} plugin${names.length === 1 ? '' : 's'} import${names.length === 1 ? 's' : ''} module paths that were removed on ${report.removal_date} and ${names.length === 1 ? 'was' : 'were'} not loaded.`
    : `${names.length} plugin${names.length === 1 ? '' : 's'} import${names.length === 1 ? 's' : ''} module paths that stop working on ${report.removal_date}.`

  const detail = report.in_effect
    ? `${list}\n\nUpdate the plugin(s), or force-load them with plugins.allow_deprecated_imports: true in config.yaml (they will still break once the compatibility layer is removed).\n\nFull list: hermes plugins compat`
    : `${list}\n\nCheck for plugin updates or notify the author before ${report.removal_date}. After that date these plugins are not loaded.\n\nFull list: hermes plugins compat`

  return { key, title, message, detail }
}
