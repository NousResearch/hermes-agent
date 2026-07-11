/**
 * Minimal reader for the `desktop.render_cache.enabled` rollback flag
 * (Phase 1 of the desktop startup-latency work; spec §7 Rollback).
 *
 * Behavioral settings live in HERMES_HOME/config.yaml (repo rule: .env is for
 * secrets only). The desktop main process has no YAML dependency, and pulling
 * one in for a single boolean would be exactly the footprint the rubric tells
 * us to avoid — so this is a deliberately tiny indentation-scan for one nested
 * key. FAIL-OPEN: the feature is ON unless config.yaml explicitly says
 *
 *   desktop:
 *     render_cache:
 *       enabled: false
 *
 * Any read/parse problem returns true (feature on) — the flag exists as a
 * rollback lever, so a broken config must not silently disable the feature the
 * user didn't ask to disable... and equally must never crash boot (I3 energy).
 */

import fs from 'node:fs'
import path from 'node:path'

/** Parse the flag out of raw config.yaml text. Exported for tests. */
export function parseRenderCacheEnabled(yamlText: string): boolean {
  try {
    const lines = String(yamlText || '').split(/\r?\n/)
    let inDesktop = false
    let desktopIndent = -1
    let inRenderCache = false
    let renderCacheIndent = -1

    for (const rawLine of lines) {
      const line = rawLine.replace(/\t/g, '  ')
      const stripped = line.trim()
      if (!stripped || stripped.startsWith('#')) {
        continue
      }
      const indent = line.length - line.trimStart().length

      if (!inDesktop) {
        if (/^desktop:\s*(#.*)?$/.test(stripped) && indent === 0) {
          inDesktop = true
          desktopIndent = indent
        }
        continue
      }

      // Left the desktop block?
      if (indent <= desktopIndent) {
        if (inRenderCache) {
          break
        }
        inDesktop = false
        // The same line might START a new top-level desktop block (unlikely);
        // re-test it.
        if (/^desktop:\s*(#.*)?$/.test(stripped) && indent === 0) {
          inDesktop = true
        }
        continue
      }

      if (!inRenderCache) {
        if (/^render_cache:\s*(#.*)?$/.test(stripped)) {
          inRenderCache = true
          renderCacheIndent = indent
        }
        continue
      }

      // Inside render_cache; left it?
      if (indent <= renderCacheIndent) {
        break
      }

      const m = /^enabled:\s*(\S+)/.exec(stripped)
      if (m) {
        const v = m[1].toLowerCase().replace(/["']/g, '')
        return !(v === 'false' || v === 'no' || v === 'off' || v === '0')
      }
    }
  } catch {
    // fall through to default
  }
  return true
}

/** Read the flag from HERMES_HOME/config.yaml. Fail-open (true) on any error. */
export function readRenderCacheEnabled(hermesHome: string): boolean {
  try {
    const raw = fs.readFileSync(path.join(hermesHome, 'config.yaml'), 'utf8')
    return parseRenderCacheEnabled(raw)
  } catch {
    return true
  }
}
