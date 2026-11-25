/**
 * `ctx.preview` — a plugin's door into the preview Browser's guest page.
 *
 * The guest is reached through the same script runner drive_preview and the
 * preview tour use (`activePreviewScriptRunner`), never a scraped
 * `<webview>`. Because that page can be one the user is signed in to, the
 * door is closed by default and the APP arbitrates every call:
 *
 *  1. the plugin must declare `capabilities: ['preview.executeJavaScript']`
 *     on its default export (undeclared → refused, nothing asked or run);
 *  2. the page must be a real web origin (`http(s)://host[:port]`);
 *  3. the user must have allowed THIS plugin on THIS origin — the first call
 *     raises the app-owned consent dialog (plugin name + origin), "Allow"
 *     persists a grant revocable in Capabilities ▸ Plugins, "Don't allow"
 *     refuses for the rest of the app run;
 *  4. the script runs behind an in-page origin guard, so a navigation between
 *     the check and the run refuses instead of landing on another site.
 */

import { activePreviewScriptRunner } from '@/app/chat/right-rail/preview-script-runner'
import { ensurePreviewScriptConsent, isGrantableOrigin } from '@/store/plugin-preview-grants'

import { $pluginRecords } from './plugins-store'

/** The capability a plugin declares to use `ctx.preview.executeJavaScript`. */
export const PREVIEW_SCRIPT_CAPABILITY = 'preview.executeJavaScript'

export const PREVIEW_NOTHING_OPEN = 'No preview page is open.'

export interface PluginPreview {
  /** True when this plugin declared `preview.executeJavaScript` and this
   *  window has a live Browser page for the focused session. Says nothing
   *  about the user's consent — that is asked on the first call. */
  isAvailable: () => boolean
  /** Run `code` in the focused session's active preview Browser tab and
   *  resolve its completion value. Always a Promise; rejects when the
   *  capability is undeclared, no page is open, the page is not a web
   *  origin, the user did not allow it, or the page changed origin. */
  executeJavaScript: (code: string) => Promise<unknown>
}

const ORIGIN_PROBE = 'window.location.origin'

// A leading `'use strict'` directive must stay first or the script silently
// leaves strict mode, so the guard goes right after it. A newline ends the
// directive only when the next line cannot continue the expression
// (`'use strict'\n.toUpperCase()` is one expression, not a directive).
const STRICT_PROLOGUE = /^\s*(['"])use strict\1[ \t]*(?:;|\r?\n(?=\s*(?:\/[/*]|[^\s.([`+\-*/%<>=&|^?,]))|$)/

/** Inserts a guard statement; the plugin's script keeps its completion value. */
function originGuarded(origin: string, code: string): string {
  const guard = `if (window.location.origin !== ${JSON.stringify(origin)}) { throw new Error('The preview page changed origin; script not run.') }\n`
  const prologue = STRICT_PROLOGUE.exec(code)?.[0] ?? ''

  return `${prologue}${prologue ? '\n' : ''}${guard}${code.slice(prologue.length)}`
}

async function runPreviewScript(pluginId: string, declared: boolean, code: unknown): Promise<unknown> {
  if (typeof code !== 'string') {
    throw new Error('ctx.preview.executeJavaScript requires a string')
  }

  if (!declared) {
    throw new Error(`Plugin "${pluginId}" did not declare the ${PREVIEW_SCRIPT_CAPABILITY} capability.`)
  }

  // Bound to the tab resolved now: the origin the user is asked about and the
  // page the script lands on are the same tab even if focus moves meanwhile.
  const run = activePreviewScriptRunner()

  if (!run) {
    throw new Error(PREVIEW_NOTHING_OPEN)
  }

  const origin = await run(ORIGIN_PROBE)

  if (typeof origin !== 'string' || !isGrantableOrigin(origin)) {
    throw new Error('Preview scripts run only on a web page (http or https).')
  }

  const name = $pluginRecords.get()[pluginId]?.name ?? pluginId

  if (!(await ensurePreviewScriptConsent(pluginId, name, origin))) {
    throw new Error(`${name} is not allowed to run scripts on ${origin}.`)
  }

  return run(originGuarded(origin, code))
}

export function createPluginPreview(pluginId: string, capabilities: readonly string[] = []): PluginPreview {
  const declared = capabilities.includes(PREVIEW_SCRIPT_CAPABILITY)

  return {
    isAvailable: () => declared && Boolean(activePreviewScriptRunner()),
    executeJavaScript: code => runPreviewScript(pluginId, declared, code)
  }
}
