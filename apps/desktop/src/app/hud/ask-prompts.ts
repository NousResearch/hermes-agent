/**
 * The four things the ask sheet can do with what is under the cursor, and
 * the prompt each one sends. Pure, so the wording is tested and the sheet
 * stays a thin surface over it.
 *
 * Every prompt names the app and window when main could identify them, so
 * the model has the same context the user has — and it goes out with
 * `surface: 'hud'`, which the prompt builder already turns into "this means
 * the app under the HUD" (agent/prompt_builder.py, hud_surface_note).
 */

import type { HudAskPayload } from '@/lib/hud-prefs'

export type HudAskAction = 'ask' | 'do' | 'explain' | 'summarize'

export const HUD_ASK_ACTIONS: readonly HudAskAction[] = ['explain', 'summarize', 'do', 'ask']

/** "Figma — Untitled" / "Figma" / "" — the app and title, when known. */
export function hudAskSource(payload: Pick<HudAskPayload, 'app' | 'title'>): string {
  const app = payload.app.trim()
  const title = payload.title.trim()

  if (app && title && title !== app) {
    return `${app} — ${title}`
  }

  return app || title
}

function where(payload: Pick<HudAskPayload, 'app' | 'title' | 'imagePath'>): string {
  const source = hudAskSource(payload)
  const picture = payload.imagePath ? 'the attached screenshot' : 'what is under the HUD'

  return source ? `${picture} (from ${source})` : picture
}

export function hudAskPrompt(action: Exclude<HudAskAction, 'ask'>, payload: HudAskPayload): string {
  const target = where(payload)

  switch (action) {
    case 'explain':
      return `Explain what I am looking at in ${target}. Be concise and concrete.`

    case 'summarize':
      return `Summarize the content shown in ${target}. Lead with the main point.`

    case 'do':
      return (
        `Look at ${target}. Work out what I am most likely trying to do here and do it, ` +
        'working in that app. Ask before anything irreversible.'
      )
  }
}
