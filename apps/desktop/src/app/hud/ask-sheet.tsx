/**
 * The ask sheet — "what do you want to do with this?" — shown in the HUD
 * after the ask chord (or Ctrl + right-click) captured what was under the
 * OS cursor. Four verbs; three of them attach the crop and submit through
 * the HUD's REAL composer (so the turn carries `surface: 'hud'` and the
 * image rides the ordinary attachment path), the fourth attaches and hands
 * the composer over for a custom question.
 */

import { useStore } from '@nanostores/react'
import type { KeyboardEvent } from 'react'

import { requestComposerFocus, requestComposerInsert, requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { attachmentId } from '@/lib/chat-runtime'
import type { HudAskPayload } from '@/lib/hud-prefs'
import { addComposerAttachment, createComposerAttachmentOccurrenceId } from '@/store/composer'
import { $hudAsk, dismissHudAsk } from '@/store/hud'

import { HUD_ASK_ACTIONS, type HudAskAction, hudAskPrompt, hudAskSource } from './ask-prompts'

/** Attach the crop exactly like a pasted screenshot: a local image path with
 *  the small thumbnail main already made, so nothing is re-read here. */
export function attachHudAskCrop(payload: HudAskPayload, label: string): boolean {
  if (!payload.imagePath) {
    return false
  }

  addComposerAttachment({
    id: attachmentId('image', payload.imagePath),
    occurrenceId: createComposerAttachmentOccurrenceId(),
    kind: 'image',
    label,
    detail: payload.imagePath,
    path: payload.imagePath,
    ...(payload.thumbnail ? { thumbnailUrl: payload.thumbnail } : {})
  })

  return true
}

/**
 * Run one verb. Returns true when a prompt was submitted; false when the
 * composer took the text/attachment but the user still has to send (the
 * "Ask…" verb, or a composer that was not ready to submit — in which case the
 * prompt is inserted rather than lost).
 */
export function runHudAskAction(action: HudAskAction, payload: HudAskPayload, label: string): boolean {
  attachHudAskCrop(payload, label)

  if (action === 'ask') {
    requestComposerFocus('main')

    return false
  }

  const text = hudAskPrompt(action, payload)

  if (requestComposerSubmit(text, { target: 'main' })) {
    return true
  }

  requestComposerInsert(text, { target: 'main' })
  requestComposerFocus('main')

  return false
}

export function HudAskSheet() {
  const ask = useStore($hudAsk)
  const { t } = useI18n()
  const h = t.hud

  if (!ask) {
    return null
  }

  const source = hudAskSource(ask)
  const labels: Record<HudAskAction, string> = { explain: h.explain, summarize: h.summarize, do: h.doIt, ask: h.ask }

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // The sheet owns Escape while it is up; the HUD's own Escape (put the
    // HUD away) is the next press.
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      dismissHudAsk()
    }
  }

  const run = (action: HudAskAction) => {
    runHudAskAction(action, ask, h.screenAtCursor)
    dismissHudAsk()
  }

  return (
    <div
      aria-label={h.askTitle}
      className="absolute inset-x-3 top-[calc(var(--hud-top-inset,0px)+var(--hud-bar-height,56px)+8px)] z-20 flex gap-3 rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-3 shadow-lg"
      data-hud-ask
      onKeyDown={onKeyDown}
      role="dialog"
    >
      {ask.thumbnail ? (
        <img alt="" className="h-20 w-32 shrink-0 rounded border border-(--ui-stroke-secondary) object-cover" src={ask.thumbnail} />
      ) : null}
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="truncate text-[0.625rem] font-medium tracking-wide text-(--ui-text-tertiary) uppercase">
          {ask.via === 'right-click' ? h.viaRightClick : h.viaShortcut}
          {source ? ` · ${source}` : ''}
        </div>
        <div className="text-sm text-(--ui-text-primary)">{h.askTitle}</div>
        <div className="flex flex-wrap items-center gap-1.5">
          {HUD_ASK_ACTIONS.map((action, index) => (
            <Button
              autoFocus={index === 0}
              key={action}
              onClick={() => run(action)}
              size="sm"
              type="button"
              variant={action === 'do' ? 'default' : 'secondary'}
            >
              {labels[action]}
            </Button>
          ))}
          <Button aria-label={h.dismiss} onClick={dismissHudAsk} size="sm" type="button" variant="ghost">
            {h.dismiss}
          </Button>
        </div>
      </div>
    </div>
  )
}
