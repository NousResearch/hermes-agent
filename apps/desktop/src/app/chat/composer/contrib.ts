/**
 * Composer contribution surface — every seam of the composer is hook-into-able
 * through the SAME registry schema as every other surface (statusbar, titlebar,
 * panes, layouts):
 *
 *   render areas (`render`):  composer.top       — banner strip above the input
 *                             composer.bottom    — row below the input grid
 *                             composer.underside — floating strip BELOW the
 *                                                  whole composer (no chrome)
 *                             composer.leading   — inline after the "+" menu
 *                             composer.actions   — inline before the model pill
 *
 *   data kinds (`data`):      composer.middleware    (ComposerMiddleware)
 *                             composer.attachments   (ComposerAttachmentProvider)
 *                             composer.microActions  (ComposerMicroActionProvider)
 *
 * Core keeps ownership of the transcript, input, and submit engine — these
 * seams AUGMENT the composer, they never replace it. Middleware runs as an
 * ordered async chain around the app's onSubmit: each handler may rewrite the
 * draft, pass it through, or cancel the send by returning null.
 */

import { useMemo } from 'react'
import type { RefObject } from 'react'

import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'
import type { TodoItem } from '@/lib/todos'
import type { ComposerAttachment } from '@/store/composer'
import type { ComposerAction } from '@/store/composer-actions'

import { composerInsertWouldChange, insertComposerContentsAtCaret } from './rich-editor'

export const COMPOSER_AREAS = {
  top: 'composer.top',
  bottom: 'composer.bottom',
  underside: 'composer.underside',
  leading: 'composer.leading',
  actions: 'composer.actions',
  middleware: 'composer.middleware',
  attachments: 'composer.attachments',
  microActions: 'composer.microActions'
} as const

export type ComposerRenderArea =
  | typeof COMPOSER_AREAS.actions
  | typeof COMPOSER_AREAS.bottom
  | typeof COMPOSER_AREAS.leading
  | typeof COMPOSER_AREAS.underside
  | typeof COMPOSER_AREAS.top

export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
}

/** Payload of a `composer.middleware` data contribution. */
export interface ComposerMiddleware {
  /** Rewrite (return a draft), pass through (same draft), or cancel (null). */
  handler: (draft: ComposerDraft) => ComposerDraft | null | Promise<ComposerDraft | null>
}

export interface ComposerAttachmentContext {
  insertText: (text: string) => void
}

/**
 * Edit bridge handed to composer RENDER-area contributions (`composer.actions`,
 * `composer.top`, `composer.bottom`, `composer.leading`, `composer.underside`)
 * as the first argument of their `render` function.
 *
 * Backed by the composer's own undo stack and DOM pipeline — an insert here
 * lands on the same ⌘Z history as typing, replaces the live selection, and
 * hydrates directives into chips exactly like a paste. Plugins must NOT reach
 * for `execCommand` or `innerHTML`; that bypasses both the app's undo stack and
 * its chip/IME safety.
 */
export interface ComposerRenderContext {
  /**
   * Insert `text` at the caret, replacing any selection inside the composer
   * editor. The pre-edit state is banked on the app undo stack first, so a
   * single ⌘Z reverts the insert. Directives in the text (`@url:…`, `/…`)
   * hydrate into chips through the app's own rendering pipeline.
   *
   * True no-op — nothing banked, redo preserved — only when the replacement
   * would leave the serialized text AND DOM structure unchanged (empty insert
   * at a collapsed caret, or text identical to what is already there).
   * Replacing a non-collapsed selection with `''` deletes the selection, and a
   * text-identical replacement that changes structure (plain `@kind:value`
   * hydrating into a chip) is a real edit that banks its own undo point.
   * Also a no-op while the editor is unmounted or during IME composition.
   * Multi-insert sequences bank one undo point per call — bundle a wrapping
   * edit into ONE `insertText` call (the full replacement string) so undo
   * reverts the whole action at once.
   */
  insertText: (text: string) => void
}

export interface CreateComposerRenderContextArgs {
  /** The rich-editor contentEditable, live. */
  editorRef: RefObject<HTMLDivElement | null>
  /** True while an IME preedit is being composed (see `composingRef` in
   *  index.tsx). Inserts are suppressed during composition — landing DOM edits
   *  inside a live preedit corrupts it (the execCommand bug class). */
  composingRef: RefObject<boolean>
  /** Bank the pre-edit state on the app undo stack. */
  recordUndoPoint: () => void
  /** Flush editor DOM → draft state after the edit (rAF-coalesced, like the
   *  paste path). */
  scheduleFlushEditorToDraft: (editor: HTMLDivElement) => void
}

/**
 * Build the composer's edit bridge for render-area contributions. Mirrors the
 * app's own paste path exactly — bank the pre-edit state on the app undo
 * stack, mutate through the app's own DOM pipeline (`insertComposerContentsAtCaret`:
 * Range-based, chips hydrate, never `execCommand`/`innerHTML`), then flush the
 * draft state. Selection-aware by construction: the insert replaces whatever
 * range the document selection holds inside the editor (a plugin's popover
 * click does not destroy the editor's selection), else lands at the caret.
 */
export function createComposerRenderContext({
  editorRef,
  composingRef,
  recordUndoPoint,
  scheduleFlushEditorToDraft
}: CreateComposerRenderContextArgs): ComposerRenderContext {
  return {
    insertText: (text: string) => {
      const editor = editorRef.current

      // No editor, or an IME preedit in flight — inserting would corrupt it.
      if (!editor || composingRef.current) {
        return
      }

      if (!composerInsertWouldChange(editor, text)) {
        return
      }

      recordUndoPoint()
      insertComposerContentsAtCaret(editor, text)
      scheduleFlushEditorToDraft(editor)
    }
  }
}


/** Payload of a `composer.attachments` data contribution — an entry in the
 *  composer's "+" attach menu. */
export interface ComposerAttachmentProvider {
  label: string
  /** Codicon name for the menu row. Defaults to `plug`. */
  icon?: string
  run: (ctx: ComposerAttachmentContext) => void | Promise<void>
}

/**
 * Run the ordered middleware chain over a draft. Contributions execute in
 * registry order (`order`, then registration order); the first `null` wins
 * and cancels the send. A throwing handler is treated as pass-through so a
 * broken plugin can't eat messages.
 */
export async function runComposerMiddleware(draft: ComposerDraft): Promise<ComposerDraft | null> {
  let current = draft

  for (const contribution of registry.getArea(COMPOSER_AREAS.middleware)) {
    const middleware = contribution.data as ComposerMiddleware | undefined

    if (!middleware?.handler) {
      continue
    }

    try {
      const next = await middleware.handler(current)

      if (next === null) {
        return null
      }

      current = next
    } catch {
      // Pass-through: a faulty middleware must never swallow the message.
    }
  }

  return current
}

/** Attach-menu entries contributed by plugins/core, with stable render keys. */
export function useComposerAttachmentProviders(): Array<ComposerAttachmentProvider & { key: string }> {
  return useContributions(COMPOSER_AREAS.attachments)
    .map(c => ({ key: `${c.source ?? 'core'}:${c.id}`, ...(c.data as ComposerAttachmentProvider) }))
    .filter(p => Boolean(p.label && p.run))
}

/**
 * Payload of a `composer.microActions` data contribution — the pill strip at
 * the top of the composer's overlay lane.
 *
 * `resolve` is called with the live session context and returns the badges to
 * show right now, or `[]` for "nothing from me". Returning a list rather than
 * a static badge is what lets a provider be conditional ("only while idle",
 * "only with unfinished tasks") without a reactive `when()`, which the
 * registry deliberately doesn't offer.
 */
export interface ComposerMicroActionProvider {
  resolve: (ctx: ComposerMicroActionContext) => ComposerAction[]
}

/** What a micro-action provider gets to branch on. Deliberately small: every
 *  field here is a standing compatibility promise to the plugins using it. */
export interface ComposerMicroActionContext {
  /** A turn is currently running in this session. */
  busy: boolean
  sessionId: string
  /** Live todo list for the session (empty when there is none). */
  todos: readonly TodoItem[]
}

/** Micro-action providers, memoised against the registry's own stable
 *  snapshot — the strip re-resolves on every composer render, so a fresh array
 *  here would defeat that. */
export function useComposerMicroActionProviders(): ComposerMicroActionProvider[] {
  const contributions = useContributions(COMPOSER_AREAS.microActions)

  return useMemo(
    () => contributions.map(c => c.data as ComposerMicroActionProvider).filter(p => typeof p?.resolve === 'function'),
    [contributions]
  )
}
