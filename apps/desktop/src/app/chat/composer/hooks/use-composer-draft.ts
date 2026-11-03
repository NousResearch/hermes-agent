import { useAui, useAuiState, useComposerRuntime } from '@assistant-ui/react'
import { SLASH_COMMAND_RE } from '@hermes/shared'
import { type RefObject, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'
import { translateNow } from '@/i18n'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import { isTouchInteraction } from '@/lib/touch-interaction'
import { useStoreSelector } from '@/lib/use-session-slice'
import {
  adoptGoneSessionDraft,
  adoptNewSessionDraft,
  type ComposerAttachment,
  type ComposerDraftSyncMode,
  freshDraftScope,
  isFreshDraftScope,
  onComposerDraftSyncRequest,
  reloadPersistedDrafts,
  savedDraftAt,
  stashSessionDraft,
  takeSessionDraft
} from '@/store/composer'
import {
  type ComposerDraftRetry,
  draftRetryTextMatches,
  loadedDraftRetry,
  matchingDraftRetry,
  readDraftRetry,
  retryAttachmentIds,
  writeDraftRetry
} from '@/store/composer-draft-retry'
import { isBrowsingHistory } from '@/store/composer-input-history'
import { adoptInterruptedSend, type InterruptedSendHandlers, restoreInterruptedSend } from '@/store/composer-outbox'
import { $composerPopout } from '@/store/composer-popout'
import { migrateQueuedPrompts } from '@/store/composer-queue'
import { notify } from '@/store/notifications'

import { type ComposerStoreWriter, createComposerStoreWriter } from '../composer-store-writer'
import {
  cloneAttachments,
  DRAFT_PERSIST_DEBOUNCE_MS,
  isPendingDraftPersistCurrent,
  type QueueEditState
} from '../composer-utils'
import {
  ackComposerInsert,
  type ComposerInsertMode,
  focusComposerInput,
  getActiveComposer,
  markActiveComposer,
  onComposerDraftRequests,
  onComposerFocusRequest,
  onComposerInsertRefsRequest,
  onComposerInsertRequest,
  releaseActiveComposer
} from '../focus'
import { type InlineRefInput, insertInlineRefsIntoEditor } from '../inline-refs'
import {
  caretOffsetInEditor,
  composerPlainText,
  normalizeComposerEditorDom,
  placeCaretAtOffset,
  placeCaretEnd,
  REF_RE,
  renderComposerContents
} from '../rich-editor'
import { useComposerScope } from '../scope'
import type { ChatBarProps } from '../types'
import { useComposerVisible } from '../visibility'

/** A draft as this window last loaded or stashed it; attachments by occurrence. */
interface SavedDraft {
  attachments: string[]
  text: string
}

const attachmentOccurrence = (attachment: ComposerAttachment) => attachment.occurrenceId ?? attachment.id

const savedDraft = (text: string, attachments: ComposerAttachment[]): SavedDraft => ({
  attachments: attachments.map(attachmentOccurrence),
  text
})

const isSavedDraft = (saved: SavedDraft, text: string, attachments: ComposerAttachment[]) =>
  text === saved.text &&
  attachments.length === saved.attachments.length &&
  attachments.every((attachment, index) => attachmentOccurrence(attachment) === saved.attachments[index])

interface UseComposerDraftArgs {
  activeQueueSessionKey: string | null
  focusKey: ChatBarProps['focusKey']
  inputDisabled: boolean
  queueEditRef: RefObject<QueueEditState | null>
  sessionId: string | null | undefined
}

// A send restored after its page died comes back as its words: attachment bytes do not outlive the
// page. Say so, or the user takes the restored draft for the whole message and sends it without them.
function noteAttachmentsNotRestored(retry: ComposerDraftRetry | undefined) {
  if (retry?.attachmentIds.length) {
    notify({
      id: 'restored-draft-attachments',
      kind: 'warning',
      message: translateNow('promptNotices.restoredDraftAttachmentsDropped'),
      // Raised while the chat opens: opening a chat clears the last chat's notices right after.
      persistent: true
    })
  }
}

/**
 * The composer's draft engine — the detached source-of-truth spine. The live
 * text lives in the contentEditable DOM + `draftRef`; React only sees coarse
 * edge selectors, so typing never re-renders the chrome. Owns the imperative
 * composer-runtime subscription (draftRef mirror + external repaint + debounced
 * per-session stash), the edit primitives (append/insert/inline-refs), focus,
 * and per-session load/clear/stash/restore. The contentEditable *event*
 * handlers stay in ChatBar (they bridge into the trigger engine) and drive the
 * primitives exposed here.
 */
export function useComposerDraft({
  activeQueueSessionKey,
  focusKey,
  inputDisabled,
  queueEditRef,
  sessionId
}: UseComposerDraftArgs) {
  const aui = useAui()
  const composerRuntime = useComposerRuntime()
  const paneVisible = useComposerVisible()
  const visibleRef = useRef(paneVisible)
  visibleRef.current = paneVisible
  const floating = useStoreSelector($composerPopout, state => state.poppedOut)
  // Which composer this is on the focus bus + which attachment set it owns.
  const { attachments: attachmentScope, target } = useComposerScope()

  // Coarse edges only — these flip rarely (empty↔non-empty, the `?` help sigil,
  // steerable-vs-slash), so typing within a line costs no render.
  const hasText = useAuiState(s => s.composer.text.trim().length > 0)
  const isHelpHint = useAuiState(s => s.composer.text === '?')

  const isSteerableText = useAuiState(s => {
    const trimmed = s.composer.text.trim()

    return trimmed.length > 0 && !SLASH_COMMAND_RE.test(trimmed)
  })

  // assistant-ui's composer mutators throw when the core isn't bound yet (a
  // startup/thread-swap window); the DOM + draftRef hold the text and the
  // subscription reconciles once it binds, so swallow the premature write.
  const auiRef = useRef(aui)
  auiRef.current = aui
  const storeWriterRef = useRef<ComposerStoreWriter | null>(null)

  storeWriterRef.current ??= createComposerStoreWriter(value => {
    try {
      auiRef.current.composer().setText(value)
    } catch {
      // Composer core not bound yet — DOM/draftRef carry the text.
    }
  })

  const storeWriter = storeWriterRef.current

  const setComposerText = useCallback((value: string) => storeWriter.now(value), [storeWriter])

  useEffect(() => () => storeWriter.cancel(), [storeWriter])

  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef('')
  // The uncertain send this draft was restored from, while the text is still exactly that send.
  const draftRetryRef = useRef<ComposerDraftRetry | undefined>(undefined)
  const pendingDraftPersistRef = useRef<{ scope: string | null; text: string } | null>(null)
  const draftPersistTimerRef = useRef<number | undefined>(undefined)
  const activeQueueSessionKeyRef = useRef(activeQueueSessionKey)
  activeQueueSessionKeyRef.current = activeQueueSessionKey
  // Owned only by the swap effect below — unlike activeQueueSessionKeyRef this
  // does NOT update on every render, so it always reflects the session whose
  // text is actually loaded in the editor. Async work (debounce timers,
  // pagehide flush) must persist against this, not the render-time ref, or a
  // session switch mid-flight files one session's draft under another's key
  // (#54527).
  const draftScopeRef = useRef(activeQueueSessionKey)
  // The draft (text + attachment occurrences) this window last loaded from or
  // stashed for draftScopeRef's scope. Another window can send (clear) or
  // rewrite that draft while this editor still shows it, so every save path
  // stashes only state that differs from this baseline: this window's own edits.
  const savedDraftRef = useRef<SavedDraft>({ attachments: [], text: '' })
  const sessionIdRef = useRef(sessionId)
  sessionIdRef.current = sessionId
  const queueEditStateRef = useRef<QueueEditState | null>(queueEditRef.current)
  queueEditStateRef.current = queueEditRef.current

  const [focusRequestId, setFocusRequestId] = useState(0)

  const focusInput = useCallback(() => {
    const editor = editorRef.current

    if (!visibleRef.current || (editor && (!editor.isConnected || isElementInHiddenPane(editor)))) {
      return
    }

    focusComposerInput(editorRef.current)
    markActiveComposer(target)
  }, [target])

  const requestMainFocus = useCallback(() => {
    setFocusRequestId(id => id + 1)
  }, [])

  // The single write path for programmatic draft mutations: mirror → AUI state →
  // repaint the editor (caret to end). Repaints even while focused — inserts /
  // restores run mid-focus, and the runtime sync only repaints an unfocused
  // editor — so the visible text never lags the store.
  const paintDraft = useCallback(
    (next: string, focus = true, preserveFocusedCaret = false) => {
      draftRef.current = next
      setComposerText(next)

      const editor = editorRef.current

      if (editor) {
        // A durable-scope rekey can repaint the SAME unsent draft while this
        // editor remains the browser's active element and another composer
        // temporarily owns the focus-routing bus. Preserve that native caret
        // across the text-node replacement, but only for an unchanged draft:
        // programmatic inserts/restores keep their existing "caret to end"
        // semantics.
        const caretOffset =
          preserveFocusedCaret &&
          visibleRef.current &&
          document.activeElement === editor &&
          !isElementInHiddenPane(editor) &&
          composerPlainText(editor) === next
            ? caretOffsetInEditor(editor)
            : null

        renderComposerContents(editor, next, { trailingCommitted: true })

        if (caretOffset !== null) {
          placeCaretAtOffset(editor, Math.min(caretOffset, composerPlainText(editor).length))
        } else if (visibleRef.current && getActiveComposer() === target && !isElementInHiddenPane(editor)) {
          // Selection is document-global: a keep-alive composer in a hidden tab
          // may repaint when its background session updates, but moving its caret
          // here steals the selection from the visible composer without changing
          // document.activeElement. The foreground then still looks focused while
          // printable keydowns produce no input.
          placeCaretEnd(editor)
        }
      }

      if (focus && visibleRef.current) {
        requestMainFocus()
      }
    },
    [requestMainFocus, setComposerText, target]
  )

  const appendExternalText = useCallback(
    (text: string, mode: ComposerInsertMode): boolean => {
      const value = text.trim()

      if (!value) {
        return false
      }

      // 'prefix' puts the value at the START of the draft — slash commands
      // only route when they lead the message.
      if (mode === 'prefix') {
        const rest = draftRef.current.trimStart()

        paintDraft(`${value} ${rest}`.trimEnd())

        return true
      }

      const base = mode === 'inline' ? draftRef.current.trimEnd() : draftRef.current
      const sep = mode === 'inline' ? (base ? ' ' : '') : base && !base.endsWith('\n') ? '\n\n' : ''

      paintDraft(`${base}${sep}${value}`)

      return true
    },
    [paintDraft]
  )

  // Keep-alive tabs keep this composer mounted. A background session whose
  // turn finished or recovered from a reconnect would otherwise re-run this
  // effect and steal the caret. usePaneVisible defaults true outside a tab
  // stack, so tiles, pop-outs, and secondary windows still auto-focus.
  useEffect(() => {
    if (!inputDisabled && paneVisible && !floating) {
      // A tab becoming visible or a turn finishing is not a request to open
      // a phone keyboard. Explicit focus requests below still focus normally.
      if (isTouchInteraction()) {
        markActiveComposer(target)
      } else {
        focusInput()
      }
    }
  }, [floating, focusInput, focusKey, inputDisabled, paneVisible, target])

  const previousFocusRequest = useRef(focusRequestId)
  // eslint-disable-next-line no-restricted-syntax -- handled request token, not a mirrored atom
  useEffect(() => {
    if (previousFocusRequest.current === focusRequestId) {
      return
    }

    previousFocusRequest.current = focusRequestId

    if (!inputDisabled && paneVisible) {
      focusInput()
    }
  }, [focusInput, focusRequestId, inputDisabled, paneVisible])

  // The mirror of the `markActiveComposer` above: give the key back when this
  // composer goes away (a session tile closing, a pane unmounting). Covers both
  // claim sites for this composer — `focusInput` here and ChatBar's `onFocus` —
  // since they mark the same scope target. Without it `'active'` keeps
  // resolving to a dead tile and every routed focus/insert request is dropped.
  // (Heal-to-visible in focus.ts covers the keep-alive-tab case where the pane
  // stays mounted behind the front tab; this covers true unmounts.)
  useEffect(() => () => releaseActiveComposer(target), [target])

  useEffect(() => {
    if (inputDisabled) {
      return undefined
    }

    const offFocus = onComposerFocusRequest(({ target: requested, typeChar }) => {
      if (requested !== target) {
        return
      }

      // Type-to-focus appends at end; bare Enter just focuses.
      if (typeChar) {
        paintDraft(`${draftRef.current}${typeChar}`, false)
        focusInput()

        return
      }

      setFocusRequestId(id => id + 1)
    })

    const offInsert = onComposerInsertRequest(({ mode, target: requested, text, token }) => {
      if (requested === target) {
        // A tokened insert came from the plugin SDK — echo whether the text
        // actually landed, so its promise never settles on a silent no-op.
        ackComposerInsert(token, appendExternalText(text, mode))
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText, focusInput, inputDisabled, paintDraft, target])

  const sameDraftScope = (a: string | null | undefined, b: string | null | undefined) =>
    (isFreshDraftScope(a) || !a ? null : a) === (isFreshDraftScope(b) || !b ? null : b)

  const stashAt = (scope: string | null, text = draftRef.current, attachments = attachmentScope.$attachments.get()) => {
    if (sameDraftScope(scope, draftScopeRef.current)) {
      savedDraftRef.current = savedDraft(text, attachments)
    }

    stashSessionDraft(scope, text, attachments)
  }

  // The automatic saves (debounce, session switch, HUD flush): a draft this
  // window has not edited since it loaded or saved it is left to the stash,
  // which may hold a newer value (or none, once sent) from another window.
  const stashOwnEdits = (scope: string | null, text: string, attachments = attachmentScope.$attachments.get()) => {
    if (sameDraftScope(scope, draftScopeRef.current) && isSavedDraft(savedDraftRef.current, text, attachments)) {
      return
    }

    stashAt(scope, text, attachments)
  }

  // A rejected or unconfirmed send is restored with its identity, so Enter on that exact draft is
  // the explicit retry. Recorded per draft scope (it survives a reload with the draft text).
  const rememberDraftRetry = (scope: string | null, retry: ComposerDraftRetry) => {
    writeDraftRetry(scope, retry)

    if (sameDraftScope(scope, draftScopeRef.current)) {
      draftRetryRef.current = retry
    }
  }

  // The identity a send of exactly this draft retries, if any. Spent by the send either way: a
  // rejection restores it again, and an accepted send leaves nothing to retry.
  const takeDraftRetry = (text: string, attachments: ComposerAttachment[]) => {
    const retry = matchingDraftRetry(draftRetryRef.current, text, attachments)
    draftRetryRef.current = undefined

    if (retry) {
      writeDraftRetry(draftScopeRef.current, undefined)
    }

    return retry
  }

  // Draft read/write bus (plugin SDK `host.composer`): answer for the sessions
  // this composer owns — the runtime id, the queue/stored key (tiles run with
  // sessionId = their stored id; the primary's queue key is the resolved
  // stored id), so a plugin addressing either identity reaches this surface.
  // Reads answer the live DOM text (the stash lags by the persist debounce);
  // writes go through paintDraft — the app's own programmatic-draft path, so
  // `@`-ref / `/` tokens hydrate as chips like official paste.
  useEffect(() => {
    if (inputDisabled) {
      return undefined
    }

    return onComposerDraftRequests(
      {
        getIds: () => {
          const ids = [sessionIdRef.current, activeQueueSessionKeyRef.current].filter((id): id is string => Boolean(id))

          // A surface with no session yet IS the new-chat draft (the stash keys
          // it by the fresh-draft key); once one opens, the new-chat draft
          // belongs elsewhere.
          return ids.length ? ids : [freshDraftScope()]
        },
        isActive: () => getActiveComposer() === target
      },
      {
        read: () => {
          const editor = editorRef.current

          return editor ? composerPlainText(editor) : draftRef.current
        },
        write: text => {
          const editor = editorRef.current

          // Hidden keep-alive panes still answer: multi-session plugins route
          // to a specific session's composer, and paintDraft never steals the
          // caret of a non-visible surface.
          if (!editor || !editor.isConnected) {
            return false
          }

          paintDraft(text)

          return true
        }
      }
    )
  }, [inputDisabled, paintDraft, target])

  const loadIntoComposer = (text: string, attachments: ComposerAttachment[], preserveFocusedCaret = false) => {
    // Diagnostic breadcrumb for #59305-class reports: identifies WHAT kind of
    // state got restored into the composer (session switch, queue-edit
    // restore, history browse) without logging any raw content. REF_RE has the
    // global flag — testing against a throwaway clone avoids mutating the
    // shared instance's lastIndex, which would otherwise corrupt this check on
    // the next call.
    if (attachments.length > 0 || new RegExp(REF_RE.source, REF_RE.flags).test(text)) {
      console.debug('[composer-rehydrate]', {
        attachmentCount: attachments.length,
        attachmentKinds: attachments.map(a => a.kind),
        hasTextRefs: new RegExp(REF_RE.source, REF_RE.flags).test(text),
        scope: activeQueueSessionKeyRef.current
      })
    }

    attachmentScope.$attachments.set(cloneAttachments(attachments))
    paintDraft(text, false, preserveFocusedCaret)
  }

  const clearDraft = useCallback(() => {
    setComposerText('')
    draftRef.current = ''

    if (editorRef.current) {
      renderComposerContents(editorRef.current, '')

      // A caret placed inside a send tap focuses the editor, and on a phone
      // raises the keyboard the user put away. Touch keeps it only if focused.
      if (
        visibleRef.current &&
        getActiveComposer() === target &&
        !isElementInHiddenPane(editorRef.current) &&
        (!isTouchInteraction() || editorRef.current.contains(document.activeElement))
      ) {
        placeCaretEnd(editorRef.current)
      }
    }
  }, [setComposerText, target])

  // Schedule the debounced per-session stash of `text`. Browsing history /
  // editing a queued prompt suppress it so recalled text never clobbers the
  // draft. A text already scheduled for this scope keeps its timer: typed text
  // is scheduled at the keystroke, and the coalesced store write that lands it
  // later must not restart the window (a window closed inside it loses the draft).
  const scheduleDraftPersist = (text: string) => {
    if (isBrowsingHistory(sessionIdRef.current) || queueEditRef.current) {
      return
    }

    const scope = draftScopeRef.current

    // An edited (or cleared) restored draft is a new message: it no longer retries the old send.
    if (draftRetryRef.current && !draftRetryTextMatches(draftRetryRef.current, text)) {
      draftRetryRef.current = undefined
      writeDraftRetry(scope, undefined)
    }

    const pending = pendingDraftPersistRef.current

    if (pending && pending.scope === scope && pending.text === text) {
      return
    }

    const entry = { scope, text }
    pendingDraftPersistRef.current = entry
    window.clearTimeout(draftPersistTimerRef.current)
    draftPersistTimerRef.current = window.setTimeout(() => {
      // Integrity guard (defense-in-depth, #54527): only commit if this is
      // still the pending write on file. A session swap or a newer
      // keystroke clears/replaces it before firing in the normal case; this
      // catches any future call site that skips that bookkeeping instead of
      // silently filing text under the wrong session.
      if (!isPendingDraftPersistCurrent(pendingDraftPersistRef.current, entry)) {
        return
      }

      pendingDraftPersistRef.current = null
      stashOwnEdits(scope, text)
    }, DRAFT_PERSIST_DEBOUNCE_MS)
  }

  // A file removed from (or added to) a restored draft makes it a new message too. Dropping the
  // identity here, not at the next load, matters: a reload legitimately loses a draft's files, so a
  // later load cannot tell a removal from that and would resend the removed file under the old id.
  useEffect(
    () =>
      attachmentScope.$attachments.listen(attachments => {
        const retry = draftRetryRef.current

        if (
          !retry ||
          queueEditRef.current ||
          isBrowsingHistory(sessionIdRef.current) ||
          retry.attachmentIds.join('\n') === retryAttachmentIds(attachments).join('\n')
        ) {
          return
        }

        draftRetryRef.current = undefined
        writeDraftRetry(draftScopeRef.current, undefined)
      }),
    [attachmentScope, queueEditRef]
  )

  const scheduleDraftPersistRef = useRef(scheduleDraftPersist)
  scheduleDraftPersistRef.current = scheduleDraftPersist

  // Read the editor's current plain text into draftRef + composer state. This
  // closes the "queued rAF flush hasn't run yet" window so scope-swap/pagehide
  // persistence captures the latest keystrokes. `typing` marks ChatBar's
  // coalesced per-keystroke flush, whose store write may wait for a typing
  // pause so the transcript's store subscribers do not re-render on every
  // character (see composer-store-writer.ts); every other caller lands the
  // text now, pending keystrokes included.
  const syncDraftFromEditor = useCallback(
    (typing = false, editor: HTMLDivElement | null = editorRef.current) => {
      if (!editor) {
        return draftRef.current
      }

      // Same normalize-then-sanitize the rAF flush does. An emptied editor still
      // holds the placeholder <br> that keeps the contenteditable from collapsing
      // to a sliver, and that serializes as "\n" — so an editor the user just
      // cleared would otherwise stash a one-newline draft and come back non-empty.
      normalizeComposerEditorDom(editor)

      const text = sanitizeComposerInput(composerPlainText(editor))

      if (text !== draftRef.current) {
        draftRef.current = text
        storeWriter[typing ? 'typing' : 'now'](text)

        // The runtime subscription schedules the stash once the store sees
        // the text; a deferred typing write must not delay it past the keystroke.
        if (storeWriter.pending()) {
          scheduleDraftPersistRef.current(text)
        }
      } else if (!typing) {
        storeWriter.flush()
      }

      return text
    },
    [storeWriter]
  )

  // Imperative draft sync — the spine of the "work only when work is to be
  // performed" model. Subscribing to the composer runtime directly (not
  // `useAuiState(text)` + a `[draft]` effect) keeps per-keystroke text out of
  // React, so typing never re-renders the chrome. On each change we (1) mirror
  // text into draftRef, (2) repaint the editor only when the change came from
  // OUTSIDE it (programmatic clear/restore/insert; the focused editor is the
  // source otherwise), and (3) schedule the debounced per-session stash.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const sync = () => {
      // Typed text the store has not seen yet is newer than the store: mirroring
      // it now would roll draftRef (and an unfocused editor) back. The pending
      // write lands shortly and runs this again.
      if (storeWriter.pending()) {
        return
      }

      const text = composerRuntime.getState().text
      storeWriter.observed(text)
      draftRef.current = text

      const editor = editorRef.current

      if (editor && document.activeElement !== editor && composerPlainText(editor) !== text) {
        renderComposerContents(editor, text, { trailingCommitted: true })
      }

      scheduleDraftPersistRef.current(text)
    }

    const unsubscribe = composerRuntime.subscribe(sync)

    return () => {
      unsubscribe()
      window.clearTimeout(draftPersistTimerRef.current)
    }
  }, [composerRuntime, queueEditRef, storeWriter])

  const insertText = (text: string) => {
    const base = draftRef.current
    const sep = base && !base.endsWith('\n') ? '\n' : ''

    paintDraft(`${base}${sep}${text}`)
  }

  // insertInlineRefs mutates the editor in place (chips), so it can't go through
  // paintDraft's re-render — it mirrors the resulting plain text and refocuses.
  const insertInlineRefs = (refs: InlineRefInput[]) => {
    const editor = editorRef.current

    if (!editor) {
      return false
    }

    const interactive = visibleRef.current && getActiveComposer() === target && !isElementInHiddenPane(editor)
    const nextDraft = insertInlineRefsIntoEditor(editor, refs, { interactive })

    if (nextDraft === null) {
      return false
    }

    draftRef.current = nextDraft
    setComposerText(nextDraft)

    if (interactive) {
      requestMainFocus()
    }

    return true
  }

  // Latest-closure ref so the once-only subscription always calls the current
  // insertInlineRefs without re-subscribing every render.
  const insertInlineRefsRef = useRef(insertInlineRefs)
  insertInlineRefsRef.current = insertInlineRefs

  useEffect(() => {
    return onComposerInsertRefsRequest(({ refs, target: requested }) => {
      if (requested === target) {
        insertInlineRefsRef.current(refs)
      }
    })
  }, [target])

  // Per-thread draft swap — the composer's only session coupling. Lifecycle
  // never clears composer state; this effect alone stashes on leave, restores
  // on enter. Keyed writes are idempotent, so no skip-sentinel.
  //
  // MUST be a layout effect, not a passive one: it swaps attachmentScope's
  // module-level $attachments atom, and a passive effect fires only after the
  // browser paints the new session's view — leaving a window where the DOM
  // already shows session B while $attachments (and therefore ChatBar's
  // `attachments` prop) still holds session A's chips. A submit fired in that
  // window (e.g. a fast session switch immediately followed by Enter) would
  // ship A's attachments into B's turn (#59305). useLayoutEffect closes the
  // window by running before paint.

  useLayoutEffect(() => {
    // A pending debounce timer from the outgoing session is now stale — its
    // scope was correct when scheduled, but the authoritative stash below
    // (and the cleanup on the way out) already covers that text. Letting it
    // fire later would just clobber with an older snapshot.
    window.clearTimeout(draftPersistTimerRef.current)
    pendingDraftPersistRef.current = null

    // A new chat writes to its own pre-session bucket (the per-lifecycle fresh
    // key) until its stored id arrives; the assigning site announces that id
    // (store/composer.ts). Move the bucket at this handoff — after the outgoing
    // cleanup stashed the live editor text under it, before the incoming scope
    // is restored — so the text the user kept typing follows the chat instead
    // of vanishing. Keyed on the scope alone: the runtime id can land a resume
    // later than the route flips the scope, so it is not a usable signal here.
    if ((isFreshDraftScope(draftScopeRef.current) || !draftScopeRef.current) && activeQueueSessionKey) {
      const freshKey = draftScopeRef.current || freshDraftScope()

      // A follow-up queued while session.create was in flight sits under the
      // fresh key; it belongs to this chat too. The queue re-key never moves
      // fresh → stored (#139190), so only this announced handoff does.
      if (adoptNewSessionDraft(activeQueueSessionKey)) {
        migrateQueuedPrompts(freshKey, activeQueueSessionKey)
      }
    } else if (!activeQueueSessionKey) {
      // The reverse handoff: a session the user was typing into turned out
      // to be gone and the window dropped to a fresh draft (#111868). The
      // outgoing composer's cleanup has already stashed the live text under
      // the dead key — whether that was this instance's previous scope or an
      // unmounted one's — so move it into the fresh draft when the gone
      // verdict announced it. No announcement, no-op.
      adoptGoneSessionDraft()
    }

    draftScopeRef.current = activeQueueSessionKey
    const storedRetry = readDraftRetry(activeQueueSessionKey)

    const { attachments, text } = takeSessionDraft(activeQueueSessionKey)
    savedDraftRef.current = savedDraft(text, attachments)
    draftRetryRef.current = loadedDraftRetry(storedRetry, text, attachments)
    loadIntoComposer(text, attachments, true)

    if (draftRetryRef.current && draftRetryRef.current !== storedRetry) {
      // Recorded text-only, so the notice is said once, not again on every return to this chat.
      writeDraftRetry(activeQueueSessionKey, draftRetryRef.current)
      noteAttachmentsNotRestored(storedRetry)
    }

    // Takes back a draft that is still exactly `retry`'s send under its identity: words AND
    // attachments, since a file attached after the offer makes it the user's new message.
    const clearRestoredDraft = (retry: ComposerDraftRetry) => {
      const stored = readDraftRetry(activeQueueSessionKey)

      if (
        draftScopeRef.current !== activeQueueSessionKey ||
        stored?.id !== retry.id ||
        !matchingDraftRetry(stored, syncDraftFromEditor(), attachmentScope.$attachments.get())
      ) {
        return
      }

      stashAt(activeQueueSessionKey, '', [])
      writeDraftRetry(activeQueueSessionKey, undefined)
      draftRetryRef.current = undefined
      loadIntoComposer('', [])
    }

    const interruptedSend: InterruptedSendHandlers = {
      offer: retry => {
        // Re-read after the owner check: anything typed or attached meanwhile keeps the composer,
        // and the send stays journaled for a later load.
        if (
          draftScopeRef.current !== activeQueueSessionKey ||
          syncDraftFromEditor().trim() ||
          attachmentScope.$attachments.get().length > 0
        ) {
          return false
        }

        // Re-offered as text: attachment bytes do not outlive the page, so the identity names exactly
        // what is offered and Enter on it is the explicit retry of that send.
        stashAt(activeQueueSessionKey, retry.text, [])
        rememberDraftRetry(activeQueueSessionKey, { ...retry, attachmentIds: [] })
        loadIntoComposer(retry.text, [])
        noteAttachmentsNotRestored(retry)

        return true
      },
      // The adopted saved draft is a copy of a send the owner already holds.
      landed: clearRestoredDraft,
      // The owner listed the offered send after all.
      withdraw: clearRestoredDraft
    }

    // The saved draft can already be an interrupted send's words (persisted before the send cleared
    // the composer): it takes that send's identity, once its owner lock proves no live window still
    // sends it, provided the composer still holds exactly that saved draft.
    void adoptInterruptedSend(activeQueueSessionKey, text, savedDraftAt(activeQueueSessionKey), {
      ...interruptedSend,
      adopt: adopted => {
        if (
          draftScopeRef.current !== activeQueueSessionKey ||
          !draftRetryTextMatches(adopted, syncDraftFromEditor()) ||
          retryAttachmentIds(attachmentScope.$attachments.get()).join() !== retryAttachmentIds(attachments).join()
        ) {
          return false
        }

        if (!draftRetryTextMatches(draftRetryRef.current, text)) {
          rememberDraftRetry(activeQueueSessionKey, { ...adopted, attachmentIds: retryAttachmentIds(attachments) })
        }

        // The saved draft can already carry a text-only identity for these words; the journaled send
        // still names the attachments the reload dropped.
        if (!attachments.length) {
          noteAttachmentsNotRestored(adopted)
        }

        return true
      }
    })

    if (!text.trim() && attachments.length === 0) {
      // A send its page never finished (the window closed or reloaded mid-send) is this draft.
      void restoreInterruptedSend(activeQueueSessionKey, interruptedSend)
    }

    return () => {
      const latestText = syncDraftFromEditor()
      const editing = queueEditStateRef.current

      if (editing?.sessionKey === activeQueueSessionKey) {
        stashOwnEdits(activeQueueSessionKey, editing.draft, editing.attachments)
      } else if (!isBrowsingHistory(sessionId)) {
        stashOwnEdits(activeQueueSessionKey, latestText)
      }
    }
  }, [activeQueueSessionKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // The HUD handoff's two verbs. Entering HUD mode flushes this editor's text
  // into the shared stash so the HUD's composer boots with it; leaving repaints
  // from the stash so whatever the HUD typed (or sent, clearing it) is what the
  // app window shows. The per-session swap effect above can't cover either one:
  // the session scope doesn't change, so it never re-consults the stash.
  const syncDraft = (mode: ComposerDraftSyncMode) => {
    if (mode === 'flush') {
      window.clearTimeout(draftPersistTimerRef.current)
      pendingDraftPersistRef.current = null
      stashOwnEdits(draftScopeRef.current, syncDraftFromEditor())

      return
    }

    reloadPersistedDrafts()
    const stashed = takeSessionDraft(draftScopeRef.current)
    savedDraftRef.current = savedDraft(stashed.text, stashed.attachments)
    draftRetryRef.current = loadedDraftRetry(readDraftRetry(draftScopeRef.current), stashed.text, stashed.attachments)
    loadIntoComposer(stashed.text, stashed.attachments)
  }

  const syncDraftRef = useRef(syncDraft)
  syncDraftRef.current = syncDraft

  useEffect(
    () =>
      onComposerDraftSyncRequest(({ mode, target: requested }) => {
        if (requested === target) {
          syncDraftRef.current(mode)
        }
      }),
    [target]
  )

  // pagehide is load-bearing: React skips effect cleanups on reload, so Cmd+R
  // inside the debounce/rAF window would drop trailing keystrokes without this.
  // beforeunload covers closing the window: Electron tears the renderer down
  // before a pagehide-time localStorage write lands (live: a close right after
  // the last keystroke kept the beforeunload write and lost the pagehide one).
  // The handler never cancels the unload.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const flushPendingDraftPersist = () => {
      const scope = draftScopeRef.current
      const editing = queueEditStateRef.current

      if (editing?.sessionKey === scope || isBrowsingHistory(sessionIdRef.current)) {
        return
      }

      const latestText = syncDraftFromEditor()

      // Only this window's own unsaved edits: an untouched editor may still
      // show a draft another window has since sent, and must not restore it.
      // Text alone: attachments are memory-only and close with the window.
      if (latestText === savedDraftRef.current.text) {
        return
      }

      pendingDraftPersistRef.current = null
      stashAt(scope, latestText)
    }

    window.addEventListener('beforeunload', flushPendingDraftPersist)
    window.addEventListener('pagehide', flushPendingDraftPersist)

    return () => {
      window.removeEventListener('beforeunload', flushPendingDraftPersist)
      window.removeEventListener('pagehide', flushPendingDraftPersist)
      flushPendingDraftPersist()
    }
  }, [syncDraftFromEditor])

  return {
    activeQueueSessionKeyRef,
    clearDraft,
    draftScopeRef,
    draftRef,
    editorRef,
    focusInput,
    hasText,
    insertInlineRefs,
    insertText,
    isHelpHint,
    isSteerableText,
    loadIntoComposer,
    rememberDraftRetry,
    requestMainFocus,
    sessionIdRef,
    setComposerText,
    stashAt,
    syncDraftFromEditor,
    takeDraftRetry
  }
}
