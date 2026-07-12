import { useAui, useAuiState, useComposerRuntime } from '@assistant-ui/react'
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { $composerAttachments, type ComposerAttachment, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { isBrowsingHistory } from '@/store/composer-input-history'

import {
  cloneAttachments,
  DRAFT_PERSIST_DEBOUNCE_MS,
  isPendingDraftPersistCurrent,
  type QueueEditState
} from '../composer-utils'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRefsRequest,
  onComposerInsertRequest
} from '../focus'
import { type InlineRefInput, insertInlineRefsIntoEditor } from '../inline-refs'
import { composerPlainText, placeCaretEnd, renderComposerContents } from '../rich-editor'
import type { ChatBarProps } from '../types'

interface UseComposerDraftArgs {
  activeQueueSessionKey: string | null
  focusKey: ChatBarProps['focusKey']
  inputDisabled: boolean
  queueEditRef: RefObject<QueueEditState | null>
  sessionId: string | null | undefined
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

  // Coarse edges only — these flip rarely (empty↔non-empty, the `?` help sigil,
  // steerable-vs-slash), so typing within a line costs no render.
  const hasText = useAuiState(s => s.composer.text.trim().length > 0)
  const isHelpHint = useAuiState(s => s.composer.text === '?')

  const isSteerableText = useAuiState(s => {
    const trimmed = s.composer.text.trim()

    return trimmed.length > 0 && !SLASH_COMMAND_RE.test(trimmed)
  })

  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef('')
  const pendingComposerTextRef = useRef<string | null>(null)
  const pendingComposerTextRafRef = useRef<number | undefined>(undefined)
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
  const sessionIdRef = useRef(sessionId)
  sessionIdRef.current = sessionId
  const queueEditStateRef = useRef<QueueEditState | null>(queueEditRef.current)
  queueEditStateRef.current = queueEditRef.current

  const [focusRequestId, setFocusRequestId] = useState(0)

  const flushPendingComposerText = useCallback(() => {
    const pending = pendingComposerTextRef.current

    if (pending === null) {
      return true
    }

    try {
      aui.composer().setText(pending)
      pendingComposerTextRef.current = null

      return true
    } catch {
      return false
    }
  }, [aui])

  const schedulePendingComposerTextFlush = useCallback(() => {
    if (pendingComposerTextRafRef.current !== undefined) {
      return
    }

    const retry = () => {
      pendingComposerTextRafRef.current = undefined

      if (pendingComposerTextRef.current === null) {
        return
      }

      if (!flushPendingComposerText()) {
        pendingComposerTextRafRef.current = window.requestAnimationFrame(retry)
      }
    }

    pendingComposerTextRafRef.current = window.requestAnimationFrame(retry)
  }, [flushPendingComposerText])

  // assistant-ui's composer mutators throw when the core isn't bound yet (a
  // startup/thread-swap window). Keep the latest desired draft on file and
  // replay it once the core binds so stale session text cannot leak back in.
  const setComposerText = useCallback(
    (value: string) => {
      pendingComposerTextRef.current = value

      if (!flushPendingComposerText()) {
        schedulePendingComposerTextFlush()
      }
    },
    [flushPendingComposerText, schedulePendingComposerTextFlush]
  )

  const focusInput = useCallback(() => {
    focusComposerInput(editorRef.current)
    markActiveComposer('main')
  }, [])

  const requestMainFocus = useCallback(() => {
    setFocusRequestId(id => id + 1)
  }, [])

  // The single write path for programmatic draft mutations: mirror → AUI state →
  // repaint the editor (caret to end). Repaints even while focused — inserts /
  // restores run mid-focus, and the runtime sync only repaints an unfocused
  // editor — so the visible text never lags the store.
  const paintDraft = useCallback(
    (next: string, focus = true) => {
      draftRef.current = next
      setComposerText(next)

      const editor = editorRef.current

      if (editor) {
        renderComposerContents(editor, next)
        placeCaretEnd(editor)
      }

      if (focus) {
        requestMainFocus()
      }
    },
    [requestMainFocus, setComposerText]
  )

  const appendExternalText = useCallback(
    (text: string, mode: ComposerInsertMode) => {
      const value = text.trim()

      if (!value) {
        return
      }

      const base = mode === 'inline' ? draftRef.current.trimEnd() : draftRef.current
      const sep = mode === 'inline' ? (base ? ' ' : '') : base && !base.endsWith('\n') ? '\n\n' : ''

      paintDraft(`${base}${sep}${value}`)
    },
    [paintDraft]
  )

  useEffect(() => {
    if (!inputDisabled) {
      focusInput()
    }
  }, [focusInput, focusKey, focusRequestId, inputDisabled])

  useEffect(
    () => () => {
      if (pendingComposerTextRafRef.current !== undefined) {
        window.cancelAnimationFrame(pendingComposerTextRafRef.current)
      }
    },
    []
  )

  useEffect(() => {
    if (inputDisabled) {
      return undefined
    }

    const offFocus = onComposerFocusRequest(target => {
      if (target === 'main') {
        setFocusRequestId(id => id + 1)
      }
    })

    const offInsert = onComposerInsertRequest(({ mode, target, text }) => {
      if (target === 'main') {
        appendExternalText(text, mode)
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText, inputDisabled])

  const stashAt = (scope: string | null, text = draftRef.current, attachments = $composerAttachments.get()) =>
    stashSessionDraft(scope, text, attachments)

  const loadIntoComposer = (text: string, attachments: ComposerAttachment[]) => {
    $composerAttachments.set(cloneAttachments(attachments))
    paintDraft(text, false)
  }

  const clearDraft = useCallback(() => {
    setComposerText('')
    draftRef.current = ''

    if (editorRef.current) {
      editorRef.current.replaceChildren()
    }
  }, [setComposerText])

  // Read the editor's current plain text into draftRef + composer state. This
  // closes the "queued rAF flush hasn't run yet" window so scope-swap/pagehide
  // persistence captures the latest keystrokes.
  const syncDraftFromEditor = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return draftRef.current
    }

    const text = composerPlainText(editor)

    if (text !== draftRef.current) {
      draftRef.current = text
      setComposerText(text)
    }

    return text
  }, [setComposerText])

  // Imperative draft sync — the spine of the "work only when work is to be
  // performed" model. Subscribing to the composer runtime directly (not
  // `useAuiState(text)` + a `[draft]` effect) keeps per-keystroke text out of
  // React, so typing never re-renders the chrome. On each change we (1) mirror
  // text into draftRef, (2) repaint the editor only when the change came from
  // OUTSIDE it (programmatic clear/restore/insert; the focused editor is the
  // source otherwise), and (3) schedule the debounced per-session stash.
  // Browsing history / editing a queued prompt suppress the stash so recalled
  // text never clobbers the draft.
  useEffect(() => {
    const sync = () => {
      const text = composerRuntime.getState().text
      draftRef.current = text

      const editor = editorRef.current

      if (editor && document.activeElement !== editor && composerPlainText(editor) !== text) {
        renderComposerContents(editor, text)
      }

      if (isBrowsingHistory(sessionIdRef.current) || queueEditRef.current) {
        return
      }

      const scope = draftScopeRef.current
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
        stashAt(scope, text)
      }, DRAFT_PERSIST_DEBOUNCE_MS)
    }

    const unsubscribe = composerRuntime.subscribe(sync)

    return () => {
      unsubscribe()
      window.clearTimeout(draftPersistTimerRef.current)
    }
  }, [composerRuntime, queueEditRef])

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

    const nextDraft = insertInlineRefsIntoEditor(editor, refs)

    if (nextDraft === null) {
      return false
    }

    draftRef.current = nextDraft
    setComposerText(nextDraft)
    requestMainFocus()

    return true
  }

  // Latest-closure ref so the once-only subscription always calls the current
  // insertInlineRefs without re-subscribing every render.
  const insertInlineRefsRef = useRef(insertInlineRefs)
  insertInlineRefsRef.current = insertInlineRefs

  useEffect(() => {
    return onComposerInsertRefsRequest(({ refs, target }) => {
      if (target === 'main') {
        insertInlineRefsRef.current(refs)
      }
    })
  }, [])

  // Per-thread draft swap — the composer's only session coupling. Lifecycle
  // never clears composer state; this effect alone stashes on leave, restores
  // on enter. Keyed writes are idempotent, so no skip-sentinel.
  useEffect(() => {
    // A pending debounce timer from the outgoing session is now stale — its
    // scope was correct when scheduled, but the authoritative stash below
    // (and the cleanup on the way out) already covers that text. Letting it
    // fire later would just clobber with an older snapshot.
    window.clearTimeout(draftPersistTimerRef.current)
    pendingDraftPersistRef.current = null
    draftScopeRef.current = activeQueueSessionKey

    const { attachments, text } = takeSessionDraft(activeQueueSessionKey)
    loadIntoComposer(text, attachments)

    return () => {
      const latestText = syncDraftFromEditor()
      const editing = queueEditStateRef.current

      if (editing?.sessionKey === activeQueueSessionKey) {
        stashAt(activeQueueSessionKey, editing.draft, editing.attachments)
      } else if (!isBrowsingHistory(sessionId)) {
        stashAt(activeQueueSessionKey, latestText)
      }
    }
  }, [activeQueueSessionKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // pagehide is load-bearing: React skips effect cleanups on reload, so Cmd+R
  // inside the debounce/rAF window would drop trailing keystrokes without this.
  useEffect(() => {
    const flushPendingDraftPersist = () => {
      const scope = draftScopeRef.current
      const editing = queueEditStateRef.current

      if (editing?.sessionKey === scope || isBrowsingHistory(sessionIdRef.current)) {
        return
      }

      const latestText = syncDraftFromEditor()
      pendingDraftPersistRef.current = null
      stashAt(scope, latestText)
    }

    window.addEventListener('pagehide', flushPendingDraftPersist)

    return () => {
      window.removeEventListener('pagehide', flushPendingDraftPersist)
      flushPendingDraftPersist()
    }
  }, [syncDraftFromEditor])

  return {
    activeQueueSessionKeyRef,
    clearDraft,
    draftRef,
    editorRef,
    focusInput,
    hasText,
    insertInlineRefs,
    insertText,
    isHelpHint,
    isSteerableText,
    loadIntoComposer,
    requestMainFocus,
    sessionIdRef,
    setComposerText,
    stashAt
  }
}
