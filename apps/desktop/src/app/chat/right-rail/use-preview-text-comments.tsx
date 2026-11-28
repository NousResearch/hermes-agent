import type { RefObject } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { getActiveComposer, getVisibleComposerSurfaceId, requestComposerSubmit } from '@/app/chat/composer/focus'
import { Codicon } from '@/components/ui/codicon'
import { PaneStripGlyph } from '@/components/ui/pane-tab'
import { type ResolvedOwner, resolveOwnerNow } from '@/hermes'
import { useI18n } from '@/i18n'
import { capturePreviewAnnotateDestination } from '@/lib/preview-annotate/handoff'
import { isTextCommentPreviewUrl } from '@/lib/preview-text-comment'
import {
  parsePreviewTextSelection,
  PREVIEW_TEXT_SELECTION_CHANNEL,
  PREVIEW_TEXT_SELECTION_MODE_CHANNEL,
  previewTextCommentPrompt,
  type PreviewTextSelection
} from '@/lib/preview-text-comment'
import { notify } from '@/store/notifications'
import { knownSessionOwner, ownerLookupSessionRows } from '@/store/session'
import { profileScopeForSessionOwner } from '@/store/session-request-router'
import { isBrowserWindow } from '@/store/windows'

interface SelectionGuest extends HTMLElement {
  getURL?: () => string
  send?: (channel: string, ...args: unknown[]) => void
}

interface TextCommentDraft {
  selection: PreviewTextSelection
  target: string
  surfaceId: string | null
  owner: ResolvedOwner
}

interface TextCommentsOptions {
  available: boolean
  annotateMode?: boolean
  stopAnnotate?: () => void
  webviewRef: RefObject<SelectionGuest | null>
  url: string
  sessionKey: string | null
}

/** The guest only reports selections. Intent, recipient and submission stay in the host. */
export function usePreviewTextComments({
  available,
  annotateMode = false,
  stopAnnotate,
  webviewRef,
  url,
  sessionKey
}: TextCommentsOptions) {
  const { t } = useI18n()
  const copy = t.preview.web
  const [enabled, setEnabled] = useState(false)
  const [draft, setDraft] = useState<TextCommentDraft | null>(null)
  // Consume synchronously so a second submit in the same frame cannot send twice.
  const pending = useRef<TextCommentDraft | null>(null)

  const replaceDraft = useCallback((next: TextCommentDraft | null) => {
    pending.current = next
    setDraft(next)
  }, [])

  const cancel = useCallback(() => replaceDraft(null), [replaceDraft])

  useEffect(() => {
    setEnabled(false)
    cancel()
  }, [sessionKey, cancel])

  useEffect(() => {
    if (!available || annotateMode) {
      setEnabled(false)
      cancel()
    }
  }, [available, annotateMode, cancel])

  useEffect(() => {
    cancel()
    const guest = webviewRef.current

    if (!guest) {
      return
    }

    const active = enabled && available && !annotateMode
    const configure = () => guest.send?.(PREVIEW_TEXT_SELECTION_MODE_CHANNEL, active)

    const onSelection = (event: Event) => {
      const ipc = event as Event & { channel?: string; args?: unknown[] }

      if (!active || ipc.channel !== PREVIEW_TEXT_SELECTION_CHANNEL) {
        return
      }

      if (ipc.args?.[0] === null) {
        cancel()

        return
      }

      const selection = parsePreviewTextSelection(ipc.args?.[0], guest.getURL?.() || url)

      if (!selection) {
        return
      }

      const target = getActiveComposer()
      const destination = capturePreviewAnnotateDestination(guest)
      const surfaceId = destination?.kind === 'composer' ? getVisibleComposerSurfaceId(target) : null
      const storedId = target.startsWith('tile:') ? target.slice(5) : sessionKey
      const owner = commentOwner(storedId)
      const scale = guest.clientWidth / selection.viewportWidth

      if (scale > 0) {
        const { x, y, width, height } = selection.rect
        selection.rect = { x: x * scale, y: y * scale, width: width * scale, height: height * scale }
      }

      replaceDraft({ selection, target, surfaceId, owner })
    }

    guest.addEventListener('ipc-message', onSelection)
    guest.addEventListener('dom-ready', configure)
    guest.addEventListener('did-navigate', cancel)
    guest.addEventListener('did-navigate-in-page', cancel)

    // Off is the preload default; only send after the user's opt-in on a ready page.
    if (active) {
      configure()
    }

    return () => {
      guest.removeEventListener('ipc-message', onSelection)
      guest.removeEventListener('dom-ready', configure)
      guest.removeEventListener('did-navigate', cancel)
      guest.removeEventListener('did-navigate-in-page', cancel)

      if (guest.isConnected) {
        guest.send?.(PREVIEW_TEXT_SELECTION_MODE_CHANNEL, false)
      }
    }
  }, [available, annotateMode, enabled, cancel, replaceDraft, sessionKey, url, webviewRef])

  const submit = (note: string): boolean => {
    const comment = pending.current

    if (!comment || !note.trim()) {
      return false
    }

    const current = webviewRef.current?.getURL?.() || url

    const storedId = comment.target.startsWith('tile:') ? comment.target.slice(5) : sessionKey
    const owner = commentOwner(storedId)

    if (
      owner.connectionId !== comment.owner.connectionId ||
      owner.profile !== comment.owner.profile ||
      current !== comment.selection.url ||
      !comment.surfaceId ||
      !requestComposerSubmit(previewTextCommentPrompt(comment.selection, note), {
        target: comment.target,
        surfaceId: comment.surfaceId,
        requireAccepted: true
      })
    ) {
      notify({ kind: 'warning', title: copy.textComment, message: copy.textCommentUnavailable })

      return false
    }

    cancel()

    return true
  }

  const toggle = () => {
    cancel()
    stopAnnotate?.()
    setEnabled(value => !value)
  }

  const hasRecipient = capturePreviewAnnotateDestination(webviewRef.current)?.kind === 'composer'

  const action = available ? (
    <PaneStripGlyph
      active={enabled}
      disabled={!hasRecipient}
      icon={<Codicon name="whole-word" size="0.8125rem" />}
      label={!hasRecipient ? copy.textCommentUnavailable : enabled ? copy.textCommentOn : copy.textComment}
      onSelect={toggle}
    />
  ) : null

  return { enabled: enabled && available && !annotateMode, draft, cancel, submit, toggle, action }
}

export function canUsePreviewTextComments(webview: boolean, loading: boolean, url: string): boolean {
  return webview && !loading && !isBrowserWindow() && isTextCommentPreviewUrl(url)
}

function commentOwner(storedId: string | null): ResolvedOwner {
  const scope = profileScopeForSessionOwner(knownSessionOwner(ownerLookupSessionRows(), storedId))

  return resolveOwnerNow(typeof scope === 'string' ? { profile: scope } : (scope ?? undefined))
}
