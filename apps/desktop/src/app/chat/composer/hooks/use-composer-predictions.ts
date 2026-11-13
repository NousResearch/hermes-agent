import { useStore } from '@nanostores/react'
import { type KeyboardEvent, type RefObject, useLayoutEffect, useRef } from 'react'

import { isElementInHiddenPane } from '@/components/pane-shell/pane-visibility'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection } from '@/store/session'

import type { ComposerTarget } from '../focus'
import { $composerPredictionAuthority, $composerPredictions, bindComposerPredictions } from '../predictions'
import { composerPlainText } from '../rich-editor'
import { useComposerVisible } from '../visibility'

interface UseComposerPredictionsArgs {
  busy: boolean
  composingRef: RefObject<boolean>
  draftRef: RefObject<string>
  editingQueue: boolean
  editorRef: RefObject<HTMLDivElement | null>
  hasAttachments: boolean
  hasText: boolean
  hasTrigger: boolean
  inputDisabled: boolean
  insertText: (text: string) => void
  profile?: string | null
  sessionId: string | null | undefined
  target: ComposerTarget
}

/** The preview has no write path into the draft except an explicit, unmodified
 * ArrowRight. Read the live editor again at publication AND keydown: rAF draft
 * sync / React selectors may still describe the previous keystroke. */
export function useComposerPredictions({
  busy,
  composingRef,
  draftRef,
  editingQueue,
  editorRef,
  hasAttachments,
  hasText,
  hasTrigger,
  inputDisabled,
  insertText,
  profile,
  sessionId,
  target
}: UseComposerPredictionsArgs) {
  const predictions = useStore($composerPredictions)
  const activeSessionId = useStore($activeSessionId)
  const activeProfile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  const connectionId = connection?.connectionId
  const authority = useStore($composerPredictionAuthority)
  const bindingRef = useRef<ReturnType<typeof bindComposerPredictions> | null>(null)
  const visible = useComposerVisible()
  const suppressed = useRef(false)

  const eligible =
    visible &&
    target === 'main' &&
    !!sessionId &&
    sessionId === activeSessionId &&
    (profile == null || profile === activeProfile) &&
    !!connectionId &&
    !busy &&
    !inputDisabled &&
    !editingQueue &&
    !hasAttachments &&
    !hasTrigger &&
    !hasText

  const canReceive = () => {
    const editor = editorRef.current

    return (
      eligible &&
      !suppressed.current &&
      !composingRef.current &&
      !!editor &&
      editor.isConnected &&
      !isElementInHiddenPane(editor) &&
      draftRef.current.length === 0 &&
      composerPlainText(editor).length === 0 &&
      !editor.querySelector('[data-ref-text]')
    )
  }

  const canReceiveRef = useRef(canReceive)
  canReceiveRef.current = canReceive

  useLayoutEffect(() => {
    if (
      target !== 'main' ||
      !sessionId ||
      sessionId !== activeSessionId ||
      !visible ||
      (profile != null && profile !== activeProfile) ||
      !connectionId
    ) {
      return
    }

    suppressed.current = false

    const binding = bindComposerPredictions({ sessionId, profile: activeProfile, connectionId }, () =>
      canReceiveRef.current()
    )

    bindingRef.current = binding

    return () => {
      binding.dispose()

      if (bindingRef.current === binding) {
        bindingRef.current = null
      }
    }
  }, [sessionId, target, activeSessionId, visible, profile, activeProfile, connectionId, authority])

  useLayoutEffect(() => {
    // User intent wins for the rest of this turn, including late async results
    // after type→erase or Escape. A new agent turn opens a fresh opportunity.
    if (busy) {
      suppressed.current = false
    }

    if (target === 'main' && visible && sessionId && !canReceiveRef.current()) {
      bindingRef.current?.clear()
    }
  }, [busy, editingQueue, hasAttachments, hasText, hasTrigger, inputDisabled, sessionId, target, visible])

  const invalidate = () => {
    if (target !== 'main' || !sessionId) {
      return
    }

    suppressed.current = true
    bindingRef.current?.clear()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (
      event.defaultPrevented ||
      event.shiftKey ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      event.nativeEvent.isComposing ||
      event.nativeEvent.keyCode === 229 ||
      composingRef.current ||
      document.activeElement !== editorRef.current ||
      event.currentTarget !== editorRef.current ||
      !canReceive()
    ) {
      return false
    }

    const current = bindingRef.current?.current()

    if (!current || current.sessionId !== sessionId || sessionId !== $activeSessionId.get()) {
      return false
    }

    if (event.key !== 'Tab' && event.key !== 'ArrowRight' && event.key !== 'Escape') {
      return false
    }

    event.preventDefault()
    event.stopPropagation()

    if (event.key === 'Tab') {
      bindingRef.current?.cycle()
    } else {
      const text = current.suggestions[current.index]
      invalidate()

      if (event.key === 'ArrowRight' && text) {
        insertText(text)
      }
    }

    return true
  }

  return {
    handleKeyDown,
    invalidate,
    preview: predictions && bindingRef.current?.current() === predictions && canReceive() ? predictions : null
  }
}
