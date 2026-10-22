import { type RefObject, useCallback, useEffect, useRef } from 'react'

import { type ComposerTarget, onComposerSpellcheckRequest } from '../focus'

interface UseComposerSpellcheckOptions {
  /** This composer's focus-bus target. Only requests addressed to it apply. */
  target: ComposerTarget
  /** The editable element whose `spellcheck` attribute is toggled. */
  editorRef: RefObject<HTMLElement | null>
}

/**
 * Let a plugin turn Chromium's spell check on for this composer.
 *
 * The inputs have spell check off by default (#44415), so the red squiggles
 * Chromium draws are absent unless something asks for them. A plugin that
 * ships its own dictionary asks for them HERE, through the SDK, instead of
 * editing the app's DOM.
 *
 * The default is unchanged: with no request, `spellcheck` stays false, so
 * every user without such a plugin sees exactly what they see today.
 *
 * The verb is per-surface -- a request addressed to another composer is
 * ignored, so a plugin can spell check the primary composer and leave a
 * tile alone.
 *
 * The attribute is applied imperatively rather than through React state: the
 * editable is a `contentEditable` whose contents the app owns, and
 * re-rendering it to flip a boolean risks clobbering an in-flight draft. The
 * DOM attribute survives re-renders, so this stays correct without a render.
 */
export function useComposerSpellcheck({ target, editorRef }: UseComposerSpellcheckOptions) {
  // The last requested state, kept so a request that arrives before the
  // editable is mounted still applies once it is.
  const enabledRef = useRef(false)

  const apply = useCallback(
    (enabled: boolean) => {
      enabledRef.current = enabled

      const editor = editorRef.current

      if (editor) {
        editor.spellcheck = enabled
      }
    },
    [editorRef]
  )

  useEffect(() => {
    const off = onComposerSpellcheckRequest(detail => {
      if (detail.target === target) {
        apply(detail.enabled)
      }
    })

    return off
  }, [apply, target])

  // The editable may mount AFTER the request fired (a tile that opens later).
  // Apply the last requested state whenever the element identity changes.
  useEffect(() => {
    const editor = editorRef.current

    if (editor) {
      editor.spellcheck = enabledRef.current
    }
  }, [editorRef, target])
}
