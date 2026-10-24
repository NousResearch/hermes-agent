import { type RefObject, useCallback, useEffect, useRef } from 'react'

import {
  ackComposerSpellcheck,
  type ComposerTarget,
  getComposerSpellcheckDesired,
  onComposerSpellcheckRequest
} from '../focus'

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
  // The state last requested for THIS surface. Held in a ref rather than
  // state because it is applied to the DOM directly, not rendered.
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
      if (detail.target !== target) {
        return
      }

      apply(detail.enabled)

      // Acknowledge so the requester learns the setting was applied rather
      // than dropped. A request for a surface with no mounted subscriber is
      // never acknowledged, and the requester settles false on its timeout.
      ackComposerSpellcheck(detail.token, true)
    })

    return off
  }, [apply, target])

  // Apply the requested state when this surface mounts.
  //
  // A request can arrive BEFORE the surface exists -- a plugin asks for a
  // session whose tile is not open yet, and the user opens it later. The
  // preference is recorded against the surface, so reading it back here is
  // what makes the setting land rather than being lost.
  //
  // This effect runs once per mount: the ref object is stable, so the
  // dependency never changes, and the state it reads only matters at mount.
  // A request that arrives after mount is applied by the handler above.
  useEffect(() => {
    const desired = getComposerSpellcheckDesired(target)

    if (desired !== undefined) {
      apply(desired)
    }
  }, [apply, target])
}
