import { type RefObject, useCallback, useEffect } from 'react'

import { ackComposerSpellcheck, type ComposerTarget, onComposerSpellcheckRequest } from '../focus'

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
 * ignored, so a plugin can spell check the primary composer and leave a tile
 * alone. It is also live-only: the setting is written when the request
 * arrives, and nothing is replayed on mount, so the SDK's `false` keeps one
 * meaning ("nothing was applied") instead of leaving a preference behind that
 * lands later on a surface the caller was told had not received it.
 *
 * The attribute is applied imperatively rather than through React state: the
 * editable is a `contentEditable` whose contents the app owns, and
 * re-rendering it to flip a boolean risks clobbering an in-flight draft. The
 * DOM attribute survives re-renders, so this stays correct without a render.
 * It is written as the content attribute -- the same channel React renders
 * `spellCheck` into -- rather than the IDL property, so the two agree (jsdom
 * does not reflect the property onto the attribute; Chromium does).
 *
 * macOS note: Chromium gates its punctuation substitutions on BOTH attributes.
 * `autocorrect=off` drops every substitution, while `spellcheck=false` drops
 * only the language ones (quotes, dashes, ellipses) and keeps the user's
 * custom Text replacements. The app sets `autocorrect="on"` on macOS to keep
 * those custom replacements, so turning spell check ON here also re-enables
 * smart quotes and dashes for this composer.
 */
export function useComposerSpellcheck({ target, editorRef }: UseComposerSpellcheckOptions) {
  /** Write the setting to the editable. False = there was nothing to write
   *  to, which is what the requester gets told. */
  const apply = useCallback(
    (enabled: boolean) => {
      const editor = editorRef.current

      if (!editor) {
        return false
      }

      editor.setAttribute('spellcheck', enabled ? 'true' : 'false')

      return true
    },
    [editorRef]
  )

  useEffect(() => {
    const off = onComposerSpellcheckRequest(detail => {
      if (detail.target !== target) {
        return
      }

      // Acknowledge the REAL outcome, not that the request arrived: a surface
      // whose editable is not attached yet has applied nothing, and the
      // requester's promise is the only way a plugin learns that.
      ackComposerSpellcheck(detail.token, apply(detail.enabled))
    })

    return off
  }, [apply, target])
}
