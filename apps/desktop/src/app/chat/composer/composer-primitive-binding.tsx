import { ComposerPrimitive } from '@assistant-ui/react'

/**
 * The invisible half of the composer input.
 *
 * assistant-ui requires `ComposerPrimitive.Input` somewhere in the tree so the
 * composer-state binding (text + IME + paste + form-submit hookup) wires up. We
 * render the real input UI ourselves as the composer's contentEditable, so the
 * primitive is invisible (sr-only) — it is only here to carry the binding.
 *
 * IMPORTANT: don't let it render its default <TextareaAutosize>. That component
 * runs `useLayoutEffect(resizeTextarea)` on every value change and reads
 * `node.scrollHeight` against a hidden measurement textarea, forcing two
 * synchronous layouts per keystroke for an element the user can't see. Profiling
 * 400-char synthetic typing showed >900ms cumulative cost in
 * getHeight2/calculateNodeHeight alone (~2.3ms/key) on top of the per-keystroke
 * React commit.
 *
 * `asChild` swaps TextareaAutosize for a Radix Slot wrapping our plain
 * <textarea>, which carries the binding but skips autosize.
 */
export function ComposerPrimitiveBinding() {
  return (
    <ComposerPrimitive.Input asChild submitMode="ctrlEnter" tabIndex={-1} unstable_focusOnScrollToBottom={false}>
      <textarea
        aria-hidden
        autoCapitalize="off"
        autoComplete="off"
        autoCorrect="off"
        className="sr-only"
        spellCheck={false}
        tabIndex={-1}
      />
    </ComposerPrimitive.Input>
  )
}
