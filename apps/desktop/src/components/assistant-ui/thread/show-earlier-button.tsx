/**
 * The "Show earlier" affordance above the transcript.
 *
 * Extracted verbatim from `./list`: `ThreadMessageListInner` is over its
 * FUNC_LINES cap and may only shrink, so this phase lives in a topical sibling
 * (the remedy the code-health ratchet names for an over-target function).
 */
export function ShowEarlierButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      className="mx-auto mb-(--conversation-turn-gap) rounded-full border border-border/65 bg-(--composer-fill) px-3 py-1 text-xs text-muted-foreground hover:text-foreground"
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  )
}
