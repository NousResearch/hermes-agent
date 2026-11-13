/**
 * The plain-English "what this does" line on an approval card.
 *
 * An approval prompt shows the raw command and, for a synthetic command, the detector's reason
 * (`request.description`). Neither helps someone who cannot read the command, so the backend can
 * send one extra sentence explaining what it is about to do (`approvals.explain`, rendered here
 * above the command). Display-only and optional: absent or blank renders nothing.
 */
export function ApprovalExplanation({ text }: { text?: string }) {
  const line = text?.trim()

  if (!line) {
    return null
  }

  return (
    <p
      className="m-0 px-2.5 pb-1 text-xs leading-relaxed text-(--ui-text-secondary)"
      data-slot="tool-approval-explanation"
    >
      {line}
    </p>
  )
}
