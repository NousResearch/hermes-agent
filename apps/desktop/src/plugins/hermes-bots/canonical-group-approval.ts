import type { CanonicalPendingAction } from './canonical-groups'

const text = (value: unknown) => (typeof value === 'string' ? value.trim() : '')

/** What an approval would allow. Allow once is offered only for this exact pending request and
 *  only when it shows a real command or the description a synthetic tool label stands for: a
 *  description with no command or label, or a label with no description, authorizes nothing. */
export function canonicalApprovalPreview(action: CanonicalPendingAction) {
  const approval = action.approval
  const command = text(approval?.command)
  const description = text(approval?.description)
  // A synthetic tool label ("<tool> (…)", as a plugin approval rule sends from
  // tools/approval.py request_tool_approval) names no command: its description is the
  // operation, which is what the chat approval card shows in its place.
  const labelled = /^<[^>]+> \(/.test(command)
  const actualCommand = labelled ? '' : command

  const matching = Boolean(text(action.request_id)) &&
    (!approval?.request_id || approval.request_id === action.request_id) &&
    (!approval?.prompt_id || approval.prompt_id === action.request_id)

  return { actualCommand, description, reviewable: matching && Boolean(actualCommand || (labelled && description)) }
}
