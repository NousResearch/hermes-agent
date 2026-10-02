import type { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalPendingAction } from './canonical-groups'

type Labels = ReturnType<typeof useCanonicalGroupLabels>
const text = (value: unknown) => (typeof value === 'string' ? value.trim() : '')

interface EditPreview {
  path: string
  before: string | null
  after: string
  patch: boolean
}

function editPreview(value: unknown): EditPreview | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const edit = value as Record<string, unknown>

  if (
    !text(edit.path) ||
    !['write_file', 'patch'].includes(String(edit.tool_name)) ||
    typeof edit.new_text !== 'string' ||
    (edit.old_text !== null && typeof edit.old_text !== 'string')
  ) {
    return null
  }

  const patch = edit.tool_name === 'patch' && edit.old_text === null

  if (patch && !edit.new_text.trimStart().startsWith('*** Begin Patch')) {
    return null
  }

  return { path: text(edit.path), before: edit.old_text as string | null, after: edit.new_text, patch }
}

export function canonicalApprovalPreview(action: CanonicalPendingAction) {
  const approval = action.approval
  const command = text(approval?.command)
  const description = text(approval?.description)
  // The ordinary chat approval card uses descriptions for synthetic plugin
  // labels too. A tool label alone is not an operation the user can review.
  const actualCommand = /^<[^>]+> \(/.test(command) ? '' : command
  const edit = approval?.edit === undefined ? undefined : editPreview(approval.edit)

  const matching =
    Boolean(text(action.request_id)) &&
    (!approval?.request_id || approval.request_id === action.request_id) &&
    (!approval?.prompt_id || approval.prompt_id === action.request_id)

  const reviewable = matching && edit !== null && Boolean(actualCommand || edit)

  return { actualCommand, description, edit, reviewable }
}

export function canonicalApprovalDetails({ action, labels }: { action: CanonicalPendingAction; labels: Labels }) {
  const { actualCommand, description, edit, reviewable } = canonicalApprovalPreview(action)

  return {
    reviewable,
    content: (
      <div className="grid min-w-0 gap-2">
        {description && description !== actualCommand && (
          <div>
            <p className="text-xs text-(--ui-text-secondary)">{labels.approvalAction}</p>
            <p className="whitespace-pre-wrap break-words text-sm">{description}</p>
          </div>
        )}
        {actualCommand && !edit && (
          <div>
            <p className="text-xs text-(--ui-text-secondary)">{labels.approvalCommand}</p>
            <pre className="m-0 max-h-40 overflow-auto whitespace-pre-wrap break-words py-2 font-mono text-xs leading-relaxed text-(--ui-text-primary)">
              {actualCommand}
            </pre>
          </div>
        )}
        {edit && (
          <div className="grid min-w-0 gap-2">
            <p className="text-xs text-(--ui-text-secondary)">{labels.approvalChanges}</p>
            <p className="break-all font-mono text-xs">{edit.path}</p>
            {edit.before !== null && (
              <div>
                <p className="text-xs text-(--ui-text-secondary)">{labels.approvalBefore}</p>
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
                  {edit.before || labels.approvalEmptyFile}
                </pre>
              </div>
            )}
            {!edit.patch && <p className="text-xs text-(--ui-text-secondary)">{labels.approvalAfter}</p>}
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
              {edit.after || labels.approvalEmptyFile}
            </pre>
          </div>
        )}
        {!reviewable && (
          <p className="text-sm text-(--ui-text-secondary)" role="status">
            {labels.approvalDetailsMissing}
          </p>
        )}
      </div>
    )
  }
}
