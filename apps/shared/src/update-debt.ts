// Owed post-commit work of a dashboard-started `hermes update`, read from the
// receipt summary the action status route attaches. Shared by the web
// dashboard toast and the Desktop backend-apply store so both surfaces name
// the same debt with the same rules as the managed SSH result.

export interface UpdateDebtStep {
  step: string
  reason?: string | null
}

export interface UpdateDebtReceipt {
  outcome?: string | null
  action_id?: string | null
  followups?: UpdateDebtStep[] | null
  user_action?: UpdateDebtStep | null
}

export interface UpdateDebt {
  /** Follow-ups a rerun of `hermes update` finishes, as `step (reason); step`; '' when none. */
  followups: string
  /** What only the user can do (e.g. re-apply a parked stash) as `step: <producer's instruction>`;
   *  '' when none. Never phrase this as "re-run hermes update": a rerun does not restore it. */
  userAction: string
}

/** The debt of THIS action's committed run, or null when nothing is owed.
 *
 *  Committed is `success` or `partial` (record_user_action turns a committed success into
 *  partial / exit 1), so the debt is read for every committed outcome, not only exit 0. The
 *  status route attaches the latest receipt when this action has none, so a receipt is only
 *  this action's when its `action_id` matches the id the start request returned. */
export function updateDebt(receipt: UpdateDebtReceipt | null | undefined, actionId: string | undefined): UpdateDebt | null {
  if (!receipt || !actionId || receipt.action_id !== actionId) {
    return null
  }

  if (receipt.outcome !== 'success' && receipt.outcome !== 'partial') {
    return null
  }

  const followups = (receipt.followups ?? [])
    .map(step => (step.reason ? `${step.step} (${step.reason})` : step.step))
    .join('; ')

  const action = receipt.user_action
  const userAction = action ? (action.reason ? `${action.step}: ${action.reason}` : action.step) : ''

  return followups || userAction ? { followups, userAction } : null
}
