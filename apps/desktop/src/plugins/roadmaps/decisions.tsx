/**
 * Roadmaps plugin — Decisions view.
 *
 * Honest empty state (no fixtures): plan governance decisions (propose,
 * validate, activate, revise) are surfaced through the Plan view today; a
 * dedicated decision log is a later tranche.
 */

import { EmptyState } from '@hermes/plugin-sdk'

export function DecisionsView() {
  return (
    <EmptyState
      description="Plan governance runs in the Plan view (propose, validate, activate). A dedicated decision log is a later tranche."
      title="No decisions recorded"
    />
  )
}
