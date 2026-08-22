/**
 * Roadmaps plugin — Files view.
 *
 * Honest empty state (no fixtures): files linked to nodes and versions will
 * be listed here once evidence lands backend.
 */

import { EmptyState } from '@hermes/plugin-sdk'

export function FilesView() {
  return (
    <EmptyState
      description="Evidence linked to nodes and versions will be listed here once it lands backend."
      title="No attached files"
    />
  )
}
