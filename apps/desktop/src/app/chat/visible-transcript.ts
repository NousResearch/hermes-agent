import { computed } from 'nanostores'
import { useEffect, useMemo, useState } from 'react'

import { type SessionTranscript, type SessionView } from '@/app/chat/session-view'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'

/**
 * The transcript payload and its owner, adopted as one snapshot.
 *
 * The primary view can publish a destination runtime/cwd before React replaces
 * the outgoing assistant-ui rows. Reading identity separately in those rows
 * lets their passive effects poison the destination session. This hook keeps
 * the last rendered snapshot through that commit, then adopts the view's live
 * messages and identity together. Hidden panes remain frozen until revealed.
 */
export function useVisibleTranscriptSnapshot(view: SessionView): SessionTranscript {
  const visible = usePaneVisible()

  const transcriptStore = useMemo(
    () =>
      view.$transcript ??
      computed([view.$runtimeId, view.$cwd, view.$messages], (runtimeId, cwd, messages) => ({
        identity: { cwd, runtimeId },
        messages
      })),
    [view]
  )

  const [snapshot, setSnapshot] = useState<SessionTranscript>(() => transcriptStore.get())

  useEffect(() => (visible ? transcriptStore.subscribe(setSnapshot) : undefined), [transcriptStore, visible])

  return snapshot
}
