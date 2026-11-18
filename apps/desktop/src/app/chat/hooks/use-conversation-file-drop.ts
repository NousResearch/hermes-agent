import { useCallback } from 'react'

import { type ComposerTarget, requestComposerInsert } from '../composer/focus'
import { droppedFileInlineRefs } from '../composer/inline-refs'

import { type DroppedFile, partitionDroppedFiles } from './use-composer-actions'
import { useFileDropZone } from './use-file-drop-zone'

interface ConversationFileDropOptions {
  /** The chat's working directory: in-app paths become refs relative to it. */
  cwd: null | string | undefined
  enabled: boolean
  /** The composer the refs land in. */
  target: ComposerTarget
  onAttachDroppedItems: (candidates: DroppedFile[]) => unknown
}

/**
 * Drop files anywhere in the conversation area, not just on the composer input. In-app drags
 * (project tree / gutter) carry workspace-relative paths the gateway resolves directly, so they stay
 * inline `@file:` refs. OS/Finder drops carry absolute local paths that don't exist on a remote
 * gateway (and images need byte upload for vision), so they go through the attachment pipeline —
 * otherwise the local path leaks into the prompt verbatim.
 */
export function useConversationFileDrop({ cwd, enabled, onAttachDroppedItems, target }: ConversationFileDropOptions) {
  const onDropFiles = useCallback(
    (candidates: DroppedFile[]) => {
      const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
      const refs = droppedFileInlineRefs(inAppRefs, cwd)

      if (refs.length) {
        requestComposerInsert(refs.join(' '), { mode: 'inline', target })
      }

      if (osDrops.length) {
        void onAttachDroppedItems(osDrops)
      }
    },
    [cwd, onAttachDroppedItems, target]
  )

  // Session drags are POINTER drags (session-drag.ts) — never native DnD. The zone only handles
  // files; session drops commit through the drag session itself, which routes a center/link drop to
  // this surface's composer via `data-composer-target`.
  return useFileDropZone({ enabled, onDropFiles })
}
