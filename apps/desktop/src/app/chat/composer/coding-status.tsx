import { useStore } from '@nanostores/react'
import type { MutableRefObject } from 'react'

import { useStoresSelector } from '@/lib/use-session-slice'
import { $showsAdvancedChrome } from '@/store/interface-mode'
import { toggleReview } from '@/store/review'
import { $botChatSessionIds, $sessionStates, $sessionTiles, isBotChatSession } from '@/store/session-states'

import { useComposerBranch } from './hooks/use-composer-branch'
import { useComposerScope } from './scope'
import { CodingStatusRow } from './status-stack/coding-row'
import { StatusDrawerContent } from './status-stack/drawer'

interface ComposerCodingStatusProps {
  clearDraft: () => void
  collapsed: boolean
  cwd: string | null | undefined
  draftRef: MutableRefObject<string>
  guidedChat: boolean
  id: string
  sessionId: string | null | undefined
}

/** Repo actions follow this composer's scope and carry its draft into a worktree. */
export function ComposerCodingStatus({
  clearDraft,
  collapsed,
  cwd,
  draftRef,
  guidedChat,
  id,
  sessionId
}: ComposerCodingStatusProps) {
  const scope = useComposerScope()
  const showsAdvancedChrome = useStore($showsAdvancedChrome)
  const botChat = useStoresSelector([$botChatSessionIds, $sessionStates, $sessionTiles], () =>
    isBotChatSession(sessionId)
  )
  const { handleBranchOff, handleConvertBranch, handleListBranches, handleSwitchBranch, openInWorktree } =
    useComposerBranch({ clearDraft, cwd, draftRef })

  if (guidedChat || !showsAdvancedChrome) {
    return null
  }

  return (
    <StatusDrawerContent collapsed={collapsed} id={id}>
      <CodingStatusRow
        onBranchOff={handleBranchOff}
        onConvertBranch={handleConvertBranch}
        onListBranches={handleListBranches}
        // Tiles review their own worktree; main follows the active session.
        onOpen={() => toggleReview(scope.target === 'main' ? null : (cwd ?? null), scope.target)}
        onOpenWorktree={openInWorktree}
        onSwitchBranch={handleSwitchBranch}
        // Companion chats have no repo: hide the row and avoid git/GitHub probes.
        repoPath={botChat ? undefined : cwd}
      />
    </StatusDrawerContent>
  )
}
