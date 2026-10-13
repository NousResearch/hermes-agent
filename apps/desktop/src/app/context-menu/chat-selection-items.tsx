import { useStore } from '@nanostores/react'

import { DropdownMenuItem } from '@/components/ui/dropdown-menu'
import type { Translations } from '@/i18n'
import { isMacPlatform } from '@/lib/platform'
import { stopVoicePlayback } from '@/lib/voice-playback'
import { notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'

import type { ChatSelection } from './chat-selection'
import { type ChatSelectionAction, runChatSelectionAction } from './chat-selection-actions'

export function ChatSelectionItems({ selection, t }: { selection: ChatSelection; t: Translations }) {
  const playback = useStore($voicePlayback)

  const run = (action: ChatSelectionAction) => {
    void runChatSelectionAction(action, selection).catch(error =>
      notifyError(error, action === 'read-aloud' ? t.notifications.voice.playbackFailed : t.selectionTranslate.failed)
    )
  }

  return (
    <>
      <DropdownMenuItem onSelect={() => run('copy')}>{t.common.copy}</DropdownMenuItem>
      <DropdownMenuItem onSelect={() => run('read-aloud')}>{t.selectionActions.readAloud}</DropdownMenuItem>
      {playback.status !== 'idle' && (
        <DropdownMenuItem onSelect={stopVoicePlayback}>{t.selectionActions.stop}</DropdownMenuItem>
      )}
      {isMacPlatform() && window.hermesDesktop?.contextMenuLookUp && (
        <DropdownMenuItem onSelect={() => run('lookup')}>{t.selectionActions.lookUp}</DropdownMenuItem>
      )}
      <DropdownMenuItem onSelect={() => run('translate')}>{t.selectionActions.translate}</DropdownMenuItem>
    </>
  )
}
