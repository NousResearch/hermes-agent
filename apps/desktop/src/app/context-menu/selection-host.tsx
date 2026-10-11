import { useStore } from '@nanostores/react'

import { SelectionTranslateDialog } from '@/app/chat/composer/selection-translate-dialog'
import { Button } from '@/components/ui/button'
import { registry } from '@/contrib/registry'
import { useI18n } from '@/i18n'
import { stopVoicePlayback } from '@/lib/voice-playback'
import { $voicePlayback } from '@/store/voice-playback'

function SelectionActionsHost() {
  const playback = useStore($voicePlayback)
  const { t } = useI18n()

  return (
    <>
      {playback.messageId === 'selection-read-aloud' && playback.status !== 'idle' && (
        <Button onClick={stopVoicePlayback} size="sm" variant="ghost">
          {t.selectionActions.stop}
        </Button>
      )}
      <SelectionTranslateDialog />
    </>
  )
}

// One host per full app window, independent of the number of open chat panes.
registry.register({
  area: 'titleBar.center',
  id: 'selection-actions.host',
  render: () => <SelectionActionsHost />,
  source: 'core'
})
