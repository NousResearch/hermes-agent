import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $desktopCommandsCatalog, reviewPromptButtons } from '@/lib/desktop-slash-commands'
import { notifyError } from '@/store/notifications'
import { $reviewFiles, $reviewScopeTarget } from '@/store/review'

// Saved prompts as Review pane actions: a `type: prompt` quick command with
// `review_button` set in config.yaml becomes a button here. A click submits
// `/<name>` into the chat that owns the changes, so it runs through the same
// slash dispatch as typing it (busy → queued, args → none).
export function ReviewPromptButtons() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const catalog = useStore($desktopCommandsCatalog)
  const files = useStore($reviewFiles)
  const scopeTarget = useStore($reviewScopeTarget)
  const buttons = useMemo(() => reviewPromptButtons(catalog), [catalog])

  if (!buttons.length || !files.length) {
    return null
  }

  return (
    <div className="flex shrink-0 flex-wrap gap-1 px-2 pt-1.5" data-suppress-pane-reveal-side="">
      {buttons.map(({ command, label }) => (
        <Tip key={command} label={command}>
          <Button
            className="h-6 min-w-0 max-w-full px-2 text-[0.7rem]"
            onClick={() => {
              if (!requestComposerSubmit(command, { target: scopeTarget })) {
                notifyError(new Error(c.agentShipUnavailable), label)
              }
            }}
            size="sm"
            variant="outline"
          >
            <span className="truncate">{label}</span>
          </Button>
        </Tip>
      ))}
    </div>
  )
}
