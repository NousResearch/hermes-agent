import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { GenerateButton } from '@/components/ui/generate-button'
import { SplitButton } from '@/components/ui/split-button'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import {
  $reviewCommitDefault,
  $reviewCommitMsgBusy,
  $reviewFiles,
  $reviewShipBusy,
  $reviewShipInfo,
  cancelCommitMessage,
  type CommitAction,
  commitChanges,
  createOrOpenPr,
  generateCommitMessage
} from '@/store/review'

// One size for every glyph in the bar so the row reads as a set of peers.
const ICON = '0.85rem'

// The commit / push / PR action bar at the bottom of the review pane. Supports
// both paths: the user drives it directly, OR hands the whole thing to the agent
// with one click (requestComposerSubmit sends it a task through the composer).
export function ReviewShipBar() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const files = useStore($reviewFiles)
  const ship = useStore($reviewShipInfo)
  const busy = useStore($reviewShipBusy)
  const generating = useStore($reviewCommitMsgBusy)
  const commitDefault = useStore($reviewCommitDefault)
  const [message, setMessage] = useState('')
  const prLabel = ship.pr?.url ? c.openPr : c.createPr

  const hasFiles = files.length > 0
  const canCommit = hasFiles && message.trim().length > 0 && !busy
  const canGenerate = hasFiles && !generating && !busy

  // Nothing to commit → no ship bar at all; the pane just shows the tree /
  // "No changes" state.
  if (!hasFiles) {
    return null
  }

  const runCommit = (action: CommitAction) => {
    if (!canCommit) {
      return
    }

    void commitChanges(message, { push: action === 'commitPush' })
      .then(() => setMessage(''))
      .catch(err => notifyError(err, c.commit))
  }

  // Draft the commit message off-thread (VS Code style); pass the current text
  // so a re-press regenerates instead of returning the same thing.
  const runGenerate = () => {
    if (!canGenerate) {
      return
    }

    void generateCommitMessage(message)
      .then(text => text && setMessage(text))
      .catch(err => notifyError(err, c.generateCommitMessage))
  }

  return (
    <div className="flex shrink-0 flex-col gap-1.5 p-2" data-suppress-pane-reveal-side="">
      {/* Auto-growing message field (CSS field-sizing); generate/stop action
          fills the right edge on one row, then sticks to the top as it grows. */}
      <div className="relative">
        <Textarea
          className="field-sizing-content max-h-40 min-h-0 resize-none pr-9"
          disabled={generating}
          onChange={event => setMessage(event.target.value)}
          onKeyDown={event => {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
              event.preventDefault()
              runCommit(commitDefault)
            }
          }}
          placeholder={c.commitPlaceholder}
          rows={1}
          size="sm"
          value={message}
        />
        <GenerateButton
          className="absolute top-px right-px h-6 w-8 rounded-l-none rounded-r-[2px]"
          disabled={!canGenerate}
          generating={generating}
          generatingLabel={c.stopGenerating}
          iconSize={ICON}
          label={c.generateCommitMessage}
          onCancel={cancelCommitMessage}
          onGenerate={runGenerate}
        />
      </div>

      {/* Commit split (VS Code style). */}
      <div className="flex min-w-0">
        <SplitButton
          actions={[
            { id: 'commit', label: c.commit },
            { id: 'commitPush', label: c.commitAndPush }
          ]}
          className="min-w-0 flex-1"
          disabled={!canCommit}
          onTrigger={id => runCommit(id as CommitAction)}
          onValueChange={id => $reviewCommitDefault.set(id as CommitAction)}
          primaryIcon={<Codicon name="check" size={ICON} />}
          value={commitDefault}
          variant="default"
        />
      </div>

      {/* Hand it to the agent (one click sends a commit+PR task to the composer).
          The PR button floats on the right (out of flow) so the label centers on
          the whole bar; px-7 reserves the icon's width on both sides. */}
      <div className="relative flex min-w-0 items-center">
        <Button
          className="min-w-0 flex-1 justify-center px-7 text-[0.7rem] text-muted-foreground/85 hover:text-foreground"
          disabled={!hasFiles}
          onClick={() => requestComposerSubmit(c.agentShipPrompt, { target: 'main' })}
          size="sm"
          variant="ghost"
        >
          <span className="truncate underline underline-offset-2">{c.agentShip}</span>
        </Button>
        <Tip label={ship.ghReady ? prLabel : c.ghMissing}>
          <span className="absolute inset-y-0 right-0 flex items-center">
            <Button
              aria-label={prLabel}
              className="size-7 text-muted-foreground/80 hover:text-foreground"
              disabled={!ship.ghReady || busy}
              onClick={() => void createOrOpenPr().catch(err => notifyError(err, prLabel))}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="git-pull-request" size={ICON} />
            </Button>
          </span>
        </Tip>
      </div>
    </div>
  )
}
