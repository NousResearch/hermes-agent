/**
 * PairBotsDialog — introduce two bots and watch them work.
 *
 * Validation, instruction composition, and the fire-and-forget contract live
 * in `bot-pairing.ts`; this component is the picker + the submit. On submit it
 * opens the initiator's canonical chat (which focuses the workspace so the
 * user sees the conversation happen) and submits the pairing instruction as a
 * real user turn — the initiator's `message_agent` tool does the rest.
 */
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Codicon } from '@/components/ui/codicon'
import { Textarea } from '@/components/ui/textarea'

import { BotFace, avatarColor, botAppearance } from './avatar'
import { pairingInstruction, pairingProblem, pairableBots } from './bot-pairing'
import { $botMeta } from './data'
import { openBotCanonicalChat } from './canonical-chat'
import { displayName } from './labels'
import { requestForBot } from './routing'
import { useValue } from '@hermes/plugin-sdk'
import type { RosterRow } from './types'

interface PairBotsDialogProps {
  open: boolean
  onClose: () => void
  roster: RosterRow[]
  /** The currently focused bot, preselected as initiator when present. */
  focusedBot?: RosterRow | null
}

export function PairBotsDialog({ open, onClose, roster, focusedBot }: PairBotsDialogProps) {
  const allMeta = useValue($botMeta)
  const candidates = useMemo(() => pairableBots(roster), [roster])

  const [initiator, setInitiator] = useState<RosterRow | null>(
    focusedBot && !focusedBot.ghost ? focusedBot : null
  )
  const [responder, setResponder] = useState<RosterRow | null>(null)
  const [task, setTask] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const problem = open ? pairingProblem(initiator, responder) : 'Pick both bots.'

  const picker = (
    label: string,
    value: RosterRow | null,
    onPick: (bot: RosterRow) => void,
    exclude: RosterRow | null
  ) => (
    <div className="grid gap-1.5">
      <span className="text-xs text-(--ui-text-tertiary)">{label}</span>
      <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
        {candidates
          .filter(bot => bot !== exclude)
          .map(bot => {
            const picked = value === bot
            const other = value === bot ? null : exclude
            const appearance = botAppearance(bot.name, allMeta[bot.name])

            return (
              <button
                className={`flex items-center gap-1.5 rounded-md border px-2 py-1.5 text-left transition-colors duration-100 ${
                  picked
                    ? 'border-(--ui-accent) bg-(--ui-row-active-background)'
                    : 'border-(--ui-stroke-tertiary) hover:bg-(--chrome-action-hover)'
                } ${other ? 'opacity-40' : ''}`}
                key={bot.name}
                onClick={() => onPick(bot)}
                type="button"
              >
                <BotFace
                  color={avatarColor(appearance.color, bot.name)}
                  image={appearance.image ?? null}
                  name={bot.name}
                  shape={appearance.shape}
                  size={22}
                />
                <span className="text-xs font-medium">{displayName(bot, allMeta[bot.name])}</span>
                {picked ? <Codicon className="text-[0.6875rem] text-(--ui-accent)" name="check" /> : null}
              </button>
            )
          })}
      </div>
    </div>
  )

  const submit = async () => {
    if (problem || submitting || !initiator || !responder) {
      return
    }

    setSubmitting(true)

    try {
      // Open (or adopt) the initiator's canonical chat first so the user is
      // LOOKING at where the conversation will land, then submit the pairing
      // instruction as a user turn into that session (prompt.submit is the
      // user-turn API — attributed, persisted, prompt-cached).
      const opened = await openBotCanonicalChat(initiator)
      const sessionId = opened?.openedId

      if (!sessionId) {
        return
      }

      await requestForBot(initiator, 'prompt.submit', {
        session_id: sessionId,
        text: pairingInstruction({ initiator, responder, task })
      })

      onClose()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={next => (!next ? onClose() : undefined)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Introduce bots</DialogTitle>
          <DialogDescription>
            Pick two agents; the first messages the second through their own chats and they take it
            from there. You keep both Bot Chats on screen.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3.5">
          {picker('Who reaches out first', initiator, setInitiator, responder)}
          {picker('Who receives', responder, setResponder, initiator)}

          {labeledTask(task, setTask)}

          {problem && initiator && responder ? (
            <div className="text-xs text-(--ui-accent)">{problem}</div>
          ) : null}

          <div className="flex items-center justify-end gap-2">
            <Button onClick={onClose} size="sm" type="button" variant="ghost">
              Cancel
            </Button>
            <Button
              disabled={Boolean(problem) || submitting}
              onClick={() => void submit()}
              size="sm"
              type="button"
            >
              {submitting ? 'Introducing…' : 'Introduce'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function labeledTask(task: string, setTask: (value: string) => void) {
  return (
    <div className="grid gap-1.5">
      <span className="text-xs text-(--ui-text-tertiary)">Joint task (optional)</span>
      <Textarea
        className="min-h-16 resize-none"
        onChange={event => setTask(event.target.value)}
        placeholder="e.g. review the login flow together and agree on a fix"
        value={task}
      />
    </div>
  )
}
