/**
 * The causal trail under one room message: what put this agent on turn, and
 * what it had read when it answered.
 *
 * Presentation only. It takes the room and a message id, asks `provenance` for
 * the chain, and draws it — so the walk stays pure and testable and this file
 * owns nothing but layout. Rendered inline beneath the message rather than in
 * a panel, because the question ("why did this happen") is always asked about
 * a specific line and answering it elsewhere costs the reader their place.
 */

import { Codicon } from '@hermes/plugin-sdk'

import { provenanceChain } from './provenance'
import type { GroupChat } from './types'

interface ProvenanceTrailProps {
  /** Resolve a raw profile name to what the roster calls it. */
  labelFor: (name: string) => string
  messageId: string
  room: GroupChat
}

function previewOf(text: string): string {
  const flat = String(text || '')
    .replace(/\s+/g, ' ')
    .trim()

  return flat.length > 72 ? `${flat.slice(0, 71)}…` : flat
}

export function ProvenanceTrail({ labelFor, messageId, room }: ProvenanceTrailProps) {
  const chain = provenanceChain(room, messageId)

  // One hop means nothing was recorded — a user send, or a room whose turns
  // predate cause stamping. Say so plainly instead of drawing an empty trail.
  if (chain.length < 2) {
    return (
      <div className="mt-1.5 rounded-lg border border-(--ui-stroke-secondary) px-2 py-1.5 text-[0.625rem] text-(--ui-text-quaternary)">
        No recorded cause — this room ran before turns carried provenance.
      </div>
    )
  }

  return (
    <ol className="mt-1.5 grid gap-px overflow-hidden rounded-lg border border-(--ui-stroke-secondary)">
      {chain.map((step, index) => {
        const isRoot = index === chain.length - 1
        const speaker = step.message.from.kind === 'user' ? 'You' : labelFor(step.message.from.name)

        return (
          <li
            className="flex items-start gap-2 bg-(--ui-bg-tertiary) px-2 py-1.5"
            key={step.message.id || `${step.message.at}:${index}`}
          >
            <Codicon
              className="mt-0.5 shrink-0 text-[0.6rem] text-(--ui-text-quaternary)"
              name={isRoot ? 'circle-filled' : 'chevron-up'}
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-1.5">
                <span className="text-[0.65rem] font-semibold text-(--ui-text-secondary)">{speaker}</span>
                {step.readCount ? (
                  <span className="text-[0.6rem] text-(--ui-text-quaternary)">
                    read {step.readCount} message{step.readCount === 1 ? '' : 's'}
                  </span>
                ) : null}
                {isRoot ? <span className="text-[0.6rem] text-(--ui-text-quaternary)">started this</span> : null}
              </div>
              <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)">
                {previewOf(step.message.text)}
              </div>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
