/**
 * Semantic @mention rendering for group-room transcripts (#91359).
 *
 * Presentation only: the stored text, the copy button, prompt payloads and
 * `parseGroupChatMentions` are untouched. At render time every `@token` in a
 * text node is classified with the SAME parser the round loop uses, so a token
 * is styled exactly when routing would honour it — unknown handles, e-mail
 * addresses and Matrix ids stay plain prose. Recognized mentions become the
 * app's inline-reference form (`.ref` + `data-ref` kind: `agent` / `human` /
 * `broadcast`), i.e. accent text with weight, never a pill.
 */

import type { ComponentProps, ReactElement, ReactNode } from 'react'
import { Children, cloneElement, Fragment, isValidElement } from 'react'

import { parseGroupChatMentions } from './group-rounds'
import type { GroupMember } from './types'

export type GroupMentionKind = 'agent' | 'broadcast' | 'human'

const MENTION = /@([a-z0-9][a-z0-9._-]*)/gi

/** Host inlines Streamdown wraps bot markdown in (`**@bot**`, `_@bot_`). */
const NESTED_INLINE = new Set(['b', 'del', 'em', 'i', 'mark', 'small', 'span', 'strong', 'u'])

/** What a lone `@token` means in this room, or null when routing ignores it. */
export function classifyGroupMention(token: string, members: GroupMember[]): GroupMentionKind | null {
  const handle = token.toLowerCase()

  if (handle === 'user') {
    return 'human'
  }

  const parsed = parseGroupChatMentions(`@${token}`, members)

  if (parsed.everyone) {
    return 'broadcast'
  }

  return parsed.mentioned.size ? 'agent' : null
}

const TITLES: Record<GroupMentionKind, string> = {
  agent: 'Bot in this room',
  broadcast: 'Everyone in the room',
  human: 'You — the room is waiting on a human'
}

/** Split one text node into prose and recognized mention spans. */
export function renderGroupMentionText(text: string, members: GroupMember[]): ReactNode {
  const parts: ReactNode[] = []
  let last = 0

  for (const match of text.matchAll(MENTION)) {
    const kind = classifyGroupMention(match[1], members)

    if (!kind) {
      continue
    }

    const start = match.index ?? 0

    if (start > last) {
      parts.push(text.slice(last, start))
    }

    parts.push(
      <span className="ref font-medium" data-ref={kind} key={`${start}:${match[0]}`} title={TITLES[kind]}>
        {match[0]}
      </span>
    )
    last = start + match[0].length
  }

  if (!parts.length) {
    return text
  }

  if (last < text.length) {
    parts.push(text.slice(last))
  }

  return <Fragment>{parts}</Fragment>
}

function mapTextChildren(children: ReactNode, members: GroupMember[]): ReactNode {
  return Children.map(children, child => {
    if (typeof child === 'string') {
      return renderGroupMentionText(child, members)
    }

    if (!isValidElement(child)) {
      return child
    }

    const element = child as ReactElement<{ children?: ReactNode }>

    if (element.type === Fragment) {
      return <Fragment>{mapTextChildren(element.props.children, members)}</Fragment>
    }

    if (typeof element.type === 'string' && NESTED_INLINE.has(element.type)) {
      return cloneElement(element, undefined, mapTextChildren(element.props.children, members))
    }

    return child
  })
}

/** `decorateText` hook for the shell's `MessageTextContent`: the same
 *  splitter, applied to the direct text nodes of its paragraph-level
 *  containers (p / li / td). Nested emphasis is walked so a bot's
 *  `**@name**` still colours; code and links stay literal. */
export function groupMentionText(members: GroupMember[]) {
  return (children: ReactNode) => mapTextChildren(children, members)
}

/** Streamdown `components` override (older shells without
 *  `MessageTextContent`): paragraph-level containers re-render
 *  their text through the mention splitter. Nested emphasis is walked
 *  so a bot's `**@name**` still colours; a mention inside a backtick
 *  or a link stays literal on purpose. */
export function groupMentionComponents(members: GroupMember[]) {
  return {
    li: ({ children, ...props }: ComponentProps<'li'>) => <li {...props}>{mapTextChildren(children, members)}</li>,
    p: ({ children, ...props }: ComponentProps<'p'>) => <p {...props}>{mapTextChildren(children, members)}</p>,
    td: ({ children, ...props }: ComponentProps<'td'>) => <td {...props}>{mapTextChildren(children, members)}</td>
  }
}
