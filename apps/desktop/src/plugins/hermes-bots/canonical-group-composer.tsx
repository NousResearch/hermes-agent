import { composerPanelCard, RowButton, Textarea } from '@hermes/plugin-sdk'
import { useId, useRef, useState } from 'react'

import { CanonicalMemberFace, canonicalMemberName } from './canonical-group-identity'
import { useCanonicalGroupLabels } from './canonical-group-labels'
import type { CanonicalRoomMember } from './canonical-groups'

interface MentionToken { start: number; end: number; query: string }

function mentionAt(value: string, caret: number): MentionToken | null {
  const match = /(^|\s)@([a-z0-9._-]*)$/i.exec(value.slice(0, caret))

  return match ? { start: caret - match[2].length - 1, end: caret, query: match[2].toLowerCase() } : null
}

/** Completes owner-issued handles while showing the group's friendly Bot names. */
export function CanonicalGroupComposerInput({ members, value, disabled, name, onChange, onSubmit }: {
  members: CanonicalRoomMember[]; value: string; disabled: boolean; name: string
  onChange: (value: string) => void; onSubmit: () => void
}) {
  const labels = useCanonicalGroupLabels()
  const input = useRef<HTMLTextAreaElement>(null)
  const listId = useId()
  const [token, setToken] = useState<MentionToken | null>(null)
  const [selected, setSelected] = useState(0)

  const options = [
    { handle: 'all', name: labels.everyone, member: undefined },
    ...members.filter(member => member.handle?.trim()).map(member => ({
      handle: member.handle, name: canonicalMemberName(member, labels.unknownBot), member
    }))
  ].filter(option => !token?.query || option.name.toLowerCase().startsWith(token.query) || option.handle.toLowerCase().startsWith(token.query))

  const open = !disabled && !!token && options.length > 0
  const active = Math.min(selected, options.length - 1)

  const locate = (target: HTMLTextAreaElement) => {
    setToken(mentionAt(target.value, target.selectionStart ?? target.value.length)); setSelected(0)
  }

  const insert = (handle: string) => {
    if (!token || disabled) {return}
    const node = input.current
    const current = node && mentionAt(node.value, node.selectionStart ?? node.value.length)

    if (!current || current.start !== token.start || current.end !== token.end || current.query !== token.query) {
      setToken(current); setSelected(0)

      return
    }

    onChange(`${value.slice(0, token.start)}@${handle} ${value.slice(token.end)}`)
    const caret = token.start + handle.length + 2
    setToken(null)
    requestAnimationFrame(() => { input.current?.focus(); input.current?.setSelectionRange(caret, caret) })
  }

  return <div className="relative min-w-0 flex-1">
    {open && <div aria-label={labels.members} className={`${composerPanelCard} absolute bottom-full left-0 z-50 mb-2 max-h-48 w-72 max-w-full overflow-y-auto p-1`} id={listId} role="listbox">
      {options.map((option, index) => <RowButton aria-selected={index === active}
        className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-xs text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-selected:bg-(--chrome-action-hover) aria-selected:text-(--ui-text-primary)"
        id={`${listId}-${index}`} key={option.member?.member_id ?? option.handle} onClick={() => insert(option.handle)}
        onMouseDown={event => { event.preventDefault(); insert(option.handle) }} onMouseEnter={() => setSelected(index)} role="option" tabIndex={-1}>
        {option.member && <CanonicalMemberFace member={option.member} name={option.name} size={20} />}
        <span className="min-w-0 truncate">{option.name}</span>
      </RowButton>)}
    </div>}
    <Textarea aria-activedescendant={open ? `${listId}-${active}` : undefined} aria-autocomplete="list"
      aria-controls={open ? listId : undefined} aria-label={labels.groupMessage}
      className="field-sizing-content max-h-[min(40vh,16rem)] min-h-12 resize-none overflow-y-auto text-[length:var(--conversation-text-font-size)] leading-(--conversation-line-height)"
      disabled={disabled} onBlur={() => setToken(null)} onChange={event => { onChange(event.target.value); locate(event.target) }}
      onClick={event => locate(event.currentTarget)} onKeyDown={event => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) {return}

        if (open && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
          event.preventDefault(); setSelected((active + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length)

          return
        }

        if (open && (event.key === 'Enter' || event.key === 'Tab')) {event.preventDefault(); insert(options[active].handle);

 return}

        if (event.key === 'Escape') {setToken(null);

 return}

        if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault(); onSubmit()}
      }} onSelect={event => locate(event.currentTarget)} placeholder={labels.messagePlaceholder.replace('{name}', name)} ref={input} value={value} variant="plain" />
  </div>
}
