import { Button, Codicon, Input, useI18n } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { updateGroupChat } from './group-chat'
import { boundedThreadTitle, GROUP_THREAD_TITLE_CHARS } from './group-thread-titles'
import { useBots } from './i18n'
import { stripPreviewMarkdown } from './labels'

interface GroupThreadHeaderProps {
  group: string
  thread: string
  rootText: string
  customTitle?: string
  collapsed: boolean
  replies: number
  onToggle: () => void
}

/** A disclosure at the first arrival, never a bucket that regroups messages. */
export function GroupThreadHeader({ group, thread, rootText, customTitle, collapsed, replies, onToggle }: GroupThreadHeaderProps) {
  const b = useBots()
  const { t } = useI18n()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const title = boundedThreadTitle(customTitle || '') || boundedThreadTitle(stripPreviewMarkdown(rootText)) || b.group.threadFallback

  return (
    <div className="flex min-w-0 items-center gap-2 pt-2 text-xs text-(--ui-text-secondary)">
      <Button
        aria-expanded={!collapsed}
        aria-label={`${collapsed ? b.group.openThread : b.group.collapseThreadLabel}: ${title}`}
        onClick={onToggle}
        size="inline"
        title={collapsed ? b.group.openThread : b.group.collapseThread}
        variant="ghost"
      >
        <Codicon name={collapsed ? 'chevron-right' : 'chevron-down'} />
      </Button>
      {editing ? (
        <form
          className="flex min-w-0 flex-1 items-center gap-2"
          onSubmit={event => {
            event.preventDefault()
            updateGroupChat(group, room => ({
              ...room,
              threadTitles: { ...room.threadTitles, [thread]: boundedThreadTitle(draft) }
            }))
            setEditing(false)
          }}
        >
          <Input
            aria-label={b.sections.rename}
            autoFocus
            maxLength={GROUP_THREAD_TITLE_CHARS * 2}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Escape') {
                event.preventDefault()
                event.stopPropagation()
                setEditing(false)
              }
            }}
            value={draft}
          />
          <Button size="inline" type="submit" variant="text">{t.common.save}</Button>
          <Button onClick={() => setEditing(false)} size="inline" type="button" variant="text">{t.common.cancel}</Button>
        </form>
      ) : (
        <>
          <span className="min-w-0 flex-1 truncate font-medium" title={title}>{title}</span>
          <span className="shrink-0 text-(--ui-text-quaternary)">{b.group.replyCount(replies)}</span>
          <Button
            aria-label={`${b.sections.rename}: ${title}`}
            onClick={() => { setDraft(title); setEditing(true) }}
            size="inline"
            title={b.sections.rename}
            variant="ghost"
          >
            <Codicon name="edit" />
          </Button>
        </>
      )}
    </div>
  )
}
