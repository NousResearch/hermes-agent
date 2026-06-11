import { useQuery } from '@tanstack/react-query'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useEffect, useMemo, useState } from 'react'

import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { listSessions } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { Check, MessageCircle } from '@/lib/icons'
import { cn } from '@/lib/utils'

interface SessionPickerDialogProps {
  /** Stored id of the session currently open, so it can be flagged in the list. */
  activeStoredSessionId?: string | null
  onOpenChange: (open: boolean) => void
  onResume: (storedSessionId: string) => void
  open: boolean
}

/**
 * Desktop equivalent of the TUI's sessions overlay (`/resume`, `/sessions`,
 * `/switch`): a focused, type-to-filter list of recent sessions that resumes
 * the picked one. Mirrors the command palette's cmdk surface but scoped to
 * sessions only, so `/resume` feels first-class instead of falling through to
 * the headless slash worker (which can't render the picker).
 */
export function SessionPickerDialog({
  activeStoredSessionId,
  onOpenChange,
  onResume,
  open
}: SessionPickerDialogProps) {
  const { t } = useI18n()
  const [search, setSearch] = useState('')

  const sessionsQuery = useQuery({
    enabled: open,
    queryFn: () => listSessions(200, 1, 'exclude'),
    queryKey: ['session-picker', 'sessions']
  })

  useEffect(() => {
    if (!open) {
      setSearch('')
    }
  }, [open])

  const sessions = useMemo(() => sessionsQuery.data?.sessions ?? [], [sessionsQuery.data])

  return (
    <DialogPrimitive.Root onOpenChange={onOpenChange} open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[200] bg-black/15 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[14vh] z-[210] w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-lg duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95"
        >
          <DialogPrimitive.Title className="sr-only">{t.commandCenter.sections.sessions}</DialogPrimitive.Title>
          <Command className="bg-transparent" loop>
            <CommandInput
              onValueChange={setSearch}
              placeholder={t.commandCenter.searchPlaceholder}
              value={search}
            />
            <CommandList className="max-h-[min(24rem,60vh)]">
              <CommandEmpty>{t.commandCenter.noResults}</CommandEmpty>
              <CommandGroup
                className="**:[[cmdk-group-heading]]:uppercase **:[[cmdk-group-heading]]:tracking-wider **:[[cmdk-group-heading]]:text-[0.6875rem] **:[[cmdk-group-heading]]:text-muted-foreground/70"
                heading={t.commandCenter.sections.sessions}
              >
                {sessions.map(session => {
                  const title = sessionTitle(session)
                  const preview = session.preview?.trim()

                  return (
                    <CommandItem
                      className="gap-2.5"
                      key={session.id}
                      onSelect={() => {
                        onResume(session.id)
                        onOpenChange(false)
                      }}
                      value={`${title} ${preview ?? ''} ${session.id}`}
                    >
                      <MessageCircle className="size-4 shrink-0 text-muted-foreground" />
                      <span className="flex min-w-0 flex-col leading-snug">
                        <span className="truncate">{title}</span>
                        {preview ? (
                          <span className="truncate text-xs text-muted-foreground/70">{preview}</span>
                        ) : null}
                      </span>
                      <Check
                        className={cn(
                          'ml-auto size-4 shrink-0 text-foreground',
                          session.id !== activeStoredSessionId && 'invisible'
                        )}
                      />
                    </CommandItem>
                  )
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}
