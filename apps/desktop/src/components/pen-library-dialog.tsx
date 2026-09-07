import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { useEffect, useState } from 'react'

import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { useI18n } from '@/i18n'
import { Check, MessageCircle, Pencil, Trash2 } from '@/lib/icons'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { openPenCanvas } from '@/store/pen'

interface PenLibraryDialogProps {
  onOpenChange: (open: boolean) => void
  open: boolean
}

export function PenLibraryDialog({ onOpenChange, open }: PenLibraryDialogProps) {
  const { t } = useI18n()
  const [search, setSearch] = useState('')
  const [confirmingDelete, setConfirmingDelete] = useState<null | string>(null)
  const queryClient = useQueryClient()

  const libraryQuery = useQuery({
    enabled: open,
    queryFn: () => window.hermesDesktop?.pen?.library() ?? Promise.resolve({ items: [], root: '' }),
    queryKey: ['pen-library']
  })

  useEffect(() => {
    if (!open) {
      setSearch('')
      setConfirmingDelete(null)
    }
  }, [open])

  const items = libraryQuery.data?.items ?? []

  const refresh = () => void queryClient.invalidateQueries({ queryKey: ['pen-library'] })

  return (
    <DialogPrimitive.Root onOpenChange={onOpenChange} open={open}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-(--z-over-modal) bg-black/15 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[14vh] z-(--z-over-modal-content) w-[min(40rem,calc(100vw-2rem))] -translate-x-1/2 overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-bubble-background) shadow-lg duration-150 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:slide-in-from-top-2 data-[state=open]:zoom-in-95"
        >
          <DialogPrimitive.Title className="sr-only">{t.penLibrary.title}</DialogPrimitive.Title>
          <Command className="bg-transparent" loop>
            <CommandInput onValueChange={setSearch} placeholder={t.penLibrary.searchPlaceholder} value={search} />
            <CommandList className="max-h-[min(24rem,60vh)]">
              <CommandEmpty>{t.penLibrary.empty}</CommandEmpty>
              <CommandGroup
                className="**:[[cmdk-group-heading]]:uppercase **:[[cmdk-group-heading]]:tracking-wider **:[[cmdk-group-heading]]:text-[0.6875rem] **:[[cmdk-group-heading]]:text-muted-foreground/70"
                heading={t.penLibrary.title}
              >
                {items.map(item => {
                  const confirming = confirmingDelete === item.path

                  return (
                    <CommandItem
                      className="group/pen gap-2.5"
                      key={item.path}
                      onSelect={() => {
                        if (confirming) {
                          return
                        }

                        void openPenCanvas({ path: item.path })
                        onOpenChange(false)
                      }}
                      value={`${item.name} ${item.path}`}
                    >
                      {item.previewPath ? (
                        <img
                          alt=""
                          className="h-7 w-9 shrink-0 rounded-[4px] border border-(--ui-stroke-secondary) object-cover"
                          src={`hermes-media://stream/${encodeURIComponent(item.previewPath)}`}
                        />
                      ) : (
                        <Pencil className="size-4 shrink-0 text-muted-foreground" />
                      )}
                      <span className="min-w-0 flex-1 truncate">{item.name}</span>
                      {item.sessionId && (
                        <MessageCircle className="size-3.5 shrink-0 text-muted-foreground/70" />
                      )}
                      {item.open && <Check className="size-3.5 shrink-0 text-muted-foreground" />}
                      <span className="shrink-0 text-[0.6875rem] text-muted-foreground/70">
                        {relativeTime(item.modifiedAt)}
                      </span>
                      {confirming ? (
                        <span className="flex shrink-0 items-center gap-1.5">
                          <button
                            className="cursor-pointer rounded px-1.5 py-0.5 text-[0.6875rem] text-(--ui-text-danger) hover:bg-(--chrome-action-hover)"
                            onClick={event => {
                              event.stopPropagation()
                              void window.hermesDesktop?.pen?.libraryDelete(item.path).then(refresh)
                              setConfirmingDelete(null)
                            }}
                            type="button"
                          >
                            {t.penLibrary.confirmDelete}
                          </button>
                          <button
                            className="cursor-pointer rounded px-1.5 py-0.5 text-[0.6875rem] text-muted-foreground hover:bg-(--chrome-action-hover)"
                            onClick={event => {
                              event.stopPropagation()
                              setConfirmingDelete(null)
                            }}
                            type="button"
                          >
                            {t.penLibrary.cancelDelete}
                          </button>
                        </span>
                      ) : (
                        <button
                          aria-label={t.penLibrary.delete}
                          className={cn(
                            'shrink-0 cursor-pointer rounded p-1 text-muted-foreground opacity-0 transition-opacity',
                            'hover:bg-(--chrome-action-hover) hover:text-(--ui-text-danger)',
                            'group-hover/pen:opacity-100 group-data-[selected=true]/pen:opacity-100'
                          )}
                          onClick={event => {
                            event.stopPropagation()
                            setConfirmingDelete(item.path)
                          }}
                          title={t.penLibrary.delete}
                          type="button"
                        >
                          <Trash2 className="size-3.5" />
                        </button>
                      )}
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
