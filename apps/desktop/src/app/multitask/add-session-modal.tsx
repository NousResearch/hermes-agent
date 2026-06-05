import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'

import { $sessions } from '@/store/session'
import { $multitaskSessionIds, addMultitaskSession } from '@/store/multitask'

export interface AddSessionModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function AddSessionModal({ open, onOpenChange }: AddSessionModalProps) {
  const sessions = useStore($sessions)
  const multitaskIds = useStore($multitaskSessionIds)
  const [search, setSearch] = useState('')

  const filtered = sessions.filter(s => {
    if (multitaskIds.includes(s.id)) return false
    if (!search.trim()) return true
    const q = search.toLowerCase()
    return (
      (s.title?.toLowerCase().includes(q) ?? false) ||
      s.id.toLowerCase().includes(q) ||
      (s.preview?.toLowerCase().includes(q) ?? false)
    )
  })

  const handleAdd = (sessionId: string) => {
    addMultitaskSession(sessionId)
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-h-[70vh] max-w-md overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add Session to Multitask</DialogTitle>
        </DialogHeader>

        <div className="relative mb-2">
          <Codicon
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-(--ui-text-tertiary)"
            name="search"
            size="0.75rem"
          />
          <input
            className="w-full rounded-md border border-(--ui-stroke-tertiary) bg-transparent py-1.5 pl-7 pr-2.5 text-[0.8125rem] text-foreground outline-none placeholder:text-(--ui-text-tertiary) focus:border-(--ui-stroke-focus)"
            onChange={e => setSearch(e.target.value)}
            placeholder="Search sessions…"
            value={search}
          />
        </div>

        {filtered.length === 0 && (
          <p className="py-4 text-center text-[0.8125rem] text-(--ui-text-tertiary)">
            {sessions.length === 0 ? 'No sessions yet' : 'No matching sessions'}
          </p>
        )}

        <div className="space-y-0.5">
          {filtered.map(session => (
            <button
              key={session.id}
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left hover:bg-(--ui-control-hover-background)"
              onClick={() => handleAdd(session.id)}
              type="button"
            >
              <Codicon className="size-4 shrink-0 text-(--ui-text-tertiary)" name="comment" />
              <span className="min-w-0 flex-1 truncate text-[0.8125rem] text-foreground">
                {session.title || session.id.slice(0, 12) + '…'}
              </span>
              {session.model && (
                <span className="shrink-0 text-[0.6875rem] text-(--ui-text-tertiary)">
                  {session.model}
                </span>
              )}
            </button>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
