import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { addDriveFolder, type DriveMemoryStatus, getDriveMemory, removeDriveFolder, syncDriveMemory } from '@/api/drive-memory'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { X } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'

const COPY = {
  en: {
    add: 'Add folder',
    body: 'Pick Drive folders. Czesiek copies their documents into your Obsidian vault (with a link back to each file), so “find the contract with client X” searches your own files.',
    empty: 'No folders yet.',
    failed: 'Drive indexing failed',
    indexed: (n: number) => `${n} documents in the vault`,
    placeholder: 'Paste a Drive folder link',
    remove: 'Remove',
    sync: 'Index now',
    syncing: (left: number) => (left > 0 ? `Indexing… ${left} left` : 'Indexing…'),
    title: 'Drive → memory',
    done: (n: number) => `Indexed ${n} documents`
  },
  pl: {
    add: 'Dodaj folder',
    body: 'Wskaż foldery na Dysku. Czesiek skopiuje ich dokumenty do Twojego vaultu Obsidiana (z linkiem do pliku), więc „znajdź umowę z klientem X” przeszukuje Twoje własne pliki.',
    empty: 'Nie ma jeszcze folderów.',
    failed: 'Indeksowanie Dysku nie powiodło się',
    indexed: (n: number) => `${n} dokumentów w vaulcie`,
    placeholder: 'Wklej link do folderu z Dysku',
    remove: 'Usuń',
    sync: 'Indeksuj teraz',
    syncing: (left: number) => (left > 0 ? `Indeksuję… zostało ${left}` : 'Indeksuję…'),
    title: 'Dysk → pamięć',
    done: (n: number) => `Zaindeksowano dokumentów: ${n}`
  }
} as const

/** Folders whose documents Czesiek mirrors into the vault; sync runs in short steps until nothing is left. */
export function DriveMemoryPanel() {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const client = useQueryClient()
  const { data } = useQuery({ queryFn: () => getDriveMemory(), queryKey: ['drive-memory'] })
  const [link, setLink] = useState('')
  const [status, setStatus] = useState<null | string>(null)
  const [busy, setBusy] = useState(false)

  const apply = (next: DriveMemoryStatus) => client.setQueryData(['drive-memory'], next)

  const guard = async (fn: () => Promise<void>) => {
    setBusy(true)

    try {
      await fn()
    } catch (error) {
      notifyError(error, copy.failed)
    } finally {
      setBusy(false)
      setStatus(null)
    }
  }

  const sync = () =>
    guard(async () => {
      let total = 0

      for (let left = 1; left > 0; ) {
        setStatus(copy.syncing(left === 1 ? 0 : left))
        const step = await syncDriveMemory()

        apply(step)
        total += step.processed
        left = step.processed === 0 ? 0 : step.remaining
      }

      notify({ kind: 'info', message: copy.done(total), title: copy.title })
    })

  return (
    <div className="grid gap-2 border-t border-(--ui-border) pt-3">
      <h4 className="text-sm font-semibold text-(--ui-text-primary)">{copy.title}</h4>
      <p className="text-xs text-(--ui-text-tertiary)">{copy.body}</p>
      <form
        className="flex items-center gap-2"
        onSubmit={event => {
          event.preventDefault()
          void guard(async () => {
            apply(await addDriveFolder(link))
            setLink('')
          })
        }}
      >
        <input
          aria-label={copy.placeholder}
          className="jarvis-well min-w-0 flex-1 px-3 py-2 text-sm text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary)"
          disabled={busy}
          onChange={event => setLink(event.target.value)}
          placeholder={copy.placeholder}
          value={link}
        />
        <Button disabled={busy || !link.trim()} size="sm" type="submit" variant="secondary">
          {copy.add}
        </Button>
      </form>
      {data?.folders.length ? (
        <ul className="flex flex-wrap gap-1.5">
          {data.folders.map(folder => (
            <li className="jarvis-well flex items-center gap-1 py-1 pl-3 pr-1 text-sm" key={folder.id}>
              {folder.name}
              <Button
                aria-label={`${copy.remove} ${folder.name}`}
                disabled={busy}
                onClick={() => void guard(async () => void apply(await removeDriveFolder(folder.id)))}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <X />
              </Button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-(--ui-text-tertiary)">{copy.empty}</p>
      )}
      <div className="flex items-center gap-3">
        <Button disabled={busy || !data?.folders.length} onClick={() => void sync()} size="sm" type="button">
          {copy.sync}
        </Button>
        <span className="text-xs text-(--ui-text-tertiary)">{status ?? copy.indexed(data?.indexed ?? 0)}</span>
      </div>
    </div>
  )
}
