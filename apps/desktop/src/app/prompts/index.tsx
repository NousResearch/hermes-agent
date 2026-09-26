import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SearchField } from '@/components/ui/search-field'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { requestComposerPrefill } from '@/store/composer'
import { $activeConnectionId } from '@/store/connections'
import { notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'

import { jarvisOnboardingScopeKey } from '../jarvis/onboarding-state'
import { NEW_CHAT_ROUTE } from '../routes'

import { promptCopy } from './copy'
import { promptStore, removePrompt, type SavedPrompt, savePrompt } from './store'

export function PromptLibrary() {
  const connectionId = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const scope = jarvisOnboardingScopeKey({ connectionId, profile })

  return <ScopedPromptLibrary key={scope} scope={scope} />
}

function ScopedPromptLibrary({ scope }: { scope: string }) {
  const { locale } = useI18n()
  const copy = promptCopy(locale)
  const navigate = useNavigate()
  const store = useMemo(() => promptStore(scope), [scope])
  const items = useStore(store)
  const [query, setQuery] = useState('')
  const [draft, setDraft] = useState<Pick<SavedPrompt, 'title' | 'body'> & { id?: string }>({ title: '', body: '' })
  const [error, setError] = useState('')
  const canSave = Boolean(draft.title.trim() && draft.body.trim())

  const matches = items.filter(item =>
    `${item.title}\n${item.body}`.toLocaleLowerCase().includes(query.toLocaleLowerCase())
  )

  const save = () => {
    try {
      const prompt = savePrompt(scope, draft)
      setDraft(prompt)
      setError('')
      notify({ message: copy.saved, kind: 'success' })
    } catch {
      setError(copy.error)
    }
  }

  const remove = () => {
    if (!draft.id) {return}
    const removed = items.find(item => item.id === draft.id)

    if (!removed) {return}

    try {
      removePrompt(scope, removed.id)
      setDraft({ title: '', body: '' })
      notify({
        message: copy.removed,
        action: {
          label: copy.undo,
          onClick: () => {
            try {
              savePrompt(scope, removed)
              setDraft(removed)
            } catch {
              setError(copy.error)
            }
          }
        }
      })
    } catch {
      setError(copy.error)
    }
  }

  return (
    <section aria-label={copy.title} className="flex h-full min-h-0 flex-col gap-5 overflow-auto p-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">{copy.title}</h1>
          <p className="mt-2 text-sm text-(--ui-text-secondary)">{copy.subtitle}</p>
        </div>
        <Button onClick={() => setDraft({ title: '', body: '' })} variant="secondary">
          {copy.add}
        </Button>
      </header>
      <div className="grid min-h-0 flex-1 gap-6 md:grid-cols-[minmax(180px,260px)_1fr]">
        <aside className="flex min-h-0 flex-col gap-3">
          <SearchField aria-label={copy.search} onChange={setQuery} placeholder={copy.search} value={query} />
          <div aria-label={copy.title} className="flex flex-col gap-1 overflow-y-auto">
            {matches.map(item => (
              <Button
                key={item.id}
                onClick={() => {
                  setDraft(item)
                  setError('')
                }}
                variant={draft.id === item.id ? 'secondary' : 'ghost'}
              >
                <span className="truncate">{item.title}</span>
              </Button>
            ))}
            {!items.length ? <p className="text-sm text-(--ui-text-tertiary)">{copy.empty}</p> : null}
          </div>
        </aside>
        <form
          className="flex min-h-0 flex-col gap-4"
          onSubmit={event => {
            event.preventDefault()
            save()
          }}
        >
          <label className="grid gap-2 text-sm">
            {copy.name}
            <Input
              maxLength={120}
              onChange={event => setDraft({ ...draft, title: event.target.value })}
              value={draft.title}
            />
          </label>
          <label className="flex min-h-48 flex-1 flex-col gap-2 text-sm">
            {copy.content}
            <Textarea
              className="min-h-48 flex-1"
              maxLength={30000}
              onChange={event => setDraft({ ...draft, body: event.target.value })}
              value={draft.body}
            />
          </label>
          {error ? (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button disabled={!canSave} type="submit">
              {copy.save}
            </Button>
            <Button
              disabled={!draft.body.trim()}
              onClick={() => {
                requestComposerPrefill(draft.body)
                navigate(NEW_CHAT_ROUTE)
              }}
              type="button"
              variant="secondary"
            >
              {copy.use}
            </Button>
            <Button
              disabled={!draft.body.trim()}
              onClick={() =>
                void navigator.clipboard
                  .writeText(draft.body)
                  .then(() => notify({ message: copy.copied }))
                  .catch(error => notifyError(error, copy.error))
              }
              type="button"
              variant="secondary"
            >
              {copy.copy}
            </Button>
            <Button disabled={!draft.id} onClick={remove} type="button" variant="ghost">
              {copy.remove}
            </Button>
          </div>
        </form>
      </div>
    </section>
  )
}
