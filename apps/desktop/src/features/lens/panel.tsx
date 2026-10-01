import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerInsertAcked } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { $previewComposerTarget, openPreview } from '@/store/preview'

import { LENS_CARD_LIMIT, type LensCard, lensPrompt } from './model'
import {
  $lensCards,
  $lensScope,
  $unassignedLensCount,
  noteLensCard,
  removeLensCard,
  unassignedLensCards
} from './store'

interface LensPanelProps {
  busy: boolean
  error: string
  onClose: () => void
  onPin: (mode: 'page' | 'selection') => void
  onRefresh: (card: LensCard) => void
  onError: (error: unknown) => void
}

export function LensPanel(props: LensPanelProps) {
  const scope = useStore($lensScope)

  return <ScopedLensPanel key={scope} {...props} />
}

function ScopedLensPanel({ busy, error, onClose, onPin, onRefresh, onError }: LensPanelProps) {
  const { t } = useI18n()
  const copy = t.lens
  const cards = useStore($lensCards)
  const unassignedCount = useStore($unassignedLensCount)
  const [selected, setSelected] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [added, setAdded] = useState(false)
  const chosen = cards.filter(card => selected.includes(card.id))

  const run = (fn: () => void) => {
    try {
      fn()
    } catch (err) {
      onError(err)
    }
  }

  const ask = async () => {
    const ok = await requestComposerInsertAcked(lensPrompt(chosen, question || copy.comparePrompt), {
      target: $previewComposerTarget.get()
    })

    if (ok) {
      setAdded(true)
    } else {
      onError(new Error('noComposer'))
    }
  }

  return (
    <section aria-label={copy.title} className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto p-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">{copy.title}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{copy.subtitle}</p>
        </div>
        <Button aria-label={t.common.close} onClick={onClose} size="icon-sm" variant="ghost">
          <Codicon name="close" />
        </Button>
      </header>
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy} onClick={() => onPin('selection')} size="sm" variant="secondary">
          {copy.pinSelection}
        </Button>
        <Button disabled={busy} onClick={() => onPin('page')} size="sm" variant="ghost">
          {copy.pinPage}
        </Button>
        <span className="text-xs text-muted-foreground">
          {cards.length} / {LENS_CARD_LIMIT}
        </span>
        {busy && (
          <span className="text-xs" role="status">
            {copy.working}
          </span>
        )}
      </div>
      {unassignedCount > 0 && (
        <div className="text-xs text-muted-foreground">
          <p>{copy.earlierCaptures}</p>
          <Button
            onClick={() =>
              run(() => {
                const url = URL.createObjectURL(
                  new Blob([JSON.stringify(unassignedLensCards(), null, 2)], { type: 'application/json' })
                )

                const link = document.createElement('a')
                link.href = url
                link.download = 'hermes-lens-earlier-captures.json'
                link.click()
                URL.revokeObjectURL(url)
              })
            }
            size="xs"
            variant="ghost"
          >
            {copy.exportEarlier}
          </Button>
        </div>
      )}
      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      {cards.length === 0 ? (
        <div className="py-10 text-center">
          <Codicon className="mb-3 text-muted-foreground" name="preview" size="2rem" />
          <h3 className="font-medium text-foreground">{copy.emptyTitle}</h3>
          <p className="mx-auto mt-2 max-w-sm text-sm">{copy.emptyBody}</p>
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,260px),1fr))] items-start gap-5">
          {cards.map(card => (
            <article className="min-w-0 border-t border-(--ui-stroke-tertiary) pt-3" key={card.id}>
              <label className="flex items-start gap-2 font-medium text-foreground">
                <input
                  checked={selected.includes(card.id)}
                  onChange={event => {
                    setAdded(false)
                    setSelected(ids => (event.target.checked ? [...ids, card.id] : ids.filter(id => id !== card.id)))
                  }}
                  type="checkbox"
                />
                <span className="line-clamp-2 break-words">{card.title || new URL(card.url).hostname}</span>
              </label>
              <p className="mt-1 truncate text-xs text-muted-foreground">{new URL(card.url).hostname}</p>
              <p className="mt-3 max-h-44 overflow-auto whitespace-pre-wrap break-words text-sm text-foreground">
                {card.text}
              </p>
              {card.truncated && <p className="mt-1 text-xs">{copy.truncated}</p>}
              {card.previousText && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer font-medium text-(--ui-accent)">{copy.changed}</summary>
                  <p className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-words">{card.previousText}</p>
                </details>
              )}
              <p className="mt-2 text-xs">
                {copy.checked}: {new Date(card.checkedAt).toLocaleString()}
              </p>
              <Textarea
                aria-label={copy.note}
                className="mt-3"
                defaultValue={card.note}
                key={card.id + card.note}
                maxLength={2000}
                onBlur={event => run(() => noteLensCard(card.id, event.target.value))}
                placeholder={copy.note}
                rows={2}
              />
              <div className="mt-2 flex flex-wrap gap-1">
                <Button
                  onClick={() => {
                    openPreview({ kind: 'url', label: card.title, url: card.url, source: card.url })
                    onClose()
                  }}
                  size="xs"
                  variant="ghost"
                >
                  {copy.openSource}
                </Button>
                <Button disabled={busy} onClick={() => onRefresh(card)} size="xs" variant="ghost">
                  {copy.refresh}
                </Button>
                <Button onClick={() => run(() => removeLensCard(card.id))} size="xs" variant="ghost">
                  {copy.remove}
                </Button>
              </div>
            </article>
          ))}
        </div>
      )}
      {cards.length > 0 && (
        <footer className="mt-auto flex flex-col gap-2 border-t border-(--ui-stroke-tertiary) pt-4">
          <Textarea
            aria-label={copy.question}
            maxLength={2000}
            onChange={event => {
              setQuestion(event.target.value)
              setAdded(false)
            }}
            placeholder={copy.question}
            rows={2}
            value={question}
          />
          <div className="flex items-center gap-3">
            <Button disabled={chosen.length === 0 || chosen.length > 8} onClick={() => void ask()} size="sm">
              {copy.ask}
            </Button>
            <span className="text-xs" role="status">
              {added ? copy.added : copy.selectHint}
            </span>
          </div>
        </footer>
      )}
    </section>
  )
}
