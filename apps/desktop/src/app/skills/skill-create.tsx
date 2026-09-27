import { useState } from 'react'

import { type ProfileScope } from '@/api/client'
import { createSkill } from '@/api/skills'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'

import { buildSkillDraft } from './skill-draft'

export function SkillCreate({
  profile,
  scopeLabel,
  onCreated
}: {
  profile: ProfileScope
  scopeLabel?: string
  onCreated: () => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState({ name: '', description: '', category: 'moje', instructions: '' })
  const [preview, setPreview] = useState<ReturnType<typeof buildSkillDraft> | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState('')

  const update = (key: keyof typeof draft, value: string) => {
    setDraft(current => ({ ...current, [key]: value }))
    setPreview(null)
    setError('')
  }

  const save = async () => {
    if (!preview || busy) {
      return
    }

    setBusy(true)
    setError('')

    try {
      await createSkill(preview, profile)
      setSaved(`Dodano ${preview.name}. Skill będzie dostępny w nowej rozmowie.`)
      setOpen(false)
      setDraft({ name: '', description: '', category: 'moje', instructions: '' })
      setPreview(null)
      await onCreated()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Nie udało się zapisać skilla. Spróbuj ponownie.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-border/50 px-4 py-3">
      <Button
        onClick={() => {
          setSaved('')
          setOpen(true)
        }}
        size="sm"
      >
        Wklej lub utwórz skilla
      </Button>
      <a
        className="text-sm text-muted-foreground underline underline-offset-4"
        href="https://skills-pack-ai-evolution.tabascocreatives.chatgpt.site/"
        rel="noreferrer"
        target="_blank"
      >
        Katalog AI Evolution ↗
      </a>
      {saved && (
        <p className="text-sm text-muted-foreground" role="status">
          {saved}
        </p>
      )}
      <Dialog
        onOpenChange={value => {
          if (!busy) {
            setOpen(value)
          }
        }}
        open={open}
      >
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
          <DialogTitle>Dodaj własną umiejętność</DialogTitle>
          <DialogDescription>
            Skopiuj instrukcję z katalogu i wklej poniżej. Możesz też napisać własną. Zapis do profilu:{' '}
            {scopeLabel || 'wybrany profil'}.
          </DialogDescription>
          <div className="grid gap-4 text-sm">
            <label className="grid gap-1">
              Nazwa techniczna
              <Input
                disabled={busy}
                onChange={event => update('name', event.target.value)}
                placeholder="np. raport-tygodniowy"
                value={draft.name}
              />
            </label>
            <label className="grid gap-1">
              Kiedy Czesiek ma jej używać?{' '}
              <span className="text-xs text-muted-foreground">
                Do 60 znaków. Przy pełnym SKILL.md użyj opisu w nagłówku.
              </span>
              <Input
                disabled={busy}
                maxLength={60}
                onChange={event => update('description', event.target.value)}
                placeholder="Gdy proszę o tygodniowe podsumowanie pracy."
                value={draft.description}
              />
            </label>
            <label className="grid gap-1">
              Kategoria
              <Input
                disabled={busy}
                onChange={event => update('category', event.target.value)}
                placeholder="np. biuro, marketing, research"
                value={draft.category}
              />
            </label>
            <label className="grid gap-1">
              Instrukcja lub pełny SKILL.md
              <textarea
                className="min-h-48 rounded-md border border-input bg-background p-3 text-sm focus-visible:ring-2 focus-visible:ring-ring"
                disabled={busy}
                onChange={event => update('instructions', event.target.value)}
                placeholder="Cel, kolejne kroki i oczekiwany wynik…"
                value={draft.instructions}
              />
            </label>
            <p className="text-xs text-muted-foreground">
              Wklej treść pojedynczego skilla, nie adres strony. Skrypty i załączniki nie są importowane przez
              wklejenie. Istniejące skille nie zostaną nadpisane.
            </p>
            {preview && (
              <pre
                aria-label="Podgląd skilla"
                className="max-h-52 overflow-auto whitespace-pre-wrap rounded-lg bg-muted p-3 text-xs"
              >
                {preview.content}
              </pre>
            )}
            {error && (
              <p className="text-destructive" role="alert">
                {error}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <Button disabled={busy} onClick={() => setOpen(false)} variant="ghost">
                Anuluj
              </Button>
              {preview ? (
                <Button disabled={busy} onClick={() => void save()}>
                  {busy ? 'Zapisywanie…' : 'Dodaj skilla'}
                </Button>
              ) : (
                <Button
                  onClick={() => {
                    try {
                      setPreview(buildSkillDraft(draft))
                      setError('')
                    } catch (cause) {
                      setError((cause as Error).message)
                    }
                  }}
                >
                  Sprawdź podgląd
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
