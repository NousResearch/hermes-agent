import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'

interface StudySetupProps {
  initialPath?: string
  onSave: (vaultPath: string) => Promise<{ requires_new_session?: boolean }>
}

export function StudySetup({ initialPath = '', onSave }: StudySetupProps) {
  const { t } = useI18n()
  const [path, setPath] = useState(initialPath)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)

  const choose = async () => {
    const [selected] = await selectDesktopPaths({ defaultPath: path || undefined, directories: true, multiple: false })

    if (selected) {
      setPath(selected)
    }
  }

  const save = async () => {
    setSaving(true)
    setError(null)
    setSaved(null)

    try {
      const result = await onSave(path)
      setSaved(result.requires_new_session ? t.study.newSessionRequired : t.study.settingsSaved)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl rounded-3xl border bg-card/70 p-6">
      <h2 className="text-2xl font-semibold">{t.study.setupTitle}</h2>
      <p className="mt-2 text-sm text-muted-foreground">{t.study.setupDescription}</p>
      <label className="mt-6 block text-sm font-medium" htmlFor="study-vault-path">
        {t.study.vaultPath}
      </label>
      <div className="mt-2 flex gap-2">
        <Input id="study-vault-path" onChange={event => setPath(event.target.value)} value={path} />
        <Button onClick={() => void choose()} variant="secondary">
          {t.study.chooseFolder}
        </Button>
      </div>
      {error && <div className="mt-3 text-sm text-destructive">{error}</div>}
      {saved && <div className="mt-3 text-sm text-primary">{saved}</div>}
      <Button className="mt-5" disabled={saving || !path.trim()} onClick={() => void save()}>
        {t.study.saveSettings}
      </Button>
    </div>
  )
}
