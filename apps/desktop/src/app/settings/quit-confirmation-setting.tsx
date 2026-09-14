import { useEffect, useId, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Loader } from '@/components/ui/loader'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'

import type { QuitConfirmationMode } from '../../../electron/quit-guard'

import { ListRow } from './primitives'

export function QuitConfirmationSetting() {
  const { t } = useI18n()
  const copy = t.settings.quitConfirmation
  const descriptionId = useId()
  const settings = window.hermesDesktop?.settings
  const [mode, setMode] = useState<QuitConfirmationMode | null>(null)
  const [pending, setPending] = useState(true)
  const [error, setError] = useState<'loadFailed' | 'saveFailed' | null>(null)
  const [loadAttempt, setLoadAttempt] = useState(0)

  useEffect(() => {
    if (!settings?.getQuitConfirmation || !settings.setQuitConfirmation) {
      return
    }

    let cancelled = false

    void settings
      .getQuitConfirmation()
      .then(value => {
        if (!cancelled) {
          setMode(value)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError('loadFailed')
        }
      })
      .finally(() => {
        if (!cancelled) {
          setPending(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [loadAttempt, settings])

  useEffect(() => {
    const refresh = () => {
      // A different app window may have changed the device preference. Keep
      // reads and writes serialized so a late read cannot undo a saved choice.
      if (!pending) {
        setPending(true)
        setError(null)
        setLoadAttempt(attempt => attempt + 1)
      }
    }

    window.addEventListener('focus', refresh)

    return () => window.removeEventListener('focus', refresh)
  }, [pending])

  if (!settings?.getQuitConfirmation || !settings.setQuitConfirmation) {
    return null
  }

  const options: { value: QuitConfirmationMode; label: string }[] = [
    { value: 'never', label: copy.never },
    { value: 'while-working', label: copy.whileWorking },
    { value: 'always', label: copy.always }
  ]

  const save = async (value: string) => {
    const next = options.find(option => option.value === value)?.value

    if (pending || mode === null || !next || next === mode || !settings.setQuitConfirmation) {
      return
    }

    setPending(true)
    setError(null)

    try {
      // Only the native process can confirm that the choice was persisted.
      setMode(await settings.setQuitConfirmation(next))
    } catch {
      setError('saveFailed')
    } finally {
      setPending(false)
    }
  }

  const retry = () => {
    setPending(true)
    setError(null)
    setLoadAttempt(attempt => attempt + 1)
  }

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          {pending && <Loader className="size-4 shrink-0" label={t.common.loading} />}
          <Select disabled={pending || mode === null} onValueChange={value => void save(value)} value={mode ?? ''}>
            <SelectTrigger aria-describedby={descriptionId} aria-label={copy.title}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      }
      below={
        error && (
          <div className="mt-2 space-y-1">
            <p className="text-[length:var(--conversation-caption-font-size)] text-destructive" role="alert">
              {copy[error]}
            </p>
            {error === 'loadFailed' && (
              <Button onClick={retry} size="inline" variant="textStrong">
                {t.common.retry}
              </Button>
            )}
          </div>
        )
      }
      description={<span id={descriptionId}>{copy.description}</span>}
      title={copy.title}
    />
  )
}
