import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import {
  type BrowserMode,
  type BrowserModeStatus,
  clearBrowserCopy,
  clearBrowserOwnProfile,
  getBrowserStatus,
  importBrowserLogins,
  openBrowserSignIn,
  setBrowserMode
} from '@/api/browser'
import { Button } from '@/components/ui/button'
import type { ProfileScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

const COPY = {
  en: {
    active: 'In use',
    clear: 'Remove copied logins',
    clearOwn: 'Sign out everywhere',
    copyBody:
      'Czesiek starts already signed in, using a copy of your default browser’s logins. Your real browser is never driven and never changed. The copy stays on this computer.',
    copyImport: 'Copy logins now',
    copyTitle: 'Use my logins from Chrome',
    live:
      'Driving your open browser as-is is not offered: Chrome 136+ blocks remote control of the default profile, so a copy is the safe way to reuse your sign-ins.',
    managed: 'Standard (no logins)',
    managedBody: 'A clean background browser. Nothing signed in.',
    ownBody: 'A separate, visible browser window for Czesiek. Sign in to Gmail there once — it stays signed in.',
    ownOpen: 'Open window & sign in to Google',
    ownTitle: 'Czesiek’s own profile',
    signedIn: 'Google: signed in',
    signedOut: 'Google: not signed in yet',
    title: 'Browser for the agent',
    use: 'Use this',
    failed: 'Browser action failed',
    done: 'Done'
  },
  pl: {
    active: 'Aktywne',
    clear: 'Usuń skopiowane logowania',
    clearOwn: 'Wyloguj wszędzie',
    copyBody:
      'Czesiek startuje już zalogowany, korzystając z kopii logowań Twojej domyślnej przeglądarki. Twoja prawdziwa przeglądarka nie jest sterowana ani zmieniana. Kopia zostaje na tym komputerze.',
    copyImport: 'Skopiuj logowania teraz',
    copyTitle: 'Użyj moich logowań z Chrome',
    live:
      'Sterowanie Twoją otwartą przeglądarką „na żywo” nie jest dostępne: Chrome 136+ blokuje zdalne sterowanie domyślnym profilem, więc kopia to bezpieczny sposób na użycie Twoich logowań.',
    managed: 'Standardowa (bez logowań)',
    managedBody: 'Czysta przeglądarka w tle. Nic nie jest zalogowane.',
    ownBody:
      'Osobne, widoczne okno przeglądarki dla Cześka. Zaloguj się tam raz do Gmaila — logowanie zostaje.',
    ownOpen: 'Otwórz okno i zaloguj się do Google',
    ownTitle: 'Własny profil Cześka',
    signedIn: 'Google: zalogowano',
    signedOut: 'Google: jeszcze niezalogowano',
    title: 'Przeglądarka agenta',
    use: 'Użyj tego',
    failed: 'Akcja przeglądarki nie powiodła się',
    done: 'Gotowe'
  }
} as const

type Copy = (typeof COPY)[keyof typeof COPY]

function signInLabel(copy: Copy, signedIn: boolean | null | undefined) {
  return signedIn == null ? null : signedIn ? copy.signedIn : copy.signedOut
}

interface OptionProps {
  active: boolean
  body: string
  children?: React.ReactNode
  copy: Copy
  onUse?: () => void
  title: string
  badge?: null | string
  busy: boolean
}

function Option({ active, badge, body, busy, children, copy, onUse, title }: OptionProps) {
  return (
    <div className="jarvis-well grid gap-2 p-3" data-active={active}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-(--ui-text-primary)">{title}</span>
        {active ? (
          <span className="text-xs text-(--ui-accent)">{copy.active}</span>
        ) : onUse ? (
          <Button disabled={busy} onClick={onUse} size="sm" type="button" variant="secondary">
            {copy.use}
          </Button>
        ) : null}
      </div>
      <p className="text-xs text-(--ui-text-tertiary)">{body}</p>
      {badge ? <p className="text-xs text-(--ui-text-secondary)">{badge}</p> : null}
      {children ? <div className="flex flex-wrap gap-2">{children}</div> : null}
    </div>
  )
}

/**
 * The three ways the agent can have a browser it is signed in to: its own profile (sign in once),
 * a copy of the user's default-browser logins, or the plain background browser. Statuses come from
 * cookie names only — the backend never reads or returns cookie values.
 */
export function BrowserModesPanel({ profile }: { profile?: ProfileScope }) {
  const { locale } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const client = useQueryClient()
  const key = ['browser-modes', profile ?? null]
  const { data } = useQuery({ queryFn: () => getBrowserStatus(profile), queryKey: key, refetchInterval: 4000 })
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<BrowserModeStatus>) => {
    setBusy(true)

    try {
      client.setQueryData(key, await action())
      notify({ kind: 'info', message: copy.done, title: copy.title })
    } catch (error) {
      notifyError(error, copy.failed)
    } finally {
      setBusy(false)
    }
  }

  const use = (mode: BrowserMode) => () => void run(() => setBrowserMode(mode, profile))
  const mode = data?.mode

  return (
    <div className="grid gap-2">
      <h3 className="text-sm font-semibold text-(--ui-text-primary)">{copy.title}</h3>
      <Option
        active={mode === 'own'}
        badge={signInLabel(copy, data?.own.google_signed_in)}
        body={copy.ownBody}
        busy={busy}
        copy={copy}
        onUse={use('own')}
        title={copy.ownTitle}
      >
        <Button disabled={busy} onClick={() => void run(() => openBrowserSignIn(profile))} size="sm" type="button">
          {copy.ownOpen}
        </Button>
        {data?.own.has_profile ? (
          <Button
            disabled={busy || data.own.window_open}
            onClick={() => void run(() => clearBrowserOwnProfile(profile))}
            size="sm"
            type="button"
            variant="ghost"
          >
            {copy.clearOwn}
          </Button>
        ) : null}
      </Option>
      {data?.copy.available ? (
        <Option
          active={mode === 'copy'}
          badge={signInLabel(copy, data.copy.google_signed_in)}
          body={copy.copyBody}
          busy={busy}
          copy={copy}
          onUse={use('copy')}
          title={copy.copyTitle}
        >
          <Button disabled={busy} onClick={() => void run(() => importBrowserLogins(profile))} size="sm" type="button">
            {copy.copyImport}
          </Button>
          {data.copy.has_copy ? (
            <Button disabled={busy} onClick={() => void run(() => clearBrowserCopy(profile))} size="sm" type="button" variant="ghost">
              {copy.clear}
            </Button>
          ) : null}
        </Option>
      ) : null}
      <Option active={mode === 'managed'} body={copy.managedBody} busy={busy} copy={copy} onUse={use('managed')} title={copy.managed} />
      <p className="text-xs text-(--ui-text-tertiary)">{copy.live}</p>
    </div>
  )
}
