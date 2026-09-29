import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { User } from '@/lib/icons'
import { $brandSession } from '@/store/brand-session'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

export function AccountSettings() {
  const brand = useStore($brandSession)
  const { t } = useI18n()
  const copy = t.settings.account
  const api = typeof window !== 'undefined' ? window.hermesDesktop?.brand : undefined
  const [busy, setBusy] = useState(false)
  const [draft, setDraft] = useState('')

  const brandLabel = brand.activeAppId || (brand.isSuper ? copy.allBrands : copy.signedOut)

  return (
    <SettingsContent>
      <SectionHeading icon={User} title={copy.title} />
      <p className="mb-2 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {brand.signedIn ? copy.intro : copy.signedOut}
      </p>

      {brand.signedIn ? (
        <>
          <ListRow description={brand.email} title={copy.email} />
          <ListRow description={brandLabel} title={copy.brand} />
          <ListRow description={brand.isSuper ? copy.superAccess : copy.brandAccess} title={copy.access} />
          {brand.appIds.length > 1 ? (
            <ListRow
              action={
                <select
                  aria-label={copy.brands}
                  className="bg-transparent text-sm"
                  onChange={event => {
                    void api?.select(event.target.value).then(next => $brandSession.set(next))
                  }}
                  value={brand.activeAppId}
                >
                  {brand.appIds.map(appId => (
                    <option key={appId} value={appId}>
                      {appId}
                    </option>
                  ))}
                </select>
              }
              title={copy.brands}
            />
          ) : null}
          {brand.isSuper ? (
            <ListRow
              action={
                <form
                  className="flex items-center gap-2"
                  onSubmit={event => {
                    event.preventDefault()
                    void api?.select(draft.trim()).then(next => {
                      $brandSession.set(next)
                      setDraft('')
                    })
                  }}
                >
                  <input
                    aria-label={copy.brand}
                    className="w-32 bg-transparent text-sm outline-none"
                    onChange={event => setDraft(event.target.value)}
                    placeholder={copy.brand}
                    value={draft}
                  />
                  {brand.activeAppId ? (
                    <Button
                      onClick={() => {
                        void api?.select('').then(next => $brandSession.set(next))
                      }}
                      type="button"
                      variant="outline"
                    >
                      {copy.allBrands}
                    </Button>
                  ) : null}
                </form>
              }
              title={copy.brand}
            />
          ) : null}
          <SectionHeading icon={User} title={copy.limitsTitle} />
          <p className="mb-4 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {copy.limitsBody}
          </p>
          <Button
            disabled={busy || !api}
            onClick={() => {
              if (!api) {
                return
              }

              setBusy(true)
              void api.signOut().then(next => $brandSession.set(next)).finally(() => setBusy(false))
            }}
            type="button"
            variant="outline"
          >
            {copy.signOut}
          </Button>
        </>
      ) : (
        <Button
          disabled={busy || !api}
          onClick={() => {
            if (!api) {
              return
            }

            setBusy(true)
            void api.signIn().then(next => $brandSession.set(next)).finally(() => setBusy(false))
          }}
          type="button"
        >
          {copy.signIn}
        </Button>
      )}
    </SettingsContent>
  )
}
