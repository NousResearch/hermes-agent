import { useStore } from '@nanostores/react'
import { type ReactElement, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { $webappOauthPending, completeWebappOauth } from '@/lib/browser-saved-connection-oauth'
import { notifyError } from '@/store/notifications'

import { ListRow } from './primitives'

/**
 * While the Webapp waits on a saved gateway's sign-in: a field for the address the sign-in tab
 * ended on. The other gateway redirects to the serving machine's loopback, which a browser on
 * another machine cannot load, so that browser's address bar holds the code to finish with.
 * Renders nothing unless this tab started such a sign-in (only the Webapp bridge sets it).
 */
export function WebappOauthPasteField(): null | ReactElement {
  const { t } = useI18n()
  const copy = t.settings.gateway.webappOauthPaste
  const pending = useStore($webappOauthPending)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)

  if (!pending) {
    return null
  }

  const finish = async () => {
    setBusy(true)

    try {
      await completeWebappOauth(window.hermesDesktop.api, pending.connectionId, url.trim())
      setUrl('')
    } catch (err) {
      notifyError(err, copy.failed)
    } finally {
      setBusy(false)
    }
  }

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          <Input
            aria-label={copy.title}
            onChange={event => setUrl(event.target.value)}
            placeholder={copy.placeholder}
            value={url}
          />
          <Button disabled={busy || !url.trim()} onClick={() => void finish()} size="sm">
            {copy.finish}
          </Button>
        </div>
      }
      description={copy.description}
      title={copy.title}
    />
  )
}
