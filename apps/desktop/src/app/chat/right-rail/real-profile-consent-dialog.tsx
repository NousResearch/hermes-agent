import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import { readUseRealProfile } from '@/app/settings/browser-real-profile-panel'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import type { HermesConnection } from '@/global'
import { saveHermesConfigRecord } from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, Globe } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import {
  $realProfilePromptClaim,
  $realProfilePromptDismissed,
  $realProfilePromptMuted,
  claimRealProfilePrompt,
  releaseRealProfilePrompt
} from '@/store/real-profile-consent'
import { $connection } from '@/store/session'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../../hooks/use-config-record'

interface RealProfileConsentDialogProps {
  /** The Browser tab this pane renders — used only to claim the prompt so
   *  several mounted Browser panes never stack duplicate dialogs. */
  tabId: string
}

interface RealProfilePromptGate {
  claim: null | string
  configLoaded: boolean
  connection: HermesConnection | null
  dismissed: boolean
  enabled: boolean
  muted: boolean
  tabId: string
}

/**
 * Whether this pane should offer the real-profile consent prompt.
 *
 * Only for a backend on THIS machine: real-profile mode snapshots the
 * backend host's default browser, and the prompt promises "your" profile and
 * "nothing leaves this computer". Over a remote/SSH/cloud connection that is
 * another machine's browser (often a headless box with none), and "Enable"
 * would write `browser.use_real_profile` into the remote config.yaml — which
 * unregisters its browser tools and re-appears after every reset (#119398).
 * An unresolved connection fails closed. Remote users can still opt in from
 * the Capabilities → Tools → Browser toggle.
 */
export function shouldOfferRealProfilePrompt(gate: RealProfilePromptGate): boolean {
  return (
    gate.connection?.mode === 'local' &&
    gate.configLoaded &&
    !gate.enabled &&
    !gate.dismissed &&
    !gate.muted &&
    gate.claim === gate.tabId
  )
}

/**
 * First-open consent prompt for real-profile browsing. Mounts with a Browser
 * pane (docked tab and `?win=browser` popout alike) and offers to turn on
 * `browser.use_real_profile` — the same key the Capabilities → Tools →
 * Browser toggle writes — when it is off.
 *
 * Accepting saves through the same deep-merging PUT /api/config + shared
 * query cache the toggle reads, so the Capabilities toggle flips on the spot
 * with no refetch. "Not now" mutes the prompt for this app run; "Don't show
 * again" persists the opt-out across launches. Turning the toggle off later
 * does NOT resurrect the prompt inside the same run. Never offered for a
 * remote backend (see shouldOfferRealProfilePrompt).
 */
export function RealProfileConsentDialog({ tabId }: RealProfileConsentDialogProps) {
  const { t } = useI18n()
  const copy = t.settings.toolsets.browserRealProfile
  const prompt = copy.prompt
  const dismissed = useStore($realProfilePromptDismissed)
  const muted = useStore($realProfilePromptMuted)
  const claim = useStore($realProfilePromptClaim)
  const connection = useStore($connection)
  const { data: config, writeScope } = useHermesConfigRecord()
  const setConfig = hermesConfigCacheWriter()
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    claimRealProfilePrompt(tabId)

    return () => releaseRealProfilePrompt(tabId)
  }, [tabId])

  const enabled = readUseRealProfile(config)

  const enable = useCallback(async () => {
    if (!config || busy) {
      return
    }

    const browser =
      config.browser && typeof config.browser === 'object' && !Array.isArray(config.browser)
        ? (config.browser as Record<string, unknown>)
        : {}

    const next = { ...config, browser: { ...browser, use_real_profile: true } }

    setBusy(true)
    setConfig(next)

    try {
      // Sparse patch: PUT /api/config deep-merges, and echoing the cached
      // snapshot would overwrite keys other surfaces changed since it loaded.
      await saveHermesConfigRecord({ browser: { use_real_profile: true } }, writeScope)

      notify({ kind: 'info', title: copy.enabledTitle, message: copy.enabledMessage })
    } catch (err) {
      setConfig(config)
      notifyError(err, copy.failedSave)
    } finally {
      setBusy(false)
    }
  }, [busy, config, copy, setConfig, writeScope])

  // Remote backend, config not loaded yet, feature already on, opted out, or
  // another pane owns the prompt — render nothing. `enabled` flipping true
  // after a successful save is also what closes the dialog.
  const open = shouldOfferRealProfilePrompt({
    claim,
    configLoaded: Boolean(config),
    connection,
    dismissed,
    enabled,
    muted,
    tabId
  })

  if (!open) {
    return null
  }

  const bullets = [prompt.bulletSnapshot, prompt.bulletLiveProfile, prompt.bulletLocal]

  return (
    <Dialog
      onOpenChange={value => {
        // Esc / backdrop are a soft "Not now": mute for this run.
        if (!value && !busy) {
          $realProfilePromptMuted.set(true)
        }
      }}
      open
    >
      <DialogContent className="max-w-md" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={Globe}>{prompt.title}</DialogTitle>
          <DialogDescription>{prompt.body}</DialogDescription>
        </DialogHeader>

        <ul className="grid gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-secondary)">
          {bullets.map(bullet => (
            <li className="flex items-start gap-2" key={bullet}>
              <Check className="mt-0.5 size-3.5 shrink-0 text-primary" />
              <span>{bullet}</span>
            </li>
          ))}
        </ul>

        <DialogFooter className="items-center sm:justify-between">
          <button
            className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) underline-offset-4 transition-colors hover:text-foreground hover:underline"
            disabled={busy}
            onClick={() => $realProfilePromptDismissed.set(true)}
            type="button"
          >
            {prompt.dontShowAgain}
          </button>
          <div className="flex gap-2">
            <Button disabled={busy} onClick={() => $realProfilePromptMuted.set(true)} type="button" variant="ghost">
              {prompt.notNow}
            </Button>
            <Button disabled={busy} onClick={() => void enable()} type="button">
              {prompt.enable}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
