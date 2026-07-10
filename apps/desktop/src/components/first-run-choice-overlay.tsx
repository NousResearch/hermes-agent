import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { ModeCard } from '@/app/settings/primitives'
import { RemoteConnectForm } from '@/app/settings/remote-connect-form'
import { useRemoteConnectionForm } from '@/app/settings/use-remote-connection-form'
import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Globe, Loader2, Monitor } from '@/lib/icons'
import { $firstRun, chooseInstall } from '@/store/first-run'

type Choice = 'install' | 'connect'

// First-run gate: shown BEFORE any local install starts on a fresh machine, so
// the user picks where their agent runs. Install proceeds with the existing
// bootstrap (DesktopInstallOverlay takes over once main flips required off);
// Connect drives the shared remote-connection form and applies a remote config
// — main aborts its wait and soft-switches to the remote backend.
export function FirstRunChoiceOverlay() {
  const { required } = useStore($firstRun)

  if (!required) {
    return null
  }

  return <FirstRunChoiceCard />
}

// The overlay backdrop is z-[1450], above the global toast region (z-[200]), so
// errors are surfaced inline in this destructive style instead of via toasts
// that would paint underneath.
function OverlayError({ message }: { message: string }) {
  return (
    <div className="mt-5 rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-xs text-destructive">
      {message}
    </div>
  )
}

function FirstRunChoiceCard() {
  const { t } = useI18n()
  const copy = t.firstRun
  const [choice, setChoice] = useState<Choice>('install')
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)

  const install = async () => {
    setInstalling(true)
    setInstallError(null)

    try {
      await chooseInstall()
      // On success main broadcasts {required:false} and this overlay unmounts;
      // no need to reset `installing`.
    } catch (err) {
      setInstallError(err instanceof Error ? err.message : copy.installFailed)
      setInstalling(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[1450] flex items-center justify-center bg-background/90 backdrop-blur-md p-4">
      <div className="flex w-full max-w-2xl flex-col rounded-xl border border-(--stroke-nous) bg-card shadow-nous">
        <div className="flex flex-shrink-0 items-start gap-4 p-8 pb-4">
          <BrandMark className="size-11 shrink-0" />
          <div className="min-w-0">
            <h2 className="text-xl font-semibold tracking-tight">{copy.title}</h2>
            <p className="mt-1.5 text-sm text-muted-foreground">{copy.subtitle}</p>
          </div>
        </div>

        <div className="px-8 pb-8">
          <div className="grid auto-rows-fr grid-cols-1 gap-2 min-[36rem]:grid-cols-2">
            <ModeCard
              active={choice === 'install'}
              description={copy.installDesc}
              icon={Monitor}
              onSelect={() => setChoice('install')}
              title={copy.installTitle}
            />
            <ModeCard
              active={choice === 'connect'}
              description={copy.connectDesc}
              icon={Globe}
              onSelect={() => setChoice('connect')}
              title={copy.connectTitle}
            />
          </div>

          {choice === 'install' ? (
            <>
              <p className="mt-5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {copy.installHint}
              </p>
              {installError ? <OverlayError message={installError} /> : null}
              <div className="mt-6 flex justify-end">
                <Button disabled={installing} onClick={() => void install()} size="sm">
                  {installing ? <Loader2 className="animate-spin" /> : null}
                  {copy.install}
                </Button>
              </div>
            </>
          ) : (
            <ConnectSection />
          )}
        </div>
      </div>
    </div>
  )
}

// Mounted only when the Connect card is chosen, so install-path users pay zero
// connection-config IPC. The form is locked to remote mode (a fresh machine's
// saved config is 'local', which would otherwise disable the probe/save flow).
function ConnectSection() {
  const { t } = useI18n()
  const form = useRemoteConnectionForm({ scope: null, lockedMode: 'remote' })

  // Unlike Settings (which gates the whole page on a LoadingState), the overlay
  // mounts the form immediately — hold it behind the spinner until the config
  // load resolves so a slow load doesn't wipe a URL the user just typed.
  if (form.loading) {
    return (
      <div className="mt-5 flex items-center justify-center py-8">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <>
      <RemoteConnectForm className="mt-5" form={form} />
      {form.lastError ? <OverlayError message={form.lastError} /> : null}
      <div className="mt-6 flex justify-end">
        <Button disabled={form.saving || !form.canUseRemote} onClick={() => void form.save(true)} size="sm">
          {form.saving ? <Loader2 className="animate-spin" /> : null}
          {t.common.connect}
        </Button>
      </div>
    </>
  )
}
