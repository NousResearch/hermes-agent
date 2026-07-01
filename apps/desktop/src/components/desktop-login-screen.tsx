import { useStore } from '@nanostores/react'
import { useMemo, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { markManagedUnavailable, markSignedIn } from '@/store/auth'
import {
  $desktopOnboarding,
  managedBrowserSignIn,
  type OnboardingContext
} from '@/store/onboarding'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// The ApexNodes app icon doubles as the brand mark on the login screen (small
// product chrome, not a splash wordmark — same asset the home screen uses).
const LOGO_ASSET = 'apple-touch-icon.png'

interface DesktopLoginScreenProps {
  /** Message to show above the buttons when the gate tripped mid-session
   *  (401 session-expired / 403 account-disabled). null on a clean first run. */
  gateNotice?: null | string
  onSignedIn?: () => void
  requestGateway: OnboardingContext['requestGateway']
}

// Full-window, centered login gate mirroring Codex "开始使用": our logo, a primary
// "sign in with ApexNodes" button, and a secondary Google quick-login button.
// No marketing copy, no how-to text (Codex layout + our logo + light-purple
// --theme-primary accent only). Both buttons drive the EXISTING managed browser
// (loopback) OAuth flow (managedBrowserSignIn) — this does not add a second auth
// system, it reuses the onboarding store's managed sign-in.
export function DesktopLoginScreen({ gateNotice, onSignedIn, requestGateway }: DesktopLoginScreenProps) {
  const { t } = useI18n()
  const a = t.auth.login
  const { managedError, managedSubmitting } = useStore($desktopOnboarding)

  // A ctx whose onCompleted flips the auth gate to signed-in (and re-reads the
  // account for the panel), then notifies the parent. Stable across renders so
  // the in-flight managed flow keeps a consistent callback.
  const onSignedInRef = useRef(onSignedIn)
  onSignedInRef.current = onSignedIn
  const requestGatewayRef = useRef(requestGateway)
  requestGatewayRef.current = requestGateway

  const ctx = useMemo<OnboardingContext>(
    () => ({
      requestGateway: (...args) => requestGatewayRef.current(...args),
      onCompleted: () => {
        markSignedIn()
        onSignedInRef.current?.()
      }
    }),
    []
  )

  const signIn = (provider: 'apex' | 'google') => {
    if (managedSubmitting) {
      return
    }

    void managedBrowserSignIn(provider, ctx).then(() => {
      // Full success calls ctx.onCompleted (→ markSignedIn) and the gate closes.
      // But if the account is valid yet the relay-key endpoint isn't deployed,
      // the onboarding store degrades to BYOK (managedAvailable → false) WITHOUT
      // completing — step aside so the BYOK picker can take over instead of
      // trapping the user on a login that can't provision a key.
      if ($desktopOnboarding.get().managedAvailable === false) {
        markManagedUnavailable()
      }
    })
  }

  // Prefer the caller's gate notice (session-expired / account-disabled); fall
  // back to an in-flight managed error surfaced by managedBrowserSignIn.
  const notice = gateNotice ?? (managedError || null)

  return (
    <div className="fixed inset-0 z-1400 flex flex-col items-center justify-center bg-(--ui-chat-surface-background) p-6 [-webkit-app-region:drag]">
      <div className="flex w-full max-w-[22rem] flex-col items-center [-webkit-app-region:no-drag]">
        <img alt="" aria-hidden className="mb-6 size-16 rounded-[1.125rem]" src={assetPath(LOGO_ASSET)} />

        <h1 className="mb-8 text-[1.375rem] font-medium tracking-[-0.01em] text-foreground">{a.title}</h1>

        {notice ? (
          <div className="mb-4 w-full rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/40 px-3.5 py-2.5 text-center text-[0.8125rem] leading-5 text-muted-foreground">
            {notice}
          </div>
        ) : null}

        <div className="grid w-full gap-2.5">
          {/* Primary: light-purple --theme-primary accent (design rule). */}
          <Button
            className="w-full bg-(--theme-primary) py-2.5 text-sm text-white hover:bg-[color-mix(in_srgb,var(--theme-primary)_90%,black)]"
            disabled={managedSubmitting}
            onClick={() => signIn('apex')}
            size="lg"
            type="button"
          >
            {managedSubmitting ? <Loader2 className="size-4 animate-spin" /> : null}
            {managedSubmitting ? a.signingIn : a.signInApex}
          </Button>

          {/* Secondary: Google quick-login. */}
          <Button
            className="w-full py-2.5 text-sm"
            disabled={managedSubmitting}
            onClick={() => signIn('google')}
            size="lg"
            type="button"
            variant="outline"
          >
            {a.signInGoogle}
          </Button>
        </div>
      </div>
    </div>
  )
}
