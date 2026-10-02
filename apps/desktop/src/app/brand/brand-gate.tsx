import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { Loader } from '@/components/ui/loader'
import { $brandConnectors, emptyBrandConnectors } from '@/store/brand-connectors'
import { $brandSession, type BrandSession } from '@/store/brand-session'

import { releaseBrandMcp, syncBrandMcp } from './sync-brand-mcp'

function BrandCover({ children }: { children: ReactNode }) {
  const cover = (
    <div
      className="grid place-items-center px-6 text-(--ui-text-primary)"
      data-brand-gate=""
      data-glass-opaque=""
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1300,
        background: 'var(--ui-bg-chrome, #161616)',
        color: 'var(--ui-text-primary, #f5f5f5)'
      }}
    >
      {children}
    </div>
  )

  if (typeof document === 'undefined') {
    return cover
  }

  return createPortal(cover, document.body)
}

function applyBrand(next: BrandSession): void {
  $brandSession.set(next)
}

export function BrandGate() {
  const brand = useStore($brandSession)
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const wasSignedIn = useRef(false)
  const api = typeof window !== 'undefined' ? window.hermesDesktop?.brand : undefined

  useEffect(() => {
    if (!api) {
      setReady(true)
      return
    }

    let cancelled = false

    void api
      .get()
      .then(next => {
        if (!cancelled) {
          applyBrand(next)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError('Could not read the portal sign-in.')
        }
      })
      .finally(() => {
        if (!cancelled) {
          setReady(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [api])

  useEffect(() => {
    if (!ready || !api?.connectors) {
      return
    }

    if (!brand.signedIn || !brand.activeAppId) {
      $brandConnectors.set(emptyBrandConnectors())

      if (wasSignedIn.current) {
        wasSignedIn.current = false
        void releaseBrandMcp().catch(() => undefined)
      }

      return
    }

    wasSignedIn.current = true

    let cancelled = false

    void api
      .connectors()
      .then(next => {
        if (cancelled) {
          return
        }

        $brandConnectors.set(next)
        void syncBrandMcp(next.connectors).catch(() => undefined)
      })
      .catch(() => {
        if (!cancelled) {
          $brandConnectors.set({
            ...emptyBrandConnectors(brand.activeAppId),
            error: 'Could not load this brand’s tools.'
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [api, brand.activeAppId, brand.signedIn, ready])

  if (!api) {
    return null
  }

  if (!ready) {
    return (
      <BrandCover>
        <Loader aria-label="Checking sign-in" className="size-24" type="lemniscate-bloom" />
      </BrandCover>
    )
  }

  if (!brand.signedIn) {
    return (
      <BrandCover>
        <div className="flex w-full max-w-md flex-col items-center gap-4 rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-bg-chrome) p-6 text-center">
          <Loader
            aria-hidden="true"
            className="size-16 text-primary"
            role="presentation"
            type="lemniscate-bloom"
          />
          <div className="space-y-2">
            <h1 className="text-xl font-semibold">Welcome to IVX Desktop</h1>
            <p className="text-sm text-(--ui-text-secondary)">
              Sign in to open your chats and tools. We’ll bring you right back here.
            </p>
          </div>
          {error ? <p className="text-sm text-red-400">{error}</p> : null}
          <button
            className="rounded-md bg-(--ui-text-primary) px-4 py-2 text-sm text-(--ui-bg-chrome) disabled:opacity-60"
            disabled={busy}
            onClick={() => {
              setBusy(true)
              setError('')
              void api
                .signIn()
                .then(next => {
                  applyBrand(next)
                  if (!next.signedIn) {
                    setError('Sign-in was not finished.')
                  }
                })
                .catch(() => setError('Sign-in window could not open.'))
                .finally(() => setBusy(false))
            }}
            type="button"
          >
            {busy ? 'Waiting for sign-in…' : 'Sign in'}
          </button>
        </div>
      </BrandCover>
    )
  }

  return null
}
