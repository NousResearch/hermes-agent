import { useStore } from '@nanostores/react'
import * as React from 'react'

import { useMenuBarCompanion } from '@/hooks/use-menu-bar-companion'
import { broadcastDesktopStateChange } from '@/lib/desktop-state-sync'
import { $menuBarTransparency, setMenuBarTransparency } from '@/store/menu-bar-transparency'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $translucency, setTranslucency } from '@/store/translucency'
import { $zoomPercent, setZoomPercent } from '@/store/zoom'
import { useTheme } from '@/themes'
import type { ThemeMode } from '@/themes/context'

import { useCompanionGateway } from '../hooks/use-companion-gateway'

type PetInfo = {
  enabled?: boolean
  scale?: number
  slug?: string
  displayName?: string
}

type GalleryPet = {
  displayName: string
  installed?: boolean
  slug: string
}

type PetGallery = {
  active?: string
  enabled?: boolean
  pets?: GalleryPet[]
}

/**
 * Appearance tab — shared Desktop and menu-bar appearance, pet, and tray state.
 */
export function AppearanceTab() {
  const activeProfile = normalizeProfileKey(useStore($activeGatewayProfile))
  const { mode, setMode, themeName, availableThemes, setTheme, resolvedMode } = useTheme()
  const translucency = useStore($translucency)
  const menuBarTransparency = useStore($menuBarTransparency)
  const zoomPercent = useStore($zoomPercent)
  const { ready, error, request, retry } = useCompanionGateway()
  const [busy, setBusy] = React.useState(false)
  const [status, setStatus] = React.useState('')
  const [info, setInfo] = React.useState<PetInfo | null>(null)
  const [gallery, setGallery] = React.useState<PetGallery | null>(null)
  const { enabled: companionEnabled, setEnabled: setCompanionEnabled } = useMenuBarCompanion()

  const refreshPet = React.useCallback(async () => {
    if (!ready) {
      return
    }

    try {
      const [petInfo, petGallery] = await Promise.all([
        request<PetInfo>('pet.info', { profile: activeProfile }),
        request<PetGallery>('pet.gallery', { localOnly: true, profile: activeProfile })
      ])

      setInfo(petInfo)
      setGallery(petGallery)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    }
  }, [activeProfile, ready, request])

  React.useEffect(() => {
    void refreshPet()
  }, [refreshPet])

  const installed = React.useMemo(() => {
    const pets = gallery?.pets ?? []

    return pets.filter(pet => pet.installed || pet.slug === gallery?.active)
  }, [gallery])

  const enabled = Boolean(gallery?.enabled ?? info?.enabled)
  const activeSlug = gallery?.active || info?.slug || ''

  const activeName =
    installed.find(pet => pet.slug === activeSlug)?.displayName || info?.displayName || activeSlug || 'none'

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(true)
    setStatus('')

    try {
      await fn()
      await refreshPet()
      broadcastDesktopStateChange('pet', { profile: activeProfile })
      setStatus(label)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const togglePet = () =>
    void run(enabled ? 'Pet off' : 'Pet on', async () => {
      if (enabled) {
        await request('pet.disable', { profile: activeProfile })

        return
      }

      const slug = activeSlug || installed[0]?.slug

      if (!slug) {
        throw new Error('No installed pet to enable')
      }

      await request('pet.select', { profile: activeProfile, slug })
    })

  const cyclePet = (direction: 1 | -1) =>
    void run('Pet cycled', async () => {
      if (!installed.length) {
        throw new Error('No installed pets')
      }

      const currentIndex = Math.max(
        0,
        installed.findIndex(pet => pet.slug === activeSlug)
      )

      const next = installed[(currentIndex + direction + installed.length) % installed.length]
      await request('pet.select', { profile: activeProfile, slug: next.slug })
    })

  const updateMode = (next: ThemeMode) => {
    setMode(next)
    setStatus(`Desktop mode set to ${next}`)
  }

  const updateTheme = (next: string) => {
    setTheme(next)
    setStatus('Color scheme applied to the menu bar and Desktop')
  }

  const updateTranslucency = (next: number) => {
    setTranslucency(next)
    setStatus(`Desktop window transparency set to ${next}%`)
  }

  const updateMenuBarTransparency = (next: number) => {
    setMenuBarTransparency(next)
    setStatus(`Menu bar window transparency set to ${next}%`)
  }

  const updateZoom = (next: number) => {
    setZoomPercent(next)
    setStatus(`Desktop UI scale set to ${next}%`)
  }

  const focusDesktop = () => {
    const focus = window.hermesDesktop?.focusMainWindow

    if (!focus) {
      setStatus('Desktop focus control unavailable')

      return
    }

    void focus()
      .then(() => setStatus('Hermes Desktop opened'))
      .catch(err => setStatus(err instanceof Error ? err.message : String(err)))
  }

  const toggleCompanion = (on: boolean) => {
    void setCompanionEnabled(on)
      .then(enabled => setStatus(enabled ? 'Menu bar companion enabled' : 'Menu bar companion disabled'))
      .catch(err => setStatus(err instanceof Error ? err.message : String(err)))
  }

  return (
    <div className="mbc-tab-panel mbc-stack" data-tab="appearance">
      <section className="mbc-card">
        <h3>Menu bar + Desktop appearance</h3>
        <p className="mbc-muted">
          Active: <strong>{themeName}</strong> / {mode}
          {mode === 'system' ? ` → ${resolvedMode}` : ''}.
        </p>
        <label className="mbc-field">
          <span>Desktop mode</span>
          <select disabled={busy} onChange={event => updateMode(event.target.value as ThemeMode)} value={mode}>
            <option value="system">system</option>
            <option value="light">light</option>
            <option value="dark">dark</option>
          </select>
        </label>
        <label className="mbc-field">
          <span>Menu bar and Desktop color scheme</span>
          <select disabled={busy} onChange={event => updateTheme(event.target.value)} value={themeName}>
            {availableThemes.map(theme => (
              <option key={theme.name} value={theme.name}>
                {theme.label}
              </option>
            ))}
          </select>
        </label>
        <label className="mbc-field">
          <span>Desktop window transparency ({translucency}%)</span>
          <input
            aria-label={`Desktop window transparency (${translucency}%)`}
            max={100}
            min={0}
            onChange={event => updateTranslucency(Number(event.target.value))}
            step={5}
            type="range"
            value={translucency}
          />
        </label>
        <label className="mbc-field">
          <span>Menu bar window transparency ({menuBarTransparency}%)</span>
          <input
            aria-label={`Menu bar window transparency (${menuBarTransparency}%)`}
            max={100}
            min={0}
            onChange={event => updateMenuBarTransparency(Number(event.target.value))}
            step={5}
            type="range"
            value={menuBarTransparency}
          />
        </label>
        <label className="mbc-field">
          <span>Desktop UI scale</span>
          <select
            aria-label="Desktop UI scale"
            disabled={busy}
            onChange={event => updateZoom(Number(event.target.value))}
            value={zoomPercent}
          >
            {[90, 100, 110, 125, 150, 175].includes(zoomPercent) ? null : (
              <option value={zoomPercent}>{zoomPercent}%</option>
            )}
            <option value={90}>90%</option>
            <option value={100}>100%</option>
            <option value={110}>110%</option>
            <option value={125}>125%</option>
            <option value={150}>150%</option>
            <option value={175}>175%</option>
          </select>
        </label>
      </section>

      <section className="mbc-card">
        <div className="mbc-card-head">
          <h3>Hermes Desktop</h3>
          <button className="mbc-button" onClick={focusDesktop} type="button">
            Open Hermes Desktop
          </button>
        </div>
        <p className="mbc-muted">
          Active profile: {activeProfile} · Backend: {ready ? 'connected' : error ? 'unavailable' : 'connecting'}
        </p>
      </section>

      <section className="mbc-card">
        <h3>Pet</h3>
        {error && !ready ? (
          <>
            <p className="mbc-muted">{error}</p>
            <button className="mbc-button" onClick={retry} type="button">
              Retry gateway
            </button>
          </>
        ) : !ready ? (
          <p className="mbc-muted">Connecting pet gateway…</p>
        ) : (
          <>
            <p className="mbc-muted">
              Active pet: <strong>{activeName}</strong> · {enabled ? 'on' : 'off'}
            </p>
            <div className="mbc-row-actions">
              <button
                className="mbc-button"
                disabled={busy || (!enabled && installed.length === 0)}
                onClick={togglePet}
                type="button"
              >
                {enabled ? 'Turn pet off' : 'Turn pet on'}
              </button>
              <button
                className="mbc-button"
                disabled={busy || installed.length < 2}
                onClick={() => cyclePet(1)}
                type="button"
              >
                Next pet
              </button>
              <button
                className="mbc-button"
                disabled={busy || installed.length < 2}
                onClick={() => cyclePet(-1)}
                type="button"
              >
                Prev pet
              </button>
            </div>
            <div aria-label="Installed pets" className="mbc-scroll-list">
              {installed.length === 0 ? (
                <p className="mbc-muted">No installed pets. Adopt one in Desktop Settings → Appearance.</p>
              ) : (
                installed.map(pet => (
                  <button
                    className={pet.slug === activeSlug ? 'mbc-list-button is-active' : 'mbc-list-button'}
                    disabled={busy}
                    key={pet.slug}
                    onClick={() =>
                      void run(`Selected ${pet.displayName || pet.slug}`, async () => {
                        await request('pet.select', { profile: activeProfile, slug: pet.slug })
                      })
                    }
                    type="button"
                  >
                    {pet.displayName || pet.slug}
                    {pet.slug === activeSlug ? ' · active' : ''}
                  </button>
                ))
              )}
            </div>
          </>
        )}
      </section>

      <section className="mbc-card">
        <h3>Companion tray</h3>
        <label className="mbc-check">
          <input checked={companionEnabled} onChange={event => toggleCompanion(event.target.checked)} type="checkbox" />
          <span>Show menu bar companion</span>
        </label>
      </section>

      {status ? <p className="mbc-status">{status}</p> : null}
    </div>
  )
}
