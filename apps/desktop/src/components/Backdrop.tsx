import { useStore } from '@nanostores/react'
import { createContext, type CSSProperties, type ReactNode, useContext } from 'react'

import type { DesktopWallpaperAsset } from '@/global'
import { wallpaperBackgroundProperties, wallpaperMaskImage, type WallpaperPreferences } from '@/lib/wallpaper'
import { $backdrop } from '@/store/backdrop'
import { $wallpaper, $wallpaperActive } from '@/store/wallpaper'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`
type BackdropScope = 'hidden' | 'surface' | 'workspace'

const BackdropScopeContext = createContext<BackdropScope>('surface')

export function wallpaperImageStyle(asset: DesktopWallpaperAsset, preferences: WallpaperPreferences): CSSProperties {
  const background = wallpaperBackgroundProperties(preferences.mode)
  const bleed = Math.max(2, preferences.blur * 2)

  return {
    backgroundImage: `url(${JSON.stringify(asset.url)})`,
    backgroundPosition: background.position,
    backgroundRepeat: background.repeat,
    backgroundSize: background.size,
    filter: preferences.blur > 0 ? `blur(${preferences.blur}px)` : undefined,
    inset: `-${bleed}px`,
    opacity: preferences.opacity / 100,
    transform: preferences.blur > 0 ? 'translateZ(0)' : undefined
  }
}

export function wallpaperOverlayStyle(preferences: WallpaperPreferences): CSSProperties {
  const mask = wallpaperMaskImage(
    preferences.overlayShape,
    preferences.overlayX,
    preferences.overlayWidth,
    preferences.overlayHeight
  )

  return {
    backgroundColor: preferences.overlayColor || 'var(--ui-chat-surface-background)',
    maskImage: mask,
    opacity: preferences.overlay / 100,
    WebkitMaskImage: mask
  }
}

function CustomWallpaperBackdrop({
  asset,
  preferences
}: {
  asset: DesktopWallpaperAsset
  preferences: WallpaperPreferences
}) {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden"
      data-wallpaper-mode={preferences.mode}
      style={{ zIndex: -1 }}
    >
      <div className="absolute" style={wallpaperImageStyle(asset, preferences)} />
      <div
        className="absolute inset-0"
        data-wallpaper-mask-shape={preferences.overlayShape}
        style={wallpaperOverlayStyle(preferences)}
      />
    </div>
  )
}

function BuiltInBackdrop() {
  const on = useStore($backdrop)

  if (!on) {
    return null
  }

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 z-2 opacity-[0.025] mix-blend-difference">
      <img
        alt=""
        className="h-[160dvh] w-auto min-w-dvw object-cover object-left-top [filter:invert(var(--backdrop-invert-mul,1))]"
        fetchPriority="low"
        src={assetPath('ds-assets/filler-bg0.jpg')}
      />
    </div>
  )
}

function SurfaceBackdrop() {
  const wallpaper = useStore($wallpaper)

  if (wallpaper.asset && wallpaper.preferences.enabled) {
    return <CustomWallpaperBackdrop asset={wallpaper.asset} preferences={wallpaper.preferences} />
  }

  return <BuiltInBackdrop />
}

function WorkspaceSurfaceFallback() {
  const wallpaperActive = useStore($wallpaperActive)

  return wallpaperActive ? null : <BuiltInBackdrop />
}

export function Backdrop() {
  const scope = useContext(BackdropScopeContext)

  if (scope === 'hidden') {
    return null
  }

  return scope === 'workspace' ? <WorkspaceSurfaceFallback /> : <SurfaceBackdrop />
}

export function WorkspaceWallpaperScope({ children }: { children: ReactNode }) {
  return <BackdropScopeContext value="workspace">{children}</BackdropScopeContext>
}

export function HiddenBackdropScope({ children }: { children: ReactNode }) {
  return <BackdropScopeContext value="hidden">{children}</BackdropScopeContext>
}

export function WorkspaceWallpaperBackdrop() {
  const wallpaper = useStore($wallpaper)

  if (!wallpaper.asset || !wallpaper.preferences.enabled) {
    return null
  }

  return <CustomWallpaperBackdrop asset={wallpaper.asset} preferences={wallpaper.preferences} />
}
