/**
 * Theme Forge — turn any image into a full Hermes desktop theme.
 *
 * Drop/paste/browse an image; the palette engine extracts dominant colors
 * (median-cut), maps them to DesktopTheme tokens, guarantees WCAG contrast,
 * builds light + dark variants, and registers via `THEMES_AREA` — themes
 * appear live in Settings → Appearance, ⌘K, and /skin.
 *
 * Pane features: dark/light forge mode, drag-to-reorder swatch priorities
 * (re-synthesizes the theme live), inline rename, terminal ANSI preview,
 * Apply (jumps to Appearance settings), reforge, delete. Sources are kept
 * downscaled so themes can reforge after restarts.
 *
 * Bundled port of the standalone disk plugin
 * (github.com/0-CYBERDYNE-SYSTEMS-0/theme-lab): same `HermesPlugin`
 * contract, same verbatim swatch→theme contract, validated by the same
 * 74-check color-math suite (see forge.test.ts).
 *
 * Ships OFF by default (`defaultEnabled: false`): it inventories in
 * Settings ▸ Plugins and registers nothing until the user flips the switch.
 */

import {
  atom,
  Button,
  cn,
  haptic,
  type HermesPlugin,
  host,
  icons,
  Input,
  PALETTE_AREA,
  type PluginContext,
  type PluginContribution,
  type PluginStorage,
  ScrollArea,
  SegmentedControl,
  THEMES_AREA,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import {
  deriveSwatches,
  extractPalette,
  type ForgeEntry,
  type ForgeMode,
  type ForgeTerminalPalette,
  forgeTheme,
  type ForgeTheme,
  hexToHsl,
  hslToHex,
  loadImageFromUrl,
  parseHexStrict,
  readableOn,
  stripForgePrefix,
  type Swatch,
  synthesize
} from './forge'

// ── reactive state (module-level, survives pane unmount) ────────────────────

const $busy = atom(false)
const $generated = atom<ForgeEntry[]>([]) // persisted themes (full objects) — single source of truth
const $mode = atom<ForgeMode>('dark')
const $expanded = atom<string | null>(null) // slug with terminal preview open
const $editing = atom<string | null>(null) // slug being renamed
const $picked = atom<{ slug: string; index: number } | null>(null) // swatch awaiting a new position
const $viewMode = atom<'cards' | 'strip'>('cards') // strip = quiet swatch-only overview
const $viewModeKey = 'theme-forge-viewMode'
const $wheelOpen = atom<{ slug: string; index: number } | null>(null) // single inline color editor

/** Strip mode is display-only; card mode is the editor. */
function normalizeViewMode(v: string): 'cards' | 'strip' {
  return v === 'strip' ? 'strip' : 'cards'
}

// ── active-skin detection ──────────────────────────────────────────────────
// The app paints the active theme's slug onto <html data-hermes-theme="…">
// (themes/context.tsx applyTheme). Reading it + a MutationObserver gives the
// pane a live "which theme is applied" signal so we can light the active card
// and pin it to the top of the list.
const $forgeActiveSkin = atom<string | null>(null)

function forgeReadActiveSkin(): string | null {
  if (typeof window === 'undefined' || !document.documentElement) {
    return null
  }

  return document.documentElement.dataset.hermesTheme || null
}

let forgeSkinObserver: MutationObserver | null = null

function forgeEnsureSkinObserver(): void {
  if (forgeSkinObserver || typeof window === 'undefined') {
    return
  }

  forgeSkinObserver = new MutationObserver(() => {
    const v = forgeReadActiveSkin()

    if (v !== $forgeActiveSkin.get()) {
      $forgeActiveSkin.set(v)
    }
  })
  forgeSkinObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-hermes-theme'] })
  $forgeActiveSkin.set(forgeReadActiveSkin())
}

function useForgeActiveSkin(): string | null {
  const v = useValue($forgeActiveSkin)
  useEffect(() => {
    forgeEnsureSkinObserver()
  }, [])

  return v
}

/** Small indicator dot: lit when this theme is the one currently applied. */
function ForgeActiveDot({ active }: { active: boolean }) {
  return (
    <span
      aria-hidden
      style={{
        width: 8,
        height: 8,
        borderRadius: 999,
        flexShrink: 0,
        background: active ? 'var(--ui-accent)' : 'var(--ui-stroke-secondary)',
        boxShadow: active ? '0 0 6px var(--ui-accent)' : 'none',
        transition: 'background 0.15s ease, box-shadow 0.15s ease'
      }}
      title={active ? 'Currently applied' : 'Not applied'}
    />
  )
}

// ── persistence + registration ──────────────────────────────────────────────

let storageRef: PluginStorage | null = null
let registerRef: ((c: PluginContribution) => () => void) | null = null
const disposersBySlug = new Map<string, () => void>()

function registerTheme(theme: ForgeTheme): void {
  if (!registerRef) {
    return
  }

  const existing = disposersBySlug.get(theme.name)

  if (existing) {
    existing()
  }

  // Fresh object identity each call → the registry snapshot cache busts and
  // the Appearance grid / active skin repaint live.
  const dispose = registerRef({ id: `theme:${theme.name}`, area: THEMES_AREA, data: { ...theme } })
  disposersBySlug.set(theme.name, dispose)
}

function saveThemes(list: ForgeEntry[]): void {
  if (storageRef) {
    storageRef.set('themes', list)
  }

  $generated.set(list)
}

function updateTheme(slug: string, patch: Partial<ForgeEntry>): ForgeEntry | undefined {
  const list = storageRef ? storageRef.get<ForgeEntry[]>('themes', []) : []
  const next = list.map(t => (t.name === slug ? { ...t, ...patch } : t))
  saveThemes(next)
  const t = next.find(x => x.name === slug)

  if (t && t.theme) {
    registerTheme(t.theme)
  }

  return t
}

// ── forge pipeline glue ─────────────────────────────────────────────────────

function handleFile(file: File | null | undefined): void {
  if (!file || !file.type.startsWith('image/')) {
    host.notify({ kind: 'warning', message: 'That file is not an image.' })

    return
  }

  $busy.set(true)
  forgeTheme(file, $mode.get())
    .then(entry => {
      const list = (storageRef ? storageRef.get<ForgeEntry[]>('themes', []) : []).filter(t => t.name !== entry.name)
      list.unshift(entry)
      saveThemes(list)
      registerTheme(entry.theme)
      haptic('tap')
      host.notify({
        kind: 'success',
        message: `"${entry.label}" forged — Apply in the pane, or pick it in Settings → Appearance.`
      })
    })
    .catch(err => host.notifyError(err, 'Theme Forge'))
    .finally(() => $busy.set(false))
}

function reforge(entry: ForgeEntry): void {
  if (!entry.source) {
    host.notify({ kind: 'warning', message: 'No source image kept for this theme — forge a new one.' })

    return
  }

  loadImageFromUrl(entry.source)
    .then(img => {
      const palette = extractPalette(img, 12)
      const ordered = [...palette].sort((a, b) => b.weight - a.weight).slice(0, 8)
      const theme = synthesize(ordered, entry)
      updateTheme(entry.name, { swatches: ordered, theme, mode: $mode.get() })
      haptic('tap')
      host.notify({ kind: 'success', message: `"${entry.label}" reforged.` })
    })
    .catch(err => host.notifyError(err, 'Theme Forge'))
}

function applyTheme(entry: ForgeEntry): void {
  // Forge themes are contributed to the DESKTOP registry only — the backend
  // can't resolve them, so config.set would silently fall back to `default`.
  // Deep-link those. Backend-known skins (built-ins) apply LIVE below.
  if (forgeIsBackendSkin(entry.name)) {
    forgeApplyLive(entry)

    return
  }

  host.navigate('/settings?tab=config:appearance')
  host.notify({ kind: 'info', message: `Click "${entry.label}" in the grid to apply.` })
}

// Backend-known skins: the gateway's `config.set display.skin=<name>` RPC
// broadcasts skin.changed, which the desktop drains through setTheme → the
// theme repaints WITHOUT navigating away from this pane. Names from
// hermes_cli/skin_engine.py _BUILTIN_SKINS (verified in the app source).
const forgeBackendSkins = new Set([
  'default',
  'ares',
  'mono',
  'slate',
  'daylight',
  'warm-lightmode',
  'poseidon',
  'sisyphus',
  'charizard'
])

function forgeIsBackendSkin(name: string): boolean {
  return Boolean(name) && forgeBackendSkins.has(name)
}

function forgeApplyLive(entry: ForgeEntry): void {
  host
    .request('config.set', { key: 'skin', value: entry.name })
    .then(() => {
      haptic('tap')
      host.notify({ kind: 'success', message: `"${entry.label}" applied live.` })
    })
    .catch(err => {
      host.notifyError(err, 'Theme Forge apply')
      // Fall back to the honest path if the gateway can't take it.
      host.navigate('/settings?tab=config:appearance')
      host.notify({ kind: 'info', message: `Click "${entry.label}" in the grid to apply.` })
    })
}

// ── escape hatch (theme-immune) ─────────────────────────────────────────────
// A broken theme (super-dark bg + dark text) makes the pane's theme-var text
// illegible. This button is the ONE thing that must survive any theme, so it
// deliberately uses HARDCODED colors and a fixed glow — never theme vars. It
// resets to the safe default via the gateway (config.set skin=default →
// skin.changed → desktop setTheme('default') → repaints to the canonical
// 'nous' theme). No navigation, no Settings, fully reversible: the user's
// forged themes stay saved in the pane.
function forgeResetToDefault(): void {
  host
    .request('config.set', { key: 'skin', value: 'default' })
    .then(() => {
      haptic('tap')
      $forgeActiveSkin.set('default')
      host.notify({ kind: 'success', message: 'Reset to the safe default theme.' })
    })
    .catch(err => host.notifyError(err, 'Theme Forge reset'))
}

function ForgeEscapeHatch() {
  return (
    <button
      aria-label="Reset to safe default theme"
      // Hardcoded, theme-independent: high-contrast amber-on-dark that reads
      // against ANY background (light or dark, any palette). boxShadow rings
      // are inline because arbitrary shadow-[…] classes are frozen-CSS dead.
      onClick={forgeResetToDefault}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        width: '100%',
        padding: '6px 10px',
        borderRadius: 6,
        fontSize: 12,
        fontWeight: 700,
        lineHeight: 1.2,
        color: '#111111',
        background: 'linear-gradient(135deg, #ffb300 0%, #ff8f00 100%)',
        border: '1px solid #6d4c00',
        boxShadow: '0 0 0 1px rgba(0,0,0,0.35), 0 0 10px rgba(255,179,0,0.55)',
        cursor: 'pointer',
        flexShrink: 0,
        userSelect: 'none'
      }}
      title="Always visible — reset to the safe default theme if this one is unreadable"
      type="button"
    >
      <icons.AlertTriangle className="size-3.5 shrink-0" style={{ color: '#111111' }} />
      <span>Reset to safe theme</span>
    </button>
  )
}

// ── strip view ──────────────────────────────────────────────────────────────

function StripRow({ active, entry, onOpen }: { active: boolean; entry: ForgeEntry; onOpen?: () => void }) {
  const theme = entry.theme || ({} as ForgeTheme)
  const t = theme.darkTerminal || theme.terminal || ({} as ForgeTerminalPalette)
  const colors = theme.darkColors || theme.colors || ({} as ForgeTheme['colors'])
  const swatches = entry.swatches && entry.swatches.length ? entry.swatches : deriveSwatches(theme)

  const handleClick = (ev: React.MouseEvent) => {
    if ((ev.target as Element).closest('button')) {
      return
    }

    onOpen?.()
  }

  const handleApply = (ev: React.MouseEvent) => {
    ev.stopPropagation()
    applyTheme(entry)
  }

  const label = entry.label || theme.label || entry.name

  const thumb = entry.source ? (
    <img alt="" className="h-5 w-5 shrink-0 rounded-[2px] object-cover" src={entry.source} />
  ) : (
    <div
      className="h-5 w-5 shrink-0 rounded-[2px]"
      style={{
        background: `linear-gradient(135deg, ${colors.background || '#222'} 0%, ${colors.primary || '#666'} 100%)`
      }}
    />
  )

  return (
    <button
      className={cn(
        'flex w-full items-center gap-2 rounded-none px-1.5 py-1 text-left',
        'hover:bg-(--chrome-action-hover) active:bg-(--chrome-active-hover)'
      )}
      onClick={handleClick}
      type="button"
    >
      {thumb}
      <div className="min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-tertiary)" title={label}>
        {label}
      </div>
      <ForgeActiveDot active={active} />
      <div className="flex shrink-0 items-center gap-1">
        <div
          className="relative flex overflow-x-auto overflow-y-visible"
          onWheel={ev => {
            const el = ev.currentTarget

            if (Math.abs(ev.deltaY) > Math.abs(ev.deltaX)) {
              ev.preventDefault()
              el.scrollLeft += ev.deltaY
            }
          }}
          ref={el => {
            if (!el) {
              return
            }

            const inner = el.firstElementChild

            if (!inner) {
              return
            }

            const hint = (el as HTMLElement & { _scrollHint?: HTMLElement })._scrollHint
            const overflow = inner.scrollWidth > el.clientWidth + 1

            if (!overflow && hint) {
              hint.style.opacity = '0'
              hint.removeAttribute('aria-hidden')
            } else if (overflow && !hint) {
              const node = document.createElement('span')
              node.setAttribute('aria-hidden', 'true')
              node.style.cssText =
                'position:absolute;right:0;top:0;bottom:0;width:14px;pointer-events:none;background:linear-gradient(to right, transparent, var(--chrome-action-hover));'
              el.appendChild(node)
              ;(el as HTMLElement & { _scrollHint?: HTMLElement })._scrollHint = node
            }
          }}
          style={{ scrollbarWidth: 'none', scrollSnapType: 'x proximity' }}
        >
          {swatches.slice(0, 8).map((s, i) => (
            <span
              className="flex shrink-0 flex-col items-center"
              key={`s-${i}`}
              style={{ scrollSnapAlign: 'start', gap: 1 }}
            >
              <span
                aria-hidden
                className="h-3.5 w-3.5 rounded-[2px]"
                style={{ background: s.hex }}
                title={i === 0 ? `#1 · bkgnd · ${s.hex}` : i === 1 ? `#2 · text · ${s.hex}` : `#${i + 1} · ${s.hex}`}
              />
              <span
                aria-hidden
                className="text-[0.5rem] leading-none text-(--ui-text-quaternary)"
                style={{ height: 7 }}
              >
                {i === 0 ? 'bkgnd' : i === 1 ? 'text' : ''}
              </span>
            </span>
          ))}
          <Button onClick={handleApply} size="icon-xs" title="Apply theme" variant="ghost">
            <icons.Palette className="size-3.5" />
          </Button>
        </div>
      </div>
    </button>
  )
}

// ── card bits ───────────────────────────────────────────────────────────────

/** Card thumbnail: the kept source image, or a color field built from the
 *  theme's own tokens for v1-era entries (no source persisted). */
function ThemeThumb({ entry }: { entry: ForgeEntry }) {
  if (entry.source) {
    return <img alt="" className="h-5 w-5 shrink-0 rounded-[2px] object-cover" src={entry.source} />
  }

  const c = entry.theme?.darkColors || entry.theme?.colors || ({} as ForgeTheme['colors'])
  const t = entry.theme?.darkTerminal || entry.theme?.terminal || ({} as ForgeTerminalPalette)
  const bg = c.background || '#222222'
  const p1 = c.primary || '#888888'
  const p2 = t.cyan || t.green || p1

  return (
    <div
      className="h-5 w-5 shrink-0 rounded-[3px]"
      style={{
        background: `linear-gradient(135deg, ${bg} 0%, ${bg} 40%, ${p1} 40%, ${p1} 70%, ${p2} 70%)`,
        boxShadow: 'inset 0 0 0 1px rgba(128,128,128,0.35)'
      }}
      title="Theme colors (no source image kept)"
    />
  )
}

function TermPreview({ theme, mode }: { theme: ForgeTheme; mode: ForgeMode }) {
  // Render a mini terminal using the theme's ANSI palette for the right mode
  const t =
    mode === 'light' && !theme.darkTerminal
      ? theme.terminal
      : mode === 'light'
        ? theme.terminal
        : theme.darkTerminal || theme.terminal

  const colors = mode === 'light' ? theme.colors : theme.darkColors || theme.colors
  const bg = colors.background
  const fg = t?.foreground || colors.foreground
  const palette = t || ({} as ForgeTerminalPalette)

  const line = (chunks: [string, string | undefined][], key: string) => (
    <div className="whitespace-pre" key={key}>
      {chunks.map(([text, color], i) => (
        <span key={`${key}-${i}`} style={{ color }}>
          {text}
        </span>
      ))}
    </div>
  )

  return (
    <div
      className="overflow-x-auto rounded-[6px] p-2 font-mono text-[0.6875rem] leading-relaxed"
      style={{ background: bg, color: fg, boxShadow: 'inset 0 0 0 1px rgba(128,128,128,0.25)' }}
    >
      {line(
        [
          ['➜ ', palette.green],
          ['~/farmfriend ', palette.cyan],
          ['git status', fg]
        ],
        'l1'
      )}
      {line(
        [
          ['On branch ', fg],
          ['main', palette.magenta]
        ],
        'l2'
      )}
      {line(
        [
          ['  modified:   ', palette.yellow],
          ['src/agent/core.py', palette.blue]
        ],
        'l3'
      )}
      {line(
        [
          ['  new file:   ', palette.green],
          ['themes/', palette.blue],
          ['forge.json', fg]
        ],
        'l4'
      )}
      {line(
        [
          ['$ ', palette.green],
          ['hermes ', palette.cyan],
          ['--profile ', palette.yellow],
          ['closer', palette.magenta],
          [' chat', fg]
        ],
        'l5'
      )}
      {line(
        [
          ['⚡ error: ', palette.red],
          ['provider timeout — retrying', fg],
          [' (bright: ', palette.brightYellow],
          ['ok', palette.brightGreen],
          [')', fg]
        ],
        'l6'
      )}
    </div>
  )
}

// ── swatch tray (reorder + pick/place + wheel edit) ─────────────────────────

function SwatchTray({ entry }: { entry: ForgeEntry }) {
  const dragIdx = useRef<number | null>(null)
  const [over, setOver] = useState<number | null>(null)
  const picked = useValue($picked)
  const pickedHere = picked && picked.slug === entry.name ? picked.index : null

  // v1-era entries persisted with an empty swatch list — recover from tokens
  const swatches = entry.swatches && entry.swatches.length > 0 ? entry.swatches : deriveSwatches(entry.theme)

  const move = (from: number, to: number): void => {
    if (from === to) {
      return
    }

    $wheelOpen.set(null)
    const sw = [...swatches]
    const [moved] = sw.splice(from, 1)
    sw.splice(to, 0, moved)
    const theme = synthesize(sw, entry)
    updateTheme(entry.name, { swatches: sw, theme })
    haptic('tap')
  }

  // Primary interaction: click a swatch to pick it up, click a slot to place
  // it. Works with any pointer; drag remains available as a fast path.
  const place = (i: number): void => {
    if ($wheelOpen.get()) {
      $wheelOpen.set(null)
    }

    if (pickedHere === null) {
      $picked.set({ slug: entry.name, index: i })

      return
    }

    if (pickedHere === i) {
      $picked.set(null) // toggle off

      return
    }

    move(pickedHere, i)
    $picked.set(null)
  }

  const wheel = useValue($wheelOpen)
  const wheelHere = wheel && wheel.slug === entry.name ? wheel.index : null

  const openWheel = (i: number): void => {
    if (pickedHere !== null) {
      return
    }

    $wheelOpen.set({ slug: entry.name, index: i })
  }

  const commitWheel = (index: number, hex: string): void => {
    const next = swatches.map((s, i) => (i === index ? { ...s, hex, hsl: hexToHsl(hex) } : s)) as Swatch[]
    const theme = synthesize(next, entry)
    updateTheme(entry.name, { swatches: next, theme })
    $wheelOpen.set(null)
    haptic('tap')
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[0.625rem] text-(--ui-text-quaternary)">
        {pickedHere !== null
          ? 'picked up — click a slot to place (click again to cancel)'
          : 'swatch 1 = background hue · swatch 2 = text · tap to pick up, double-click to edit'}
      </div>
      <div
        className="relative flex gap-1.5 overflow-x-auto overflow-y-visible"
        onWheel={ev => {
          const el = ev.currentTarget

          if (Math.abs(ev.deltaY) > Math.abs(ev.deltaX)) {
            ev.preventDefault()
            el.scrollLeft += ev.deltaY
          }
        }}
        ref={el => {
          if (!el) {
            return
          }

          const inner = el.firstElementChild

          if (!inner) {
            return
          }

          const hint = (el as HTMLElement & { _scrollHint?: HTMLElement })._scrollHint
          const overflow = inner.scrollWidth > el.clientWidth + 1

          if (!overflow && hint) {
            hint.style.opacity = '0'
            hint.removeAttribute('aria-hidden')
          } else if (overflow && !hint) {
            const node = document.createElement('span')
            node.setAttribute('aria-hidden', 'true')
            node.style.cssText =
              'position:absolute;right:0;top:0;bottom:0;width:18px;pointer-events:none;background:linear-gradient(to right, transparent, var(--chrome-action-hover));'
            el.appendChild(node)
            ;(el as HTMLElement & { _scrollHint?: HTMLElement })._scrollHint = node
          }
        }}
        style={{ scrollbarWidth: 'none', scrollSnapType: 'x proximity' }}
      >
        {swatches.map((s, i) => (
          <div
            className="flex shrink-0 flex-col items-center gap-0.5"
            key={`swc-${i}`}
            style={{ scrollSnapAlign: 'start' }}
          >
            <div
              className="flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-[5px] text-xs font-bold"
              draggable
              key={`sw-${i}`}
              onClick={() => place(i)}
              onDoubleClick={() => openWheel(i)}
              onDragEnd={() => {
                dragIdx.current = null
                setOver(null)
              }}
              onDragLeave={() => setOver(v => (v === i ? null : v))}
              onDragOver={ev => {
                ev.preventDefault()
                ev.dataTransfer.dropEffect = 'move'
                setOver(i)
              }}
              onDragStart={ev => {
                dragIdx.current = i
                ev.dataTransfer.effectAllowed = 'move'
                ev.dataTransfer.setData('text/plain', String(i))
              }}
              onDrop={ev => {
                ev.preventDefault()
                ev.stopPropagation()
                setOver(null)

                if (dragIdx.current !== null) {
                  move(dragIdx.current, i)
                }

                dragIdx.current = null
              }}
              onKeyDown={ev => {
                if (ev.key === 'Enter' || ev.key === ' ') {
                  ev.preventDefault()
                  place(i)
                }
              }}
              role="button"
              style={{
                background: s.hex,
                color: readableOn(s.hex),
                // ring/scale inline: arbitrary shadow-[…] and scale-* are not
                // in the app's frozen build CSS
                boxShadow:
                  'inset 0 0 0 1px rgba(128,128,128,0.45)' +
                  (pickedHere === i || over === i ? ', 0 0 0 2px var(--ui-accent)' : ''),
                transform: pickedHere === i || over === i ? 'scale(1.08)' : 'none',
                transition: 'transform 0.1s ease'
              }}
              tabIndex={0}
              title={
                i === 0
                  ? `#1 · background seed · ${s.hex}`
                  : i === 1
                    ? `#2 · text seed · ${s.hex}`
                    : `#${i + 1} · ${s.hex}`
              }
            >
              {i + 1}
            </div>
            {/* Role captions: slot 1 seeds the background, slot 2 seeds the
                text color (UI + terminal). Fixed-height slot keeps the row
                aligned where no caption applies. */}
            <div aria-hidden className="h-3 text-center text-[0.5625rem] leading-none text-(--ui-text-quaternary)">
              {i === 0 ? 'bkgnd' : i === 1 ? 'text' : ''}
            </div>
          </div>
        ))}
      </div>
      {wheelHere !== null && wheelHere < swatches.length ? (
        <ColorWheelPanel
          onCancel={() => $wheelOpen.set(null)}
          // Live preview: only update the swatch hex in memory so the wheel's
          // own preview chip follows. Do NOT synthesize/save on every drag —
          // that races with commit and bleaches the final color.
          onChange={hex => {
            const next = swatches.map((s, i) => (i === wheelHere ? { ...s, hex, hsl: hexToHsl(hex) } : s)) as Swatch[]
            updateTheme(entry.name, { swatches: next })
          }}
          onCommit={hex => commitWheel(wheelHere, hex)}
          value={swatches[wheelHere].hex}
        />
      ) : null}
    </div>
  )
}

// ── color wheel (H/S/L editor) ──────────────────────────────────────────────

// Curated fast-pick cells for the picker grid (standard picker behavior).
const PRESET_CELLS = [
  '#ffffff',
  '#f1f3f5',
  '#ced4da',
  '#868e96',
  '#495057',
  '#161616',
  '#000000',
  '#fa5252',
  '#ff922b',
  '#fcc419',
  '#82c91e',
  '#37b24d',
  '#12b886',
  '#20c997',
  '#22b8cf',
  '#339af0',
  '#1971c2',
  '#4c6ef5',
  '#7048e8',
  '#be4bdb',
  '#f06595',
  '#ff8787'
]

function PickerSlider({
  display,
  label,
  max,
  min,
  onChange,
  step,
  track,
  value
}: {
  display: string
  label: string
  max: number
  min: number
  onChange: (v: number) => void
  step: number
  track: string
  value: number
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-4 shrink-0 text-[0.625rem] text-(--ui-text-quaternary)">{label}</span>
      <input
        className="h-1 min-w-0 flex-1"
        max={max}
        min={min}
        onChange={ev => onChange(Number(ev.target.value))}
        step={step}
        style={{ background: track, borderRadius: 999, accentColor: 'var(--ui-accent)' }}
        type="range"
        value={value}
      />
      <span className="shrink-0 text-[0.625rem] text-(--ui-text-tertiary)" style={{ width: 40, textAlign: 'right' }}>
        {display}
      </span>
    </div>
  )
}

function ColorWheelPanel({
  onChange,
  onCancel,
  onCommit,
  value
}: {
  onChange: (hex: string) => void
  onCancel: () => void
  onCommit: (hex: string) => void
  value: string
}) {
  const base = hexToHsl(value) || { h: 0, s: 0.75, l: 0.5 }
  const [h, setH] = useState(base.h)
  const [s, setS] = useState(base.s)
  const [l, setL] = useState(base.l)
  const live = hslToHex(h, s, l)
  const [hexDraft, setHexDraft] = useState(live.toUpperCase())
  const wheelRef = useRef<HTMLDivElement | null>(null)
  const dragging = useRef(false)

  // Keep the hex field in sync with wheel/slider edits while you drag.
  useEffect(() => setHexDraft(live.toUpperCase()), [live])

  const hueDeg = Math.round(h * 360)
  const satPct = Math.round(s * 100)
  const liPct = Math.round(l * 100)

  const pickFromScreen = async (): Promise<void> => {
    if (typeof window === 'undefined' || !('EyeDropper' in window)) {
      host.notify({ kind: 'warning', message: 'Screen pick (eyedropper) is not supported in this build.' })

      return
    }

    try {
      // EyeDropper ships without TS lib types in Electron's Chromium — a
      // minimal local declaration keeps the call typed without widening the
      // global surface.
      const ed = new (
        window as unknown as { EyeDropper: new () => { open: () => Promise<{ sRGBHex: string }> } }
      ).EyeDropper()

      const res = await ed.open()
      const hit = parseHexStrict(res.sRGBHex)

      if (!hit) {
        return
      }

      const hsl = hexToHsl(hit)

      if (hsl) {
        setH(hsl.h)
        setS(hsl.s)
        setL(hsl.l)
      }
    } catch (err) {
      // AbortError = user pressed Escape to cancel; ignore.
      if (!err || (err as Error).name !== 'AbortError') {
        host.notify({ kind: 'warning', message: 'Screen pick failed.' })
      }
    }
  }

  const onHexInput = (ev: React.ChangeEvent<HTMLInputElement>): void => {
    const raw = ev.target.value
    setHexDraft(raw)
    const hit = parseHexStrict(raw)

    if (!hit) {
      return
    }

    const hsl = hexToHsl(hit)

    if (hsl) {
      setH(hsl.h)
      setS(hsl.s)
      setL(hsl.l)
    }
  }

  const onHexBlur = (): void => setHexDraft(live.toUpperCase())

  const pickCell = (hex: string): void => {
    const hsl = hexToHsl(hex)

    if (!hsl) {
      return
    }

    setH(hsl.h)
    setS(hsl.s)
    setL(hsl.l)
  }

  const wheelFromPoint = (clientX: number, clientY: number): { h: number; s: number } | null => {
    const rect = wheelRef.current?.getBoundingClientRect()

    if (!rect) {
      return null
    }

    const x = clientX - rect.left - rect.width / 2
    const y = clientY - rect.top - rect.height / 2
    const radius = Math.min(rect.width, rect.height) / 2
    const dist = Math.sqrt(x * x + y * y) / radius

    if (dist < 0.08 || dist > 1.02) {
      return null
    }

    let angle = Math.atan2(y, x) / (2 * Math.PI)

    if (angle < 0) {
      angle += 1
    }

    return { h: angle, s: Math.min(1, dist) }
  }

  const onWheelPointerDown = (ev: React.PointerEvent<HTMLDivElement>): void => {
    const hit = wheelFromPoint(ev.clientX, ev.clientY)

    if (!hit) {
      return
    }

    dragging.current = true
    setH(hit.h)
    setS(hit.s)
    ev.currentTarget.setPointerCapture?.(ev.pointerId)
    ev.preventDefault()
  }

  const onWheelPointerMove = (ev: React.PointerEvent<HTMLDivElement>): void => {
    if (!dragging.current) {
      return
    }

    const hit = wheelFromPoint(ev.clientX, ev.clientY)

    if (!hit) {
      return
    }

    setH(hit.h)
    setS(hit.s)
    ev.preventDefault()
  }

  const onWheelPointerUp = (): void => {
    dragging.current = false
  }

  return (
    <div
      // flex-wrap: when the pane is too narrow for wheel + controls
      // side-by-side, the controls column wraps below the wheel instead of
      // clipping. min() / vw arbitrary classes are NOT in the app's frozen
      // build CSS, so the wheel size lives inline.
      className="flex min-w-0 flex-wrap items-stretch gap-2 overflow-hidden rounded-[6px] border border-(--ui-stroke-secondary) p-2"
      style={{ background: 'var(--chrome-action-hover)' }}
    >
      <div
        className="relative shrink-0 cursor-crosshair select-none overflow-hidden rounded-full"
        onPointerCancel={onWheelPointerUp}
        onPointerDown={onWheelPointerDown}
        onPointerMove={onWheelPointerMove}
        onPointerUp={onWheelPointerUp}
        ref={wheelRef}
        style={{
          width: 128,
          height: 128,
          background:
            `conic-gradient(from 0deg, hsl(0,100%,50%), hsl(60,100%,50%), hsl(120,100%,50%), hsl(180,100%,50%), hsl(240,100%,50%), hsl(300,100%,50%), hsl(360,100%,50%)),` +
            `radial-gradient(farthest-corner, #fff 0%, rgba(255,255,255,0) 58%, rgba(0,0,0,0.45) 100%)`,
          backgroundBlendMode: 'normal, normal'
        }}
        title="angle = hue · radius = saturation"
      >
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            transform: `translate(calc(-50% + ${Math.cos(h * 2 * Math.PI) * s * 56}px), calc(-50% + ${
              Math.sin(h * 2 * Math.PI) * s * 56
            }px))`,
            background: live,
            border: '1px solid rgba(255,255,255,0.7)',
            boxShadow: '0 0 0 1px rgba(0,0,0,0.45)'
          }}
        />
        <div
          className="pointer-events-none absolute inset-0 rounded-full"
          style={{ background: `radial-gradient(circle, transparent 56%, rgba(0,0,0,0.18) 78%, rgba(0,0,0,0.5) 100%)` }}
        />
      </div>
      <div className="flex min-w-0 flex-col gap-2" style={{ flex: '1 1 100px' }}>
        <div className="flex items-center gap-1.5">
          <div
            className="h-8 w-8 shrink-0 rounded-[4px]"
            style={{ background: live, boxShadow: 'inset 0 0 0 1px rgba(128,128,128,0.45)' }}
          />
          <Input
            className="h-6 min-w-0 flex-1 font-mono text-xs"
            onBlur={onHexBlur}
            onChange={onHexInput}
            onKeyDown={ev => {
              if (ev.key === 'Enter') {
                onCommit(live)
              }

              if (ev.key === 'Escape') {
                onCancel()
              }
            }}
            style={{ width: 88 }}
            value={hexDraft}
          />
          <Button onClick={pickFromScreen} size="icon-xs" title="Pick color from screen (eyedropper)" variant="ghost">
            <icons.Eye className="size-3.5" />
          </Button>
        </div>
        <div className="text-[0.625rem] text-(--ui-text-tertiary)">{`${hueDeg}° hue · ${satPct}% sat · ${liPct}% light`}</div>
        <div className="flex items-center gap-1.5">
          <Button onClick={onCancel} size="xs" variant="secondary">
            Cancel
          </Button>
          <Button onClick={() => onCommit(live)} size="xs" variant="default">
            OK
          </Button>
        </div>
      </div>
      {/* Full H/S/L slider set with gradient tracks (standard picker). */}
      <div className="flex w-full min-w-0 flex-col gap-1">
        <PickerSlider
          display={`${hueDeg}°`}
          label="H"
          max={360}
          min={0}
          onChange={v => setH(v / 360)}
          step={1}
          track={`linear-gradient(to right, ${[0, 60, 120, 180, 240, 300, 360].map(a => `hsl(${a},100%,50%)`).join(', ')})`}
          value={hueDeg}
        />
        <PickerSlider
          display={`${satPct}%`}
          label="S"
          max={100}
          min={0}
          onChange={v => setS(v / 100)}
          step={1}
          track={`linear-gradient(to right, hsl(${hueDeg},0%,${liPct}%), hsl(${hueDeg},100%,${liPct}%))`}
          value={satPct}
        />
        <PickerSlider
          display={`${liPct}%`}
          label="L"
          max={100}
          min={0}
          onChange={v => setL(v / 100)}
          step={1}
          track={`linear-gradient(to right, hsl(${hueDeg},${satPct}%,0%), hsl(${hueDeg},${satPct}%,50%), hsl(${hueDeg},${satPct}%,100%))`}
          value={liPct}
        />
      </div>
      {/* Clickable preset cells. */}
      <div className="flex w-full min-w-0 flex-wrap gap-1">
        {PRESET_CELLS.map(cell => (
          <button
            aria-label={cell}
            className="h-3.5 w-3.5 shrink-0 cursor-pointer rounded-[3px]"
            key={cell}
            onClick={() => pickCell(cell)}
            style={{ background: cell, boxShadow: 'inset 0 0 0 1px rgba(128,128,128,0.45)' }}
            title={cell}
            type="button"
          />
        ))}
      </div>
    </div>
  )
}

// ── theme card ──────────────────────────────────────────────────────────────

function ThemeCard({ active, entry }: { active: boolean; entry: ForgeEntry }) {
  const expanded = useValue($expanded) === entry.name
  const editing = useValue($editing) === entry.name
  const mode = useValue($mode)
  const [draft, setDraft] = useState(entry.label)

  useEffect(() => {
    if (editing) {
      setDraft(stripForgePrefix(entry.label))
    }
  }, [editing, entry.label])

  const commitRename = (): void => {
    const clean = draft.trim()

    if (clean) {
      const label = clean
      const theme = { ...entry.theme, label }
      updateTheme(entry.name, { label, theme })
      host.notify({ kind: 'success', message: `Renamed to "${label}".` })
    }

    $editing.set(null)
  }

  return (
    <div
      className="flex flex-col gap-1.5 rounded-[6px] p-2"
      style={{ boxShadow: 'inset 0 0 0 1px var(--ui-stroke-secondary)' }}
    >
      {/* header row */}
      <div className="flex min-w-0 flex-wrap items-center gap-1">
        <ThemeThumb entry={entry} />
        <ForgeActiveDot active={active} />
        {editing ? (
          <div className="flex min-w-0 flex-1 items-center gap-1">
            <Input
              autoFocus
              className="h-6 min-w-0 flex-1 text-xs"
              onChange={ev => setDraft(ev.target.value)}
              onKeyDown={ev => {
                if (ev.key === 'Enter') {
                  commitRename()
                }

                if (ev.key === 'Escape') {
                  $editing.set(null)
                }
              }}
              value={draft}
            />
            <Button onClick={commitRename} size="icon-xs" variant="ghost">
              <icons.Check />
            </Button>
          </div>
        ) : (
          <div className="min-w-0 flex-1 truncate">
            <button
              className="min-w-0 truncate text-left text-xs font-medium text-(--ui-text-primary) hover:underline"
              onClick={() => $editing.set(entry.name)}
              title="Rename"
              type="button"
            >
              {entry.label}
            </button>
          </div>
        )}
        <div className="flex shrink-0 items-center gap-0.5">
          <Button onClick={() => $editing.set(entry.name)} size="icon-xs" title="Rename theme" variant="ghost">
            <icons.Pencil />
          </Button>
          <Button
            onClick={() => $expanded.set(expanded ? null : entry.name)}
            size="icon-xs"
            title={expanded ? 'Hide terminal preview' : 'Terminal preview'}
            variant="ghost"
          >
            <icons.Terminal />
          </Button>
          <Button onClick={() => reforge(entry)} size="icon-xs" title="Reforge from source image" variant="ghost">
            <icons.RefreshCw />
          </Button>
          <Button
            onClick={() => {
              const list = (storageRef ? storageRef.get<ForgeEntry[]>('themes', []) : []).filter(
                t => t.name !== entry.name
              )

              saveThemes(list)
              const d = disposersBySlug.get(entry.name)

              if (d) {
                d()
                disposersBySlug.delete(entry.name)
              }

              haptic('tap')
              host.notify({ kind: 'info', message: `Removed "${entry.label}".` })
            }}
            size="icon-xs"
            title="Delete theme"
            variant="ghost"
          >
            <icons.Trash2 />
          </Button>
        </div>
      </div>

      <SwatchTray entry={entry} />

      <Button
        onClick={() => {
          host.navigate('/settings?tab=config:appearance')
          host.notify({ kind: 'info', message: `Click "${entry.label}" in the grid to apply.` })
        }}
        size="xs"
        variant="secondary"
      >
        Apply…
      </Button>

      {expanded ? <TermPreview mode={entry.mode || mode} theme={entry.theme} /> : null}
    </div>
  )
}

// ── the pane ────────────────────────────────────────────────────────────────

function ForgePane() {
  const busy = useValue($busy)
  const generated = useValue($generated)
  const mode = useValue($mode)
  const viewMode = useValue($viewMode)
  const activeSkin = useForgeActiveSkin()

  // Pin the currently-applied theme to the top so it's always visible for
  // quick customization; the indicator dot marks it. Rest keeps its order.
  const list = (() => {
    if (!activeSkin) {
      return generated
    }

    const active = generated.find(e => e.name === activeSkin)

    if (!active) {
      return generated
    }

    return [active, ...generated.filter(e => e.name !== activeSkin)]
  })()

  const onDrop = (ev: React.DragEvent): void => {
    ev.preventDefault()
    handleFile(ev.dataTransfer?.files?.[0])
  }

  const setViewMode = (v: string): void => {
    $viewMode.set(normalizeViewMode(v))
    storageRef?.set($viewModeKey, normalizeViewMode(v))
  }

  const openCard = (entry: ForgeEntry): void => {
    $viewMode.set('cards')
    storageRef?.set($viewModeKey, 'cards')
    $expanded.set(entry.name)
  }

  return (
    <div
      className="flex h-full flex-col gap-3 overflow-hidden p-3 text-sm outline-none"
      data-forge-pane="true"
      onDragOver={ev => ev.preventDefault()}
      onDrop={onDrop}
      tabIndex={0}
    >
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 truncate font-medium text-(--ui-text-primary)">Theme Forge</div>
        <div className="flex min-w-0 items-center gap-1">
          <div className="min-w-0">
            <SegmentedControl
              className="max-w-full"
              onChange={v => setViewMode(v)}
              options={[
                { id: 'cards', label: 'Cards' },
                { id: 'strip', label: 'Strip' }
              ]}
              value={viewMode}
            />
          </div>
          <div className="min-w-0">
            <SegmentedControl
              className="max-w-full"
              onChange={v => $mode.set(v)}
              options={[
                { id: 'dark', label: 'Dark' },
                { id: 'light', label: 'Light' }
              ]}
              value={mode}
            />
          </div>
        </div>
      </div>

      <label
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-[6px] border border-dashed p-3 text-center transition-colors',
          'border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover)'
        )}
      >
        <icons.Upload className="size-4 text-(--ui-text-tertiary)" />
        <div className="text-xs text-(--ui-text-secondary)">{busy ? 'Forging…' : 'Drop an image here'}</div>
        <div className="text-[0.6875rem] text-(--ui-text-tertiary)">
          or click to browse · click pane then ⌘V to paste
        </div>
        <input accept="image/*" className="hidden" onChange={ev => handleFile(ev.target.files?.[0])} type="file" />
      </label>

      {/* Pinned escape hatch: always visible, never scrolled away, and immune
          to the theme's own colors (hardcoded) so it works even under a
          broken/unreadable theme. */}
      <ForgeEscapeHatch />

      <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-hidden">
        <div className="min-w-0 text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          {`Forged themes (${generated.length})`}
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <div className={viewMode === 'strip' ? 'flex min-w-0 flex-col gap-px' : 'flex min-w-0 flex-col gap-2 pb-2'}>
            {list.length ? (
              list.map(entry =>
                viewMode === 'strip' ? (
                  <StripRow
                    active={entry.name === activeSkin}
                    entry={entry}
                    key={entry.name}
                    onOpen={() => openCard(entry)}
                  />
                ) : (
                  <ThemeCard active={entry.name === activeSkin} entry={entry} key={entry.name} />
                )
              )
            ) : (
              <div className="py-2 text-xs text-(--ui-text-tertiary)">None yet — forge one above.</div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

// ── plugin entry ────────────────────────────────────────────────────────────

let pasteHandler: ((ev: ClipboardEvent) => void) | null = null

const themeForgePlugin: HermesPlugin = {
  id: 'theme-forge',
  name: 'Theme Forge',
  defaultEnabled: false,

  register(ctx: PluginContext) {
    storageRef = ctx.storage
    registerRef = ctx.register

    // One-time schema migration: v1 stored raw theme objects; v2 stores
    // { name, label, mode, swatches, theme, source } entries. v3 (sleek
    // naming) strips the auto-injected 'Forge · ' prefix from both the
    // card label and the registered theme label, so names show clean —
    // including legacy data persisted before this change.
    const migrated = ctx.storage.get<ForgeEntry[]>('themes', []).map(e => {
      const legacy = e as ForgeEntry & { colors?: unknown }

      const base: ForgeEntry =
        legacy && !legacy.theme && legacy.colors
          ? {
              name: legacy.name,
              label: legacy.label,
              mode: 'dark',
              swatches: [],
              theme: legacy as unknown as ForgeTheme,
              source: null,
              forgedAt: Date.now()
            }
          : legacy

      if (!base || !base.theme) {
        return base
      }

      const label = stripForgePrefix(base.label ?? base.theme.label)

      return { ...base, label, theme: { ...base.theme, label } }
    })

    ctx.storage.set('themes', migrated)

    // Re-register every persisted theme so they survive restarts.
    for (const entry of migrated) {
      if (entry?.theme?.name && entry.theme.colors) {
        registerTheme(entry.theme)
      }
    }

    $generated.set(migrated)

    $viewMode.set(normalizeViewMode(ctx.storage.get($viewModeKey, 'cards')))

    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'theme forge',
      data: { placement: 'right', width: '280px', minWidth: '220px', maxWidth: '520px' },
      render: () => <ForgePane />
    })

    ctx.register({
      id: 'palette-open',
      area: PALETTE_AREA,
      data: {
        id: 'theme-forge-open',
        label: 'Theme Forge: forge a theme from an image',
        keywords: ['theme', 'skin', 'color', 'palette', 'image'],
        run: () =>
          host.notify({ kind: 'info', message: 'Drop or paste an image into the Theme Forge pane (right side).' })
      }
    })

    // Capture-phase paste: forge image pastes aimed at the pane or plain
    // chrome; never steal pastes aimed at the chat composer.
    pasteHandler = ev => {
      const item = Array.from(ev.clipboardData?.items || []).find(i => i.type.startsWith('image/'))

      if (!item) {
        return
      }

      const t = ev.target
      const inPane = t instanceof Element && !!t.closest('[data-forge-pane]')
      const editable = t instanceof Element && !!t.closest('input, textarea, [contenteditable="true"]')

      if (!inPane && editable) {
        return
      }

      const file = item.getAsFile()

      if (file) {
        ev.preventDefault()
        ev.stopPropagation()
        handleFile(file)
      }
    }

    window.addEventListener('paste', pasteHandler, true)

    ctx.onDispose(() => {
      if (pasteHandler) {
        window.removeEventListener('paste', pasteHandler, true)
      }

      pasteHandler = null

      if (forgeSkinObserver) {
        forgeSkinObserver.disconnect()
        forgeSkinObserver = null
      }

      disposersBySlug.forEach(d => d())
      disposersBySlug.clear()
    })
  }
}

export default themeForgePlugin
