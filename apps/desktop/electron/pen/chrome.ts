// Host chrome inside the editor page: hiding pen's own agent UI, theme
// blending, and the injected boot assets (style, tagger, agent cursor).

import { documents, log, runtime } from './state'
import { runPenGuestScript } from './webview'

/** Whether the embedded canvas hides pen's OWN agent — the floating chat
 *  panel, its composer, and the toolbar button that reopens it. Hermes is the
 *  agent for this canvas, so a second one is a duplicate that can also drive
 *  the document behind hermes's back; hidden by default, but a toggle, since
 *  pen's agent is a real feature.
 *
 *  This is the value baked into each canvas as it loads; flipping an
 *  already-open canvas is the host's job (it owns the WebContents). */
export let penAgentHidden = true

export function setPenAgentHidden(hidden: boolean): void {
  penAgentHidden = hidden
}

export function isPenAgentHidden(): boolean {
  return penAgentHidden
}

/** The one-liner that flips an already-loaded canvas, so the host doesn't
 *  hand-roll the attribute name. */
export function penAgentScript(hidden: boolean): string {
  return `document.documentElement.dataset.hermesPenAgent = ${JSON.stringify(hidden ? 'hidden' : 'shown')}`
}

/** The theme kind last delivered to live editors, so repaint only nudges them
 *  when the host's kind actually flipped — 'toggle-theme' is a blind flip
 *  (it's pen's own menu verb), so an idempotent repaint must gate it. */
export let lastPenThemeKind: null | string = null

/** Cross-module write door — ESM imports are read-only live bindings, so the
 *  protocol handler (which seeds the gate at boot) goes through this. */
export function setLastPenThemeKind(kind: null | string): void {
  lastPenThemeKind = kind
}

/** Re-theme every live editor to the CURRENT host chrome. Pen takes its theme
 *  once at boot (initParams.theme ← getActiveThemeKind), so a hermes theme
 *  flip must push: notify 'toggle-theme' — the same signal Pen.app's own menu
 *  sends — which makes the editor flip dark/light, and re-run the chrome
 *  script so the page background follows. Call AFTER setPenHostChrome. */
export async function repaintPenTheme(): Promise<void> {
  const doc = [...documents.values()][0]

  if (doc) {
    // What SHOULD the editors show now? (host background luminance)
    const kind = doc.device.getActiveThemeKind()

    if (kind !== lastPenThemeKind) {
      lastPenThemeKind = kind

      for (const live of documents.values()) {
        try {
          live.ipc?.notify('toggle-theme', {})
        } catch {
          // Cosmetic.
        }
      }
    }
  }

  await runPenGuestScript(penHostChromeScript())
}

/** Host chrome the canvas should match: hermes's window background. Set by
 *  main (which owns the theme) so the canvas can blend instead of theming
 *  itself. NO scale plumbing: the editor renders 1:1 in its pane — a
 *  fullscreen canvas in a tile, nothing more. */
export const penHostChrome = { background: '#1e1e1e' }

/** The host window's background — also the theme oracle for the editor
 *  (getActiveThemeKind derives dark/light from its luminance). */
export function penHostBackground(): string {
  return penHostChrome.background
}

export function setPenHostChrome(next: { background?: string }): void {
  if (next.background) {
    penHostChrome.background = next.background
  }
}

/** Apply host chrome to an already-open canvas, so a theme flip or a zoom
 *  change doesn't wait for a reopen. */
export function penHostChromeScript(): string {
  return `(() => {
    document.documentElement.dataset.hermesPenBg = ''
    document.documentElement.style.setProperty('--hermes-host-bg', ${JSON.stringify(penHostChrome.background)})
    // Keep the editor's own persisted theme in agreement (localStorage wins
    // over boot params in pen's resolution order).
    try {
      const kind = ${JSON.stringify(lastPenThemeKind || 'light')}
      localStorage.setItem('theme', kind)
      document.documentElement.classList.remove('dark', 'light')
      document.documentElement.classList.add(kind)
    } catch {}
  })()`
}

/**
 * Host chrome suppression for the embedded canvas.
 *
 * The drawer is a CANVAS, not a second app: hermes already owns the window,
 * the titlebar, the traffic lights, and the chat. So pen's own app chrome is
 * hidden — its left panel (Agent / Layers / Slides / Components / Libraries),
 * the layer-list, new-file, and settings buttons, its Agents menu, and the
 * titlebar drag strip (which would otherwise fight hermes's drag region and
 * move the window when you meant to draw).
 *
 * What STAYS is anything that acts on the design: Share, present, and
 * open-in-browser, plus the whole toolbar and the canvas itself.
 *
 * Targeted by pen's own stable aria-labels rather than its hashed utility
 * classes, and injected as a stylesheet — pen's markup and bundle are never
 * modified, so an upstream update can't be broken by this, only ignored.
 */
export const PEN_HOST_CHROME_STYLE = `<style id="hermes-pen-host-chrome">
      /* Pen's own app chrome — hermes already owns the window and the chat. */
      [aria-label="toggle-layer-list"],
      [aria-label="toggle-new-file"],
      [aria-label="open-settings"],
      [aria-label="Agent sessions"],
      [aria-label="Agent panel controls"] {
        display: none !important;
      }

      /* Pen's LEFT PANEL — the Agent/Layers/Slides/Components/Libraries rail
         and everything it hosts. This is a ~320px column, which on a drawer
         is most of the canvas; hiding only its buttons left the column itself
         sitting there. Tagged at runtime by its tab rail (below). */
      html[data-hermes-pen-agent='hidden'] [data-hermes-pen-side-panel] {
        display: none !important;
      }

      /* Pen's agent, in full: the floating chat panel, its header bar, its
         composer, and the toolbar button that reopens it. Tagged at runtime
         (see the tagger below) because these carry hashed utility classes,
         not stable hooks.

         Hermes IS the agent here — a second chat inside the canvas is a
         duplicate that can also drive the document behind hermes's back. Kept
         as a toggle rather than a deletion: pen's agent is a real feature. */
      html[data-hermes-pen-agent='hidden'] [data-hermes-pen-agent-chat],
      html[data-hermes-pen-agent='hidden'] [data-hermes-pen-agent-launcher] {
        display: none !important;
      }

      /* The canvas lives INSIDE a hermes pane now — pen's own titlebar drag
         strip must not fight the embedding window's drag regions. */
      .drag,
      [style*="app-region: drag"] {
        -webkit-app-region: no-drag !important;
      }

      /* Pen's example-prompt chips ("Control panel for a humanoid robotics
         factory floor", …). They're pen's agent onboarding; hermes is the
         agent here, so they're noise that eats the top of a narrow canvas. */
      html[data-hermes-pen-agent='hidden'] [data-hermes-pen-examples] {
        display: none !important;
      }

      /* The bottom bar: rotating preset prompts + the zoom pill. Chromeless
         means chromeless — pan/zoom stay on trackpad and keyboard. */
      html[data-hermes-pen-agent='hidden'] [data-hermes-pen-bottom-bar] {
        display: none !important;
      }

      /* BACKGROUND — blend with hermes.
         pen's own html/body are transparent, so the canvas shows whatever the
         native view paints behind it. Painting the host's window colour here
         means the drawer reads as part of the app instead of a pasted-in
         rectangle with its own theme. */
      html[data-hermes-pen-bg],
      html[data-hermes-pen-bg] body {
        background: var(--hermes-host-bg) !important;
      }

    </style>`

/**
 * Runtime tagger for pen's agent surface.
 *
 * Pen's chat panel is styled with hashed utility classes, so there's nothing
 * stable to write a CSS selector against. What IS stable is its accessible
 * labels ("Minimize chat", "Open agent tab", "New agent"), so we find those
 * and tag their panel — CSS above does the hiding. Runs on load and on any
 * DOM change, because the panel mounts late and remounts per document.
 *
 * Never edits pen's markup: it only adds data-attributes on the host side,
 * so an upstream update can ignore this rather than break on it.
 */
export const PEN_HOST_CHROME_TAGGER = `<script id="hermes-pen-host-tagger">
      (() => {
        // Set from the host on every load. Done HERE rather than by patching
        // the <html> tag in the served markup: that replace silently no-ops if
        // the tag doesn't match byte-for-byte, and it did — the flag never
        // arrived and nothing was hidden.
        document.documentElement.dataset.hermesPenAgent = '__HERMES_PEN_AGENT__'

        // Expose the editor's scene manager. In pen's own code this flag has
        // exactly ONE effect (verified in the shipped bundle): the scene
        // manager assigns itself to window.__SCENE_MANAGER at construction.
        // That handle carries the REAL selection + camera APIs the agent
        // cursor needs (selectionManager.getWorldspaceBounds, camera.toScreen,
        // camera.ensureVisible) — the same calls pen's paste and zoom-keys
        // use internally.
        window.IS_DEV = true

        // THEME, at the editor's real source of truth. Pen resolves theme as
        // localStorage("theme") ?? "dark" — localStorage BEATS the host's
        // initParams.theme (verified in the shipped bundle), and the default
        // is dark. So a stale/absent localStorage value paints pen's whole
        // workspace near-black inside a light hermes no matter what boot
        // params say. This script runs before the editor's module scripts:
        // writing the host's kind here makes every later read agree.
        try {
          localStorage.setItem('theme', '__HERMES_PEN_THEME__')
          document.documentElement.classList.remove('dark', 'light')
          document.documentElement.classList.add('__HERMES_PEN_THEME__')
        } catch {}

        // Blend with hermes: paint the host window's own background colour
        // instead of pen's, so the drawer doesn't read as a pasted-in panel.
        document.documentElement.dataset.hermesPenBg = ''
        document.documentElement.style.setProperty('--hermes-host-bg', '__HERMES_PEN_BG__')

        const CHAT_MARKS = ['Minimize chat', 'Open agent tab', 'New agent', 'Send message', 'Agent panel controls', 'Agent sessions', 'New conversation']

        const tagChat = () => {
          // The drawing surface — the one thing hiding must never touch.
          const canvas = document.querySelector('canvas')

          for (const mark of CHAT_MARKS) {
            for (const el of document.querySelectorAll('[aria-label="' + mark + '"]')) {
              // The agent panel has a real boundary in pen's own markup: a
              // <section> wrapping the whole panel (measured live: one
              // section holds Agent panel controls / Agent sessions / Send
              // message). Prefer that boundary — climbing "as far as
              // possible" swallowed the wrapper that also hosts the floating
              // TOOL RAIL, which is how manual design controls vanished.
              const section = el.closest('section')

              if (section && !(canvas && section.contains(canvas))) {
                section.dataset.hermesPenAgentChat = ''
                continue
              }

              // Fallback (no section boundary): bounded walk, tight area cap
              // so it can never swallow toolbar-bearing wrappers.
              let node = el.parentElement
              let panel = el

              for (let hops = 0; node && node !== document.body && hops < 10; hops += 1) {
                if (canvas && node.contains(canvas)) break

                const rect = node.getBoundingClientRect()

                if (rect.width * rect.height > innerWidth * innerHeight * 0.35) break

                if (rect.width > 80 || rect.height > 40) panel = node

                node = node.parentElement
              }

              if (panel) panel.dataset.hermesPenAgentChat = ''
            }
          }

          // The BOTTOM BAR — preset prompt chips + the zoom pill. Located
          // structurally: the zoom readout is the only "NN%" text on the
          // page, so climb from it to the outermost bottom-anchored strip.
          // No text-matching of preset copy (it rotates per launch).
          for (const el of document.querySelectorAll('button, span, div')) {
            if (el.children.length > 0) continue
            if (!/^\\d{1,3}%$/.test((el.textContent || '').trim())) continue

            let node = el.parentElement
            let bar = null

            for (let hops = 0; node && node !== document.body && hops < 10; hops += 1) {
              if (canvas && node.contains(canvas)) break

              const rect = node.getBoundingClientRect()

              // Bottom-anchored, shallow, not the whole page: the bar.
              if (rect.height < 140 && rect.bottom > innerHeight - 16) bar = node

              node = node.parentElement
            }

            if (bar) bar.dataset.hermesPenBottomBar = ''
          }

          // The preset-prompt strip that FLOATS over the canvas bottom (its
          // live fingerprint: absolute + bottom-anchored + z-40 + select-text,
          // measured in the running canvas). Structural, not text: any
          // absolutely-positioned bottom-anchored layer that only contains
          // buttons/text, sits above the canvas, and is not the zoom pill.
          for (const el of document.querySelectorAll('div')) {
            if ('hermesPenBottomBar' in el.dataset || 'hermesPenExamples' in el.dataset) continue

            const style = getComputedStyle(el)

            if (style.position !== 'absolute') continue
            if (canvas && el.contains(canvas)) continue

            const rect = el.getBoundingClientRect()

            if (rect.height < 8 || rect.height > 160) continue
            // anchored to the bottom edge of the viewport (within 96px)
            if (innerHeight - rect.bottom > 96) continue
            // pure text/button content — a strip of prompt pills
            if (el.querySelector('canvas, input, [contenteditable="true"]')) continue
            if (!el.querySelector('button') && !/\\S/.test(el.textContent || '')) continue
            // never the zoom pill itself (NN%)
            if (/^\\s*[-+]?\\s*\\d{1,3}%/.test((el.textContent || '').trim())) continue

            el.dataset.hermesPenExamples = ''
          }

          // The toolbar button that reopens the agent. Matched on its own text
          // because it carries no aria-label.
          for (const button of document.querySelectorAll('button')) {
            if ((button.textContent || '').trim() === 'Agents') button.dataset.hermesPenAgentLauncher = ''
          }

          // Pen's left panel. Identified by its tab rail — the row holding
          // Layers/Slides/Components, which is stable naming even though the
          // classes are hashed. From the rail we climb to the column that
          // OWNS it (tall, and a real fraction of the viewport), so the whole
          // panel goes rather than just the tabs.
          const TABS = ['Layers', 'Slides', 'Components', 'Libraries']

          for (const button of document.querySelectorAll('button')) {
            if ((button.textContent || '').trim() !== 'Layers') continue

            const rail = button.parentElement
            if (!rail) continue

            const labels = [...rail.children].map(c => (c.textContent || '').trim())
            if (TABS.filter(tab => labels.includes(tab)).length < 3) continue

            let node = rail
            let panel = null

            for (let hops = 0; node && node !== document.body && hops < 8; hops += 1) {
              const rect = node.getBoundingClientRect()

              // A tall column that is NOT the whole editor: that's the panel.
              if (rect.height > innerHeight * 0.5 && rect.width > 80 && rect.width < innerWidth * 0.9) {
                panel = node
              }

              node = node.parentElement
            }

            if (panel) panel.dataset.hermesPenSidePanel = ''
          }
        }

        // Pen's example-prompt chips. They live in TWO places (the agent
        // panel's column and the canvas's bottom bar), so tag every chip
        // directly and every container holding two or more — a single-return
        // walk here left the bottom-bar set visible.
        const tagExamples = () => {
          const chips = [...document.querySelectorAll('button')].filter(b =>
            /humanoid robotics|Lisbon coworking|matcha|Retro-futuristic|meditation app|trading|magazine-style|reservation app/i.test(b.textContent || '')
          )

          for (const chip of chips) chip.dataset.hermesPenExamples = ''

          for (const chip of chips) {
            let node = chip.parentElement
            for (let hops = 0; node && node !== document.body && hops < 6; hops += 1) {
              const inside = chips.filter(c => node.contains(c)).length
              if (inside >= 2 && inside === node.querySelectorAll('button').length) {
                node.dataset.hermesPenExamples = ''
                break
              }
              node = node.parentElement
            }
          }
        }

        // Each step isolated: one selector churning under pen's re-renders
        // must not starve the others.
        const boot = () => {
          try { tagChat() } catch {}
          try { tagExamples() } catch {}
        }

        boot()
        document.addEventListener('DOMContentLoaded', boot)
        new MutationObserver(boot).observe(document.documentElement, { childList: true, subtree: true })
      })()
    </script>`

/**
 * AGENT PRESENCE — the "Hermes is here" cursor.
 *
 * pen drives the canvas through MCP operations, not synthetic input, so there
 * is no real pointer to mirror. This draws one: a labelled cursor that parks
 * over whatever hermes is touching, plus a status chip, so the user can see
 * WHERE the agent is working instead of watching nodes appear from nowhere.
 *
 * Positioning is driven from the editor's own selection bounds
 * (selectionBoundsWorld → screen), so it tracks the real thing rather than a
 * guess, and it rides pen's viewport transform: pan/zoom and it stays put.
 *
 * Injected into the canvas page; pen's own markup is never modified.
 */
export const PEN_AGENT_CURSOR = `<script id="hermes-pen-agent-cursor">
      (() => {
        const NS = 'hermesPenCursor'
        if (window[NS]) return

        const ACCENT = '#7c5cff'
        let el = null
        let hideTimer = 0

        const build = () => {
          if (el || !document.body) return el

          el = document.createElement('div')
          el.setAttribute('data-hermes-pen-cursor', '')
          el.style.cssText = [
            'position:fixed',
            'left:0','top:0',
            'z-index:2147483600',
            'pointer-events:none',
            'opacity:0',
            'display:flex',
            'align-items:flex-start',
            'gap:4px',
            'transform:translate3d(-100px,-100px,0)',
            'transition:transform 320ms cubic-bezier(.22,1,.36,1), opacity 160ms ease-out',
            'will-change:transform'
          ].join(';')

          // Arrow + label, the shape every multiplayer cursor uses.
          el.innerHTML =
            '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" style="filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))">' +
            '<path d="M3 2.5 L14 8.2 L9.1 9.6 L7.3 14.3 Z" fill="' + ACCENT + '" stroke="#fff" stroke-width="1.1" stroke-linejoin="round"/>' +
            '</svg>' +
            '<span data-label style="' +
            'background:' + ACCENT + ';color:#fff;font:500 11px/1.45 ui-sans-serif,system-ui,sans-serif;' +
            'padding:2px 7px;border-radius:999px;white-space:nowrap;margin-top:10px;' +
            'box-shadow:0 1px 3px rgba(0,0,0,.28)">Hermes</span>'

          document.body.append(el)
          return el
        }

        // The editor's live viewport transform, so canvas coords land on the
        // right pixels at any pan/zoom. __SCENE_MANAGER is exposed by pen
        // itself under IS_DEV (set in the boot tagger); camera.toScreen is
        // the same call its own overlays use.
        const toScreen = (x, y) => {
          try {
            const sm = window.__SCENE_MANAGER
            if (sm && sm.camera && typeof sm.camera.toScreen === 'function') {
              const p = sm.camera.toScreen(x, y)
              if (p && typeof p.x === 'number') return [p.x, p.y]
            }
          } catch {}
          return null
        }

        const place = (label, point) => {
          const node = build()
          if (!node) return

          if (label) node.querySelector('[data-label]').textContent = label

          let screen = point ? toScreen(point.x, point.y) : null

          // No selection / no mappable point: still show up. Parked at
          // (-100,-100) the cursor is invisible even lit — reads, boots, and
          // ops that clear selection land bottom-center like a presence chip.
          if (!screen) {
            const parked = node.style.transform.indexOf('-100px') !== -1
            if (parked) screen = [Math.round(innerWidth / 2), Math.round(innerHeight * 0.82)]
          }

          if (screen) {
            node.style.transform = 'translate3d(' + Math.round(screen[0]) + 'px,' + Math.round(screen[1]) + 'px,0)'
          }

          node.style.opacity = '1'
          clearTimeout(hideTimer)
        }

        const idle = () => {
          if (!el) return
          clearTimeout(hideTimer)
          // Linger briefly so a fast op is still visible, then fade.
          hideTimer = setTimeout(() => { if (el) el.style.opacity = '0' }, 1400)
        }

        window[NS] = { place: place, idle: idle }
      })()
    </script>`
