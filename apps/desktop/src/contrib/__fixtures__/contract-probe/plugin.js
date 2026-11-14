/**
 * Contract probe — a test fixture, not a shipped plugin. It registers ONE contribution in every
 * area the desktop plugin SDK exports, each rendering a `data-testid="probe-<area>"` marker (data
 * contributions carry a recognizable label instead), so a test can ask "does a plugin's
 * contribution reach the screen, and behind which control?" for every shell and interface mode.
 *
 * Plain ESM with `jsx()` calls, importing only `@hermes/plugin-sdk` and `react/jsx-runtime`: the
 * exact form the runtime loader accepts from `<hermes home>/desktop-plugins/contract-probe/plugin.js`,
 * so the same file loads in vitest (bundled import), in the browser harness and in Electron.
 *
 * Its page (`/probe`) reports `host.state.viewport` and sizes itself the way many plugin pages do —
 * `flex-1 min-h-0` on the root, no `h-full` — and marks whether it filled its container
 * (`data-probe-fill`), which is how a page layer that is not a flex container shows up.
 */

import {
  APPEARANCE_AREAS,
  CHAT_EMPTY_AREA,
  COMPOSER_AREAS,
  host,
  KANBAN_CARD_AREAS,
  KEYBINDS_AREA,
  MODEL_MENU_ROW_AREA,
  PALETTE_AREA,
  ROUTES_AREA,
  SESSION_ROW_AREAS,
  SIDEBAR_NAV_AREA,
  SIDEBAR_NAV_PREFS_AREA,
  SIDEBAR_PROFILE_GROUP_HEADER_AREA,
  STATUSBAR_AREAS,
  THEMES_AREA,
  TITLEBAR_AREAS,
  TOOL_MEDIA_RENDERER_AREA,
  TRANSCRIPT_DIRECTIVE_AREA,
  usePluginI18n,
  useValue,
  WorkspacePageHeaderControl
} from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

export const PROBE_ID = 'contract-probe'
export const PROBE_ROUTE = '/probe'

const marker = (area, text = area) =>
  jsx('span', {
    'data-testid': `probe-${area}`,
    className: 'text-[0.625rem] text-(--ui-text-tertiary)',
    children: text
  })

/** The probe's page: the viewport contract, an i18n string, a page-header control, and a flex root
 *  that reports whether it filled the box its route surface gave it. */
function ProbePage() {
  const viewport = useValue(host.state.viewport)
  const t = usePluginI18n(PROBE_ID)
  const root = useRef(null)
  const [fill, setFill] = useState('pending')

  useEffect(() => {
    const element = root.current
    let parent = element?.parentElement

    // The box the route surface gave the page: the nearest ancestor that has one.
    while (parent && getComputedStyle(parent).display === 'contents') {
      parent = parent.parentElement
    }

    if (!element || !parent) {
      return undefined
    }

    const measure = () =>
      setFill(
        element.getBoundingClientRect().height + 1 >= parent.getBoundingClientRect().height ? 'filled' : 'collapsed'
      )

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    observer.observe(parent)

    return () => observer.disconnect()
  }, [])

  return jsxs('div', {
    className: 'flex min-h-0 flex-1 flex-col gap-2 p-4',
    'data-probe-fill': fill,
    'data-testid': 'probe-route',
    ref: root,
    children: [
      jsxs('div', {
        className: 'flex items-center gap-2',
        children: [
          jsx('h1', { 'data-testid': 'probe-i18n', children: t('title') }),
          jsx(WorkspacePageHeaderControl, { children: marker('workspace.pageHeader') })
        ]
      }),
      jsx('pre', {
        'data-testid': 'probe-viewport',
        children: JSON.stringify({ narrow: viewport.narrow, width: viewport.width })
      }),
      jsx('div', { className: 'min-h-0 flex-1 rounded border border-dashed border-(--ui-border)', children: 'fill' })
    ]
  })
}

const pane = (id, title, data) => ({ id, area: 'panes', title, data, render: () => marker(id) })

/** A theme only has to be valid to list; the probe's is a plain dark palette. */
const PROBE_THEME = {
  name: 'contract-probe',
  label: 'Contract probe theme',
  description: 'Listed by the contract probe.',
  colors: {
    background: '#101014',
    foreground: '#f2f2f2',
    card: '#16161c',
    cardForeground: '#f2f2f2',
    muted: '#1c1c24',
    mutedForeground: '#a0a0aa',
    popover: '#16161c',
    popoverForeground: '#f2f2f2',
    primary: '#5b6cff',
    primaryForeground: '#ffffff',
    secondary: '#22222c',
    secondaryForeground: '#f2f2f2',
    accent: '#5b6cff',
    accentForeground: '#ffffff',
    border: '#2a2a34',
    input: '#2a2a34',
    ring: '#5b6cff',
    destructive: '#e5484d',
    destructiveForeground: '#ffffff'
  }
}

export default {
  id: PROBE_ID,
  name: 'Contract probe',
  description: 'Test fixture: one contribution in every plugin SDK area.',
  register(ctx) {
    ctx.i18n.register({ en: { title: 'Contract probe' } }, { redesign: { en: { title: 'Contract probe (redesign)' } } })

    ctx.registerMany([
      // Panes, one per placement role.
      pane('pane-right', 'Probe right', { placement: 'right', width: '240px' }),
      pane('pane-left', 'Probe left', { placement: 'left', width: '240px' }),
      pane('pane-bottom', 'Probe bottom', { placement: 'bottom', height: '120px' }),
      pane('pane-floating', 'Probe floating', {
        placement: 'floating',
        anchor: 'bottom-right',
        width: '200px',
        height: '80px'
      }),
      pane('pane-collapsible', 'Probe collapsible', { placement: 'left', collapsible: true, width: '300px' }),
      // A page and the nav row that reaches it; the preference moves that row to the front.
      { id: 'route', area: ROUTES_AREA, data: { path: PROBE_ROUTE }, render: () => jsx(ProbePage, {}) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: PROBE_ROUTE, label: 'Probe page', codicon: 'beaker' } },
      { id: 'nav-prefs', area: SIDEBAR_NAV_PREFS_AREA, data: { order: [`${PROBE_ID}:nav`] } },
      // Sidebar decorations.
      { id: 'row-leading', area: SESSION_ROW_AREAS.leading, data: { render: () => marker('sessionRow.leading', '◆') } },
      {
        id: 'row-trailing',
        area: SESSION_ROW_AREAS.trailing,
        data: { render: () => marker('sessionRow.trailing', '◇') }
      },
      {
        id: 'group-header',
        area: SIDEBAR_PROFILE_GROUP_HEADER_AREA,
        data: { render: route => marker('sidebar.profileGroup.header', `probe · ${route.profile}`) }
      },
      // Bars.
      { id: 'status-left', area: STATUSBAR_AREAS.left, render: () => marker('statusBar.left') },
      { id: 'status-right', area: STATUSBAR_AREAS.right, render: () => marker('statusBar.right') },
      { id: 'title-left', area: TITLEBAR_AREAS.left, render: () => marker('titleBar.left') },
      { id: 'title-center', area: TITLEBAR_AREAS.center, render: () => marker('titleBar.center') },
      { id: 'title-right', area: TITLEBAR_AREAS.right, render: () => marker('titleBar.right') },
      // A titlebar TOOL: a `TitlebarTool` the band draws as one of its own buttons.
      {
        id: 'title-tool',
        area: 'titleBar.tools.right',
        data: {
          id: 'contract-probe-tool',
          label: 'Probe tool',
          icon: marker('titleBar.tools', '◆'),
          onSelect: () => host.navigate(PROBE_ROUTE)
        }
      },
      // Commands.
      {
        id: 'palette',
        area: PALETTE_AREA,
        data: { id: `${PROBE_ID}.open`, label: 'Probe: open workspace', run: () => openProbeWorkspace() }
      },
      {
        id: 'keybind',
        area: KEYBINDS_AREA,
        data: {
          id: `${PROBE_ID}.page`,
          label: 'Probe: open page',
          category: 'Contract probe',
          defaults: ['mod+alt+shift+p'],
          run: () => host.navigate(PROBE_ROUTE)
        }
      },
      { id: 'theme', area: THEMES_AREA, data: PROBE_THEME },
      // The composer.
      { id: 'composer-top', area: COMPOSER_AREAS.top, render: () => marker('composer.top') },
      { id: 'composer-bottom', area: COMPOSER_AREAS.bottom, render: () => marker('composer.bottom') },
      { id: 'composer-underside', area: COMPOSER_AREAS.underside, render: () => marker('composer.underside') },
      { id: 'composer-leading', area: COMPOSER_AREAS.leading, render: () => marker('composer.leading', '◆') },
      { id: 'composer-actions', area: COMPOSER_AREAS.actions, render: () => marker('composer.actions', '◇') },
      { id: 'composer-middleware', area: COMPOSER_AREAS.middleware, data: { handler: draft => draft } },
      {
        id: 'composer-attachments',
        area: COMPOSER_AREAS.attachments,
        data: { label: 'Probe attachment', run: () => undefined }
      },
      { id: 'composer-micro', area: COMPOSER_AREAS.microActions, data: { resolve: () => [] } },
      {
        id: 'composer-at',
        area: COMPOSER_AREAS.atCompletions,
        data: { provide: () => [{ insert: '@probe', display: 'probe', meta: 'Contract probe' }] }
      },
      { id: 'composer-pill', area: COMPOSER_AREAS.modelPill, data: { label: () => null } },
      {
        id: 'composer-menu',
        area: COMPOSER_AREAS.contextMenu,
        data: { items: () => [{ label: 'Probe menu row', onSelect: () => undefined }] }
      },
      { id: 'model-row', area: MODEL_MENU_ROW_AREA, data: { decorate: () => ({ badge: 'probe' }) } },
      { id: 'kanban-badge', area: KANBAN_CARD_AREAS.badges, data: { decorate: () => ({ label: 'probe' }) } },
      // The transcript.
      {
        id: 'directive',
        area: TRANSCRIPT_DIRECTIVE_AREA,
        data: { name: 'probe', render: () => marker('transcript.directives') }
      },
      {
        id: 'tool-media',
        area: TOOL_MEDIA_RENDERER_AREA,
        data: { tools: ['contract_probe_tool'], render: () => marker('transcript.toolMedia') }
      },
      { id: 'chat-empty', area: CHAT_EMPTY_AREA, data: { render: () => marker('chat.empty') } },
      // Settings.
      { id: 'appearance', area: APPEARANCE_AREAS.extra, render: () => marker('appearance.extra') },
      {
        id: 'layout',
        area: 'layouts',
        title: 'Probe layout',
        data: {
          type: 'split',
          id: 'probe-layout-root',
          orientation: 'row',
          weights: [1, 3],
          children: [
            { type: 'group', id: 'probe-layout-left', panes: ['sessions'], active: 'sessions' },
            { type: 'group', id: 'probe-layout-main', panes: ['workspace'], active: 'workspace' }
          ]
        }
      }
    ])

    ctx.registerSettingsPage({ id: 'settings', title: 'Contract probe', render: () => marker('settings.plugins') })
  }
}

/** `host.openWorkspace`: a plugin-rendered main-area tile. */
export function openProbeWorkspace() {
  return host.openWorkspace('contract-probe', { title: 'Probe workspace', render: () => marker('openWorkspace') })
}
