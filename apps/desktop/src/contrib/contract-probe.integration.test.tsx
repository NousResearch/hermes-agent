// @vitest-environment jsdom
/**
 * The plugin contract, end to end, for every shell × interface mode: the contract probe
 * (`__fixtures__/contract-probe/plugin.js`, one contribution in every SDK area) is loaded through the
 * REAL runtime pipeline (the same blob import a disk plugin takes), and each area's own consumer
 * is mounted to show where the contribution lands. The on-screen reachability matrix (which control
 * reveals what, at which width) comes from the browser and e2e runs; this suite pins the
 * shell-dependent seams so a redesign change cannot silently drop a plugin surface.
 */
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { $widthClass } from '@/app/redesign/breakpoints'
import { TitlebarControls } from '@/app/shell/titlebar-controls'
import { StableNavigateProvider } from '@/app/stable-navigate'
import { ChatEmptySlot } from '@/components/assistant-ui/chat-empty-slot'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { registry } from '@/contrib/registry'
import { I18nProvider } from '@/i18n'
import { translatePlugin } from '@/i18n/plugin-i18n'
import { host } from '@/sdk'
import { $interfaceMode } from '@/store/interface-mode'
import { $shellChoice } from '@/store/shell-variant'
import { makeSessionInfo } from '@/test/session-info'

import { SimplifiedGlobalNav } from '../app/chat/sidebar/simplified-global-nav'
import { SimplifiedSessionList } from '../app/chat/sidebar/simplified-session-list'
import { useTitlebarToolContributions } from '../app/contrib/panes'
import { AppearanceExtraSlot } from '../app/settings/appearance-contrib'

import probeSource from './__fixtures__/contract-probe/plugin.js?raw'
import { $pluginRecords } from './plugins-store'
import { loadRuntimePlugin, unloadRuntimePlugin } from './runtime-loader'
import { routeBlobImportsToDataUrls } from './runtime-loader.test-helpers'

const PROBE = 'contract-probe'

let restoreBlobs = () => {}

beforeAll(async () => {
  restoreBlobs = routeBlobImportsToDataUrls()
  expect(await loadRuntimePlugin(probeSource, 'contract-probe/plugin.js')).toBe(PROBE)
})

afterAll(() => {
  unloadRuntimePlugin(PROBE)
  restoreBlobs()
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  $interfaceMode.set('advanced')
  $widthClass.set('wide')
})

const markers = (area: string) => screen.queryAllByTestId(`probe-${area}`)

/** The band as the shell wires it: contributed `titleBar.tools.*` arrive as its tool props. */
function Band({ path }: { path: string }) {
  return (
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider configClient={null} initialLocale="en">
        <WiredTitlebar />
      </I18nProvider>
    </MemoryRouter>
  )
}

function WiredTitlebar() {
  return (
    <TitlebarControls
      leftTools={useTitlebarToolContributions('left')}
      onOpenSettings={() => {}}
      tools={useTitlebarToolContributions('right')}
    />
  )
}

it('registers one contribution in every SDK area through the runtime loader', () => {
  expect($pluginRecords.get()[PROBE]).toMatchObject({ status: 'loaded' })

  const areas = [
    'panes',
    'routes',
    'sidebar.nav',
    'sidebarNav.prefs',
    'sessionRow.leading',
    'sessionRow.trailing',
    'sidebar.profileGroup.header',
    'statusBar.left',
    'statusBar.right',
    'titleBar.left',
    'titleBar.center',
    'titleBar.right',
    'titleBar.tools.right',
    'palette',
    'keybinds',
    'themes',
    'composer.top',
    'composer.bottom',
    'composer.underside',
    'composer.leading',
    'composer.actions',
    'composer.middleware',
    'composer.attachments',
    'composer.microActions',
    'composer.atCompletions',
    'composer.modelPill',
    'composer.contextMenu',
    'modelMenu.row',
    'kanban.card.badges',
    'transcript.directives',
    'transcript.toolMedia',
    'chat.empty',
    'appearance.extra',
    'layouts',
    'settings.plugins'
  ]

  for (const area of areas) {
    expect(registry.getArea(area).filter(c => c.source === `plugin:${PROBE}`).length, area).toBeGreaterThan(0)
  }

  expect(registry.getArea('panes').filter(c => c.source === `plugin:${PROBE}`)).toHaveLength(5)
})

describe.each([
  ['classic', false],
  ['redesign', true]
] as const)('%s shell', (shell, portal) => {
  describe.each(['simple', 'advanced'] as const)('%s mode', mode => {
    const setShell = () => {
      $shellChoice.set(portal)
      $interfaceMode.set(mode)
    }

    it('words its own strings in the shell’s wording layer', () => {
      setShell()
      expect(translatePlugin(PROBE, 'en', 'title', [])).toBe(portal ? 'Contract probe (redesign)' : 'Contract probe')
    })

    it('reads viewport.narrow as the 640px dock point', () => {
      setShell()
      $widthClass.set('medium')
      expect(host.state.viewport.get().narrow).toBe(false)
      $widthClass.set('compact')
      expect(host.state.viewport.get().narrow).toBe(true)
    })

    it('mounts every titleBar slot and its tool on the chat', () => {
      setShell()
      render(<Band path="/" />)

      for (const area of ['titleBar.left', 'titleBar.center', 'titleBar.right']) {
        expect(markers(area), area).toHaveLength(1)
      }

      expect(markers('titleBar.tools')).toHaveLength(1)
    })

    it('keeps titleBar slots mounted (redesign) or unmounts them (classic, as main) on an overlay page', () => {
      setShell()
      render(<Band path="/settings" />)

      expect(markers('titleBar.right')).toHaveLength(portal ? 1 : 0)
    })

    it('renders the chat-empty and appearance contributions', () => {
      setShell()
      render(
        <>
          <ChatEmptySlot sessionId="empty-session" />
          <AppearanceExtraSlot />
        </>
      )
      expect(markers('chat.empty')).toHaveLength(1)
      expect(markers('appearance.extra')).toHaveLength(1)
    })

    if (portal) {
      it('decorates the redesign’s session rows, leading before the title and trailing before the meta', () => {
        setShell()
        const rows = [makeSessionInfo({ id: 'chat-1', connection_id: 'local', profile: 'default', title: 'Chat one' })]
        render(
          <MemoryRouter>
            <StableNavigateProvider>
              <RootTooltipProvider>
                <SimplifiedSessionList
                  activeSessionId={null}
                  ambiguousIds={new Set()}
                  onArchiveSession={vi.fn()}
                  onBranchSession={vi.fn()}
                  onDeleteSession={vi.fn()}
                  onResumeSession={vi.fn()}
                  onUpdate={vi.fn()}
                  pinnedIds={new Set()}
                  sessions={rows}
                />
              </RootTooltipProvider>
            </StableNavigateProvider>
          </MemoryRouter>
        )

        const row = screen.getByText('Chat one').closest('.portal-session-row')!

        const order = [
          ...row.querySelectorAll(
            '[data-testid^="probe-sessionRow"], [data-slot="session-title"], [data-slot="session-meta"]'
          )
        ].map(node => node.getAttribute('data-testid') ?? node.getAttribute('data-slot'))

        expect(order).toEqual([
          'probe-sessionRow.leading',
          'session-title',
          'probe-sessionRow.trailing',
          'session-meta'
        ])
      })

      it('puts the probe’s page on the rail first among movable destinations (its nav preference)', () => {
        setShell()
        render(
          <MemoryRouter>
            <RootTooltipProvider>
              <SimplifiedGlobalNav onNavigate={vi.fn()} />
            </RootTooltipProvider>
          </MemoryRouter>
        )

        const labels = within(screen.getByRole('navigation'))
          .getAllByRole('button')
          .map(button => button.getAttribute('aria-label'))

        expect(labels.indexOf('Probe page')).toBe(2)
      })
    }

    // OPEN DECISION (status bar home in the redesign): classic renders statusBar.left/right in its
    // strip (Advanced, and Simple when the bar is shown); the redesign has no strip and lists the
    // items only inside the rail's System popover, under a collapsed "System" disclosure, and not at
    // all in Simple mode while the bar preference rests hidden. Pinned as current behaviour by the
    // browser/e2e matrix, not asserted here until the home is decided.
    it.todo('status bar contributions have a decided home in the redesign')
    // OPEN DECISION (theme fidelity): a contributed theme lists in both shells, but the redesign
    // paints its own tokens over theme CSS aimed at classic markup.
    it.todo('contributed themes paint the redesign as they paint classic')
  })
})
