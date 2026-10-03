import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

// The picker under test is a plugin: at runtime it reaches the project tree
// only through host.projects. This test seeds the REAL stores so the host
// verbs read true state — the same dispensation kanban's placement tests use.
// eslint-disable-next-line no-restricted-imports
import { COMPOSER_AREAS } from '@/app/chat/composer/contrib'
// eslint-disable-next-line no-restricted-imports
import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
// eslint-disable-next-line no-restricted-imports
import { PALETTE_AREA } from '@/app/command-palette/contrib'
// eslint-disable-next-line no-restricted-imports
import { APPEARANCE_AREAS } from '@/app/settings/appearance-contrib'
// eslint-disable-next-line no-restricted-imports
import { createPluginContext } from '@/contrib/plugin'
// eslint-disable-next-line no-restricted-imports
import { Slot } from '@/contrib/react/slot'
// eslint-disable-next-line no-restricted-imports
import { registry } from '@/contrib/registry'
// eslint-disable-next-line no-restricted-imports
import { setRuntimeI18nLocale } from '@/i18n/runtime'
// eslint-disable-next-line no-restricted-imports
import { $activeGatewayProfile, setShowAllProfiles } from '@/store/profile'
// eslint-disable-next-line no-restricted-imports
import { $projectTree, $startWorkSessionRequest } from '@/store/projects'
// eslint-disable-next-line no-restricted-imports
import { $activeSessionId } from '@/store/session'

import { PROJECT_PICKER_LOCALES } from './i18n'
import { $showPicker } from './picker'
import plugin from './plugin'

const STORAGE_KEY = 'hermes.plugin.project-picker.showPicker'

const treeNode = (overrides: Record<string, unknown>) => ({
  id: 'p_x',
  label: 'X',
  path: '/repos/x',
  repos: [],
  sessionCount: 0,
  ...overrides,
})

const seedTree = () => {
  $projectTree.set([
    treeNode({ id: 'p_web', label: 'Website', path: '/repos/website', sessionCount: 2 }),
    treeNode({ id: 'p_api', label: 'API', path: '/repos/api', sessionCount: 0 }),
    treeNode({ id: NO_PROJECT_ID, label: 'Home', path: null, isNoProject: true, sessionCount: 1 }),
  ] as never)
}

const disposers: Array<() => void> = []

/** Register the plugin the way the bundled loader does (scoped context). */
function registerPlugin() {
  plugin.register(createPluginContext(plugin.id, dispose => disposers.push(dispose)))
}

function disposeAll() {
  disposers.splice(0).forEach(dispose => dispose())
}

/** Open the picker's menu and return the named row. Radix's dropdown trigger
 *  opens on pointerdown (not on the synthetic 'click' fireEvent alone would
 *  dispatch), so fire the full mouse sequence a real click produces — the
 *  same sequence `session-actions-menu.test.tsx` uses. */
async function openMenuItem(name: string) {
  const trigger = screen.getByRole('button', { name: 'Project' })

  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)

  return screen.findByRole('menuitem', { name })
}

beforeEach(() => {
  $activeGatewayProfile.set('default')
  setShowAllProfiles(false)
  $startWorkSessionRequest.set(null)
  $activeSessionId.set(null)
  window.localStorage.removeItem(STORAGE_KEY)
  $showPicker.set(true)
  seedTree()
})

afterEach(() => {
  cleanup()
  disposeAll()
  setShowAllProfiles(false)
  $startWorkSessionRequest.set(null)
  $activeSessionId.set(null)
  $projectTree.set([])
  window.localStorage.removeItem(STORAGE_KEY)
  $showPicker.set(true)
})

describe('project-picker plugin', () => {
  it('contributes an enabled picker to the composer toolbar, right before the model pill', () => {
    registerPlugin()

    // Literal area id: the composer renders this slot inline before its
    // controls (the model pill is the first control). Any other area is a
    // sidebar/statusbar/pane workaround and fails this test.
    const entries = registry
      .getArea(COMPOSER_AREAS.actions)
      .filter(entry => entry.id === `${plugin.id}:picker`)

    expect(COMPOSER_AREAS.actions).toBe('composer.actions')
    expect(plugin.defaultEnabled).toBe(true)
    expect(entries).toHaveLength(1)
    expect(typeof entries[0]?.render).toBe('function')
  })

  it('contributes its visibility toggle beside the conversation display settings', () => {
    registerPlugin()

    const entries = registry
      .getArea(APPEARANCE_AREAS.chatDisplay)
      .filter(entry => entry.id === `${plugin.id}:settings`)

    expect(APPEARANCE_AREAS.chatDisplay).toBe('appearance.chatDisplay')
    expect(entries).toHaveLength(1)
    expect(typeof entries[0]?.render).toBe('function')
  })

  it('contributes a palette toggle bound to the same visibility atom and storage key', () => {
    registerPlugin()

    const entries = registry.getArea(PALETTE_AREA).filter(entry => entry.id === `${plugin.id}:toggle`)

    expect(PALETTE_AREA).toBe('palette')
    expect(entries).toHaveLength(1)

    const data = entries[0]?.data as {
      id: string
      label: string
      run: () => void
      detail?: () => string
    }

    expect(data.id).toBe('project-picker.toggle')
    expect(data.label).toMatch(/project picker/i)
    expect(typeof data.run).toBe('function')

    expect($showPicker.get()).toBe(true)

    act(() => {
      data.run()
    })

    expect($showPicker.get()).toBe(false)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('false')
    expect(data.detail?.()).toBe('off')

    act(() => {
      data.run()
    })

    expect($showPicker.get()).toBe(true)
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('true')
    expect(data.detail?.()).toBe('on')
  })

  it('lists active-profile projects through the real slot and starts a fresh draft in the picked folder', async () => {
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    // The trigger keeps the pill's accessible name — the e2e targets it.
    const trigger = screen.getByRole('button', { name: 'Project' })

    expect(trigger.getAttribute('aria-haspopup')).toBe('menu')

    const api = await openMenuItem('API')

    expect(await screen.findByRole('menuitem', { name: 'Website' })).toBeDefined()
    expect(screen.queryByRole('menuitem', { name: 'Home' })).toBeNull()

    fireEvent.click(api)

    expect($startWorkSessionRequest.get()?.path).toBe('/repos/api')
  })

  it('appears when the project tree arrives after the composer mounted', async () => {
    $projectTree.set([])
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    // The tree is fetched asynchronously and the composer mounts before it
    // lands — a one-shot read at mount renders nothing and never recovers.
    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()

    act(() => {
      seedTree()
    })

    const picker = await screen.findByRole('button', { name: 'Project' })

    expect(picker).toBeDefined()

    fireEvent.pointerDown(picker, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(picker, { button: 0, pointerType: 'mouse' })
    fireEvent.click(picker)
    expect(await screen.findByRole('menuitem', { name: 'Website' })).toBeDefined()
  })

  it('says why it cannot list while viewing all profiles instead of rendering a dead picker', () => {
    setShowAllProfiles(true)
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()
    expect(screen.getByText(/all profiles/i)).toBeDefined()
  })

  it('hides the picker while the settings toggle is off and shows it again when on', async () => {
    registerPlugin()

    render(
      <>
        <Slot area={COMPOSER_AREAS.actions} />
        <Slot area={APPEARANCE_AREAS.chatDisplay} />
      </>
    )

    expect(screen.getByRole('button', { name: 'Project' })).toBeDefined()

    const toggle = screen.getByRole('switch', { name: 'Show project picker' })

    expect(toggle.getAttribute('aria-checked')).toBe('true')

    fireEvent.click(toggle)

    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()

    fireEvent.click(screen.getByRole('switch', { name: 'Show project picker' }))

    expect(await screen.findByRole('button', { name: 'Project' })).toBeDefined()
  })

  it('persists the toggle across a remount', async () => {
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    expect(screen.getByRole('button', { name: 'Project' })).toBeDefined()

    act(() => {
      $showPicker.set(false)
    })

    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('false')

    // A remount (unload + fresh register, the way the bundled loader does)
    // re-hydrates from storage, so the toggle survives it.
    cleanup()
    disposeAll()
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()

    act(() => {
      $showPicker.set(true)
    })

    expect(await screen.findByRole('button', { name: 'Project' })).toBeDefined()
  })

  it('renders on a fresh draft with no focused session', () => {
    $activeSessionId.set(null)
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    expect(screen.getByRole('button', { name: 'Project' })).toBeDefined()
  })

  it('does not render once the user is inside a conversation', async () => {
    registerPlugin()

    render(<Slot area={COMPOSER_AREAS.actions} />)

    expect(screen.getByRole('button', { name: 'Project' })).toBeDefined()

    act(() => {
      $activeSessionId.set('runtime-live')
    })

    expect(screen.queryByRole('button', { name: 'Project' })).toBeNull()
    expect(screen.queryByText(/all profiles/i)).toBeNull()

    // Back to a fresh draft: the picker returns without a remount.
    act(() => {
      $activeSessionId.set(null)
    })

    expect(await screen.findByRole('button', { name: 'Project' })).toBeDefined()
  })

  it('registers its own locale bundle, so its strings resolve instead of falling back to keys', () => {
    setRuntimeI18nLocale('en')

    const ctx = createPluginContext(plugin.id, dispose => disposers.push(dispose))

    plugin.register(ctx)

    // Resolution goes through the plugin translator: if the bundle were not
    // registered on ctx, these would come back as the raw dotted keys.
    expect(ctx.i18n.t('picker.label')).toBe('Project')
    expect(ctx.i18n.t('picker.blocked')).toMatch(/all profiles/i)
    expect(ctx.i18n.t('settings.label')).toBe('Show project picker')
    expect(ctx.i18n.t('settings.description')).toMatch(/composer toolbar/i)
    expect(ctx.i18n.t('palette.label')).toMatch(/project picker/i)
    expect(PROJECT_PICKER_LOCALES.en).toBeDefined()
  })
})
