import {
  APPEARANCE_AREAS,
  COMPOSER_AREAS,
  SIDEBAR_SESSIONS_HEADER_AREA,
  type HermesPlugin,
  PALETTE_AREA,
  type PaletteContribution,
} from '@hermes/plugin-sdk'

import { PROJECT_PICKER_LOCALES } from './i18n'
import { $showPicker, bindVisibility, ProjectPicker, ProjectPickerSettings } from './picker'

const plugin: HermesPlugin = {
  id: 'project-picker',
  name: 'Project Picker',
  description: 'Pick a project folder for a new chat from the composer or conversation sidebar.',
  defaultEnabled: true,
  register(ctx) {
    // Own strings only — never core en.ts. The disposer rides on ctx.
    ctx.i18n.register(PROJECT_PICKER_LOCALES)
    // Visibility toggle: hydrate from the plugin's own storage (kanban
    // pattern) and retire the subscription on unload/disable.
    ctx.onDispose(bindVisibility(ctx.storage))
    ctx.register({
      id: 'picker',
      area: COMPOSER_AREAS.actions,
      render: () => <ProjectPicker />,
    })
    ctx.register({
      id: 'sidebar-picker',
      area: SIDEBAR_SESSIONS_HEADER_AREA,
      render: () => <ProjectPicker surface="sidebar" />,
    })
    ctx.register({
      id: 'settings',
      area: APPEARANCE_AREAS.chatDisplay,
      render: () => <ProjectPickerSettings />,
    })

    // Palette door to the same toggle: shares $showPicker (and its storage
    // key) with the settings row, so either surface flips the other.
    const registerPalette = () =>
      ctx.register({
        id: 'toggle',
        area: PALETTE_AREA,
        data: {
          id: 'project-picker.toggle',
          label: ctx.i18n.t('palette.label'),
          keywords: ['project', 'picker', 'show', 'hide', 'toggle', 'composer', 'on', 'off', 'enable', 'disable'],
          run: () => $showPicker.set(!$showPicker.get()),
          detail: () => ($showPicker.get() ? 'on' : 'off'),
          detailVariant: 'state',
          keepOpen: true,
        } satisfies PaletteContribution,
      })

    let disposePalette = registerPalette()
    ctx.i18n.onLocaleChange(() => {
      disposePalette()
      disposePalette = registerPalette()
    })
  },
}

export default plugin
