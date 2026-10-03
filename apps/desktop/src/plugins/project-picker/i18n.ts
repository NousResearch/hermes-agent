/**
 * Plugin-scoped i18n for the project picker — bundles shipped under the plugin
 * id via `ctx.i18n.register`, never touching core `en.ts`. The active locale is
 * always the app's (`display.language`); `usePluginI18n('project-picker')`
 * resolves against it and re-renders on a locale switch or a late registration.
 */
import type { PluginLocaleBundles } from '@hermes/plugin-sdk'

export const PROJECT_PICKER_LOCALES: PluginLocaleBundles = {
  en: {
    picker: {
      /** Accessible name of the select — also what the e2e targets. */
      label: 'Project',
      placeholder: 'Project…',
      title: 'Start a new chat in a project folder',
      sidebarLabel: 'Choose a project for a new chat',
      blocked: 'Projects unavailable while viewing all profiles',
      blockedTitle: 'Switch to a single profile to pick a project',
      error: 'Could not start a session in that project',
    },
    settings: {
      label: 'Show project picker',
      description: 'Show the project picker in the composer toolbar and conversation sidebar.',
    },
    palette: {
      label: 'Toggle project picker',
    },
  },
}
