import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'

/** What a test steers and reads of Bot Mode's Screen: the tone the state hook reports, the rows it
 *  was asked about, and the Open action with the route it was built for. */
export interface ScreenPluginSpy {
  bots?: unknown[]
  open: () => unknown
  route: unknown
  tone?: string
}

/** `@/plugins/hermes-bots/i18n` with the copy hook reading the bundles directly (no plugin runtime). */
export async function botsI18nMock(original: () => Promise<unknown>) {
  const module = (await original()) as typeof BotsI18n

  return { ...module, useBots: () => module.botsText() }
}

/** `@/plugins/hermes-bots/screen-portal` with its state hook and Open action in the test's hands. */
export async function screenPortalMock(original: () => Promise<unknown>, plugin: ScreenPluginSpy) {
  return {
    ...((await original()) as typeof ScreenPortal),
    profileScreenCommand: (route: unknown) => {
      plugin.route = route

      return { label: '', run: plugin.open }
    },
    useScreenPortalState: (bot: unknown) => {
      plugin.bots?.push(bot)

      return { lease: null, status: null, tone: plugin.tone ?? 'unknown' }
    }
  }
}
