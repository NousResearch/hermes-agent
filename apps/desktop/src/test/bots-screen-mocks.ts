import type * as BotsI18n from '@/plugins/hermes-bots/i18n'
import type * as ScreenPortal from '@/plugins/hermes-bots/screen-portal'

/** What a test steers and reads of Bot Mode's Screen: the tone the state hook reports and the rows
 *  it was asked about. */
export interface ScreenPluginSpy {
  bots?: unknown[]
  tone?: string
}

/** `@/plugins/hermes-bots/i18n` with the copy hook reading the bundles directly (no plugin runtime). */
export async function botsI18nMock(original: () => Promise<unknown>) {
  const module = (await original()) as typeof BotsI18n

  return { ...module, useBots: () => module.botsText() }
}

/** `@/plugins/hermes-bots/screen-portal` with its state hook in the test's hands. */
export async function screenPortalMock(original: () => Promise<unknown>, plugin: ScreenPluginSpy) {
  return {
    ...((await original()) as typeof ScreenPortal),
    useScreenPortalState: (bot: unknown) => {
      plugin.bots?.push(bot)

      return { lease: null, status: null, tone: plugin.tone ?? 'unknown' }
    }
  }
}
