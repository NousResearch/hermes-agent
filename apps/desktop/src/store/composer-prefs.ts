import { atom } from 'nanostores'

export const $composerEnterSends = atom<boolean>(true)

export function applyComposerPrefsFromConfig(config: { desktop?: { composer?: { enter_sends?: unknown } } }): void {
  const value = config.desktop?.composer?.enter_sends

  $composerEnterSends.set(typeof value === 'boolean' ? value : true)
}
