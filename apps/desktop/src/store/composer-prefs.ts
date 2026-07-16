import { atom } from 'nanostores'

export const $composerEnterSends = atom<boolean>(true)

export function applyComposerPrefsFromConfig(config: { desktop?: { composer?: { enter_sends?: unknown } } }): void {
  $composerEnterSends.set(config.desktop?.composer?.enter_sends !== false)
}
