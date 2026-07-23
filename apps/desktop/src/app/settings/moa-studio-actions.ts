import { SETTINGS_ROUTE } from '../routes'

export const MOA_STUDIO_ROUTE = `${SETTINGS_ROUTE}?tab=moa`

export interface ModelSelectionHandler {
  (selection: { model: string; provider: string }): Promise<boolean> | void
}

/** Route a Studio activation through the same persistent session selection path as the model picker. */
export function selectMoaPresetInChat(selectModel: ModelSelectionHandler, name: string): Promise<boolean> | void {
  return selectModel({ model: name, provider: 'moa' })
}
