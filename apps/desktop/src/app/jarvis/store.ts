import { atom } from 'nanostores'

import { initialJarvisUiState, type JarvisEvent, reduceJarvisEvent } from './projector'

export const $jarvisUi = atom(initialJarvisUiState())

export const publishJarvisEvent = (event: JarvisEvent): void => {
  $jarvisUi.set(reduceJarvisEvent($jarvisUi.get(), event))
}

export const resetJarvisSession = (sessionId: string | null): void => {
  $jarvisUi.set({ ...initialJarvisUiState(), sessionId })
}
