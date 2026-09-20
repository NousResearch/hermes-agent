import { atom } from 'nanostores'

import { initialJarvisUiState, reduceJarvisEvent, type JarvisEvent } from './projector'

export const $jarvisUi = atom(initialJarvisUiState())

export const publishJarvisEvent = (event: JarvisEvent): void => {
  $jarvisUi.set(reduceJarvisEvent($jarvisUi.get(), event))
}

export const resetJarvisSession = (sessionId: string): void => {
  $jarvisUi.set({ ...initialJarvisUiState(), sessionId })
}
