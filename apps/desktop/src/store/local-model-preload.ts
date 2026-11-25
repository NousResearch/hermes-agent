import { atom } from 'nanostores'

/**
 * The managed local model the backend is loading because the user began typing in a new chat, or
 * null. Written by `useLocalModelPreload`, read by the composer's hint; it is presentation only, so
 * it lives in this window and is cleared whenever the draft stops being a typing new chat.
 */
export const $preloadingModel = atom<null | string>(null)
