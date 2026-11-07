import type { ContentDictionary } from './index'

/**
 * One lazy loader per language that ships catalog content. The JSON becomes its own chunk and is
 * fetched only when the Capabilities page runs in that language; English never loads anything.
 * Add a language by dropping `<lang>.json` next to this file and registering it here.
 */
export const CONTENT_DICTIONARY_LOADERS: Record<string, () => Promise<ContentDictionary>> = {
  ru: () => import('./ru.json').then(module => module.default as ContentDictionary)
}
