import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from '../types'

/** A locale's redesigned-shell wording: a sparse override of the base catalog holding ONLY the
 *  keys whose text differs there (Chats, Library, Agents, Routines, Machines). Typed against
 *  `Translations`, so renaming or removing a base key breaks the overlay at compile time. */
export type RedesignOverlay = TranslationOverride<Translations>
