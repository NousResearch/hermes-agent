import { describe, expect, it } from 'vitest'

import {
  matchSpellCheckerLanguage,
  parseSpellCheckerLanguageList,
  resolveSpellCheckerLanguages
} from './spellcheck-languages'

const AVAILABLE = ['de-DE', 'en-GB', 'en-US', 'es', 'es-MX', 'fr-FR', 'it-IT', 'pt-BR', 'pt-PT']

describe('spellchecker language selection', (): void => {
  it('loads every language from the override, in order', (): void => {
    expect(
      resolveSpellCheckerLanguages({ override: 'pt-BR, en-US es', locale: 'en-US', available: AVAILABLE })
    ).toEqual(['pt-BR', 'en-US', 'es'])
  })

  it('does not append the locale or system languages when an override is set', (): void => {
    expect(
      resolveSpellCheckerLanguages({ override: 'fr-FR', preferred: ['de-DE'], locale: 'en-US', available: AVAILABLE })
    ).toEqual(['fr-FR'])
  })

  it('uses every OS preferred language plus the locale when there is no override', (): void => {
    expect(
      resolveSpellCheckerLanguages({ preferred: ['pt-BR', 'it-IT', 'en-US'], locale: 'en-US', available: AVAILABLE })
    ).toEqual(['pt-BR', 'it-IT', 'en-US'])
  })

  it('skips bare languages already covered by a regional dictionary', (): void => {
    const available = [...AVAILABLE, 'en', 'pt']

    expect(
      resolveSpellCheckerLanguages({ preferred: ['pt-BR', 'pt', 'en-US', 'en', 'es'], locale: 'pt-BR', available })
    ).toEqual(['pt-BR', 'en-US', 'es'])
  })

  it('normalizes POSIX locales and widens bare or unknown regions', (): void => {
    expect(matchSpellCheckerLanguage('pt_BR.UTF-8', AVAILABLE)).toBe('pt-BR')
    expect(matchSpellCheckerLanguage('es-AR', AVAILABLE)).toBe('es')
    expect(matchSpellCheckerLanguage('pt', AVAILABLE)).toBe('pt-BR')
    expect(matchSpellCheckerLanguage('FR-fr', AVAILABLE)).toBe('fr-FR')
    expect(matchSpellCheckerLanguage('xx-YY', AVAILABLE)).toBeNull()
  })

  it('drops unknown tags and duplicates', (): void => {
    expect(resolveSpellCheckerLanguages({ override: 'klingon,pt,pt-BR,en', available: AVAILABLE })).toEqual([
      'pt-BR',
      'en-GB'
    ])
  })

  it('falls back to en-US when nothing matches', (): void => {
    expect(resolveSpellCheckerLanguages({ preferred: ['C'], locale: 'xx', available: AVAILABLE })).toEqual(['en-US'])
    expect(resolveSpellCheckerLanguages({ override: 'klingon', available: [] })).toEqual(['en-US'])
  })

  it('ignores the C/POSIX pseudo-locales in lists', (): void => {
    expect(parseSpellCheckerLanguageList('C POSIX en_US.UTF-8:pt_BR')).toEqual(['en-US', 'pt-BR'])
  })
})
