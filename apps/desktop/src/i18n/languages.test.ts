import { describe, expect, it } from 'vitest'

import { fr } from './fr'
import { DEFAULT_LOCALE, isLocale, isSupportedLocaleValue, localeConfigValue, normalizeLocale } from './languages'

describe('desktop i18n languages', () => {
  it('keeps every fixed French sidebar navigation label non-empty', () => {
    expect(fr.sidebar.nav).toMatchObject({
      'new-session': 'Nouvelle session',
      artifacts: 'Artefacts',
      cron: 'Tâches planifiées',
      messaging: 'Messagerie',
      skills: 'Capacités'
    })
  })

  it('keeps the complete French session filter menu localized', () => {
    expect(fr.sidebar.filterMenu).toMatchObject({
      archived: 'Archivées',
      collapseAll: 'Tout replier',
      filters: 'Filtres',
      grouping: 'Regroupement',
      inboxStyle: 'Style boîte de réception',
      ordering: 'Tri',
      pullRequest: 'Demande de fusion',
      resetToDefaults: 'Rétablir les valeurs par défaut',
      show: 'Afficher'
    })
    expect(fr.sidebar.filterMenu.options).toMatchObject({
      closed: 'Fermées',
      cost: 'Coût',
      created: 'Création',
      idle: 'Inactives',
      merged: 'Fusionnées',
      needsInput: 'Saisie requise',
      noPullRequest: 'Sans PR',
      open: 'Ouvertes',
      tokens: 'Jetons',
      unread: 'Non lues',
      updated: 'Dernière activité',
      working: 'En cours'
    })
  })

  it('keeps reviewed French copy localized and plural-aware', () => {
    expect(fr.settings.gateway.sshErrHostKey).not.toContain('CHANGED')
    expect(fr.settings.mcp.testOk(1)).toContain('1 outil disponible')
    expect(fr.settings.mcp.testOk(2)).toContain('2 outils disponibles')
    expect(fr.skills.bulkUpdated(2)).toContain('2 éléments mis à jour')
    expect(fr.starmap.importSuccess(2)).toBe('Carte chargée avec 2 nœuds.')
    expect(fr.commandCenter.generatePet.hatchRow('', 2, 3)).toBe("Dessin de l'image 2 sur 3…")
    expect(fr.notifications.updateReadyMessage(1)).toContain('1 nouvelle modification disponible.')
    expect(fr.notifications.updateReadyMessage(2)).toContain('2 nouvelles modifications disponibles.')
    expect(fr.settings.appearance.embedsReset(2)).toContain('2 services autorisés')
    expect(fr.composer.attachments(2)).toBe('2 pièces jointes')
    expect(fr.statusStack.coding.changed(2)).toBe('2 modifiés')
    expect(fr.updates.moreChanges(2)).toBe('+ 2 changements supplémentaires inclus.')
  })

  it('normalizes supported locale aliases', () => {
    expect(normalizeLocale('en')).toBe('en')
    expect(normalizeLocale('EN-US')).toBe('en')
    expect(normalizeLocale('fr')).toBe('fr')
    expect(normalizeLocale('FR-FR')).toBe('fr')
    expect(normalizeLocale(' fr_fr ')).toBe('fr')
    expect(normalizeLocale('French')).toBe('fr')
    expect(normalizeLocale(' français ')).toBe('fr')
    expect(normalizeLocale('zh')).toBe('zh')
    expect(normalizeLocale('zh-CN')).toBe('zh')
    expect(normalizeLocale('zh-Hans')).toBe('zh')
    expect(normalizeLocale(' zh_hans_cn ')).toBe('zh')
    expect(normalizeLocale('zh-Hant')).toBe('zh-hant')
    expect(normalizeLocale('zh-TW')).toBe('zh-hant')
    expect(normalizeLocale('zh_HK')).toBe('zh-hant')
    expect(normalizeLocale('ja')).toBe('ja')
    expect(normalizeLocale('ja-JP')).toBe('ja')
    expect(normalizeLocale('ar')).toBe('ar')
    expect(normalizeLocale('AR-SA')).toBe('ar')
    expect(normalizeLocale(' ar_eg ')).toBe('ar')
    expect(normalizeLocale('ru')).toBe('ru')
    expect(normalizeLocale('RU-RU')).toBe('ru')
    expect(normalizeLocale(' ru_ru ')).toBe('ru')
    expect(normalizeLocale('Русский')).toBe('ru')
  })

  it('falls back to English for empty or unsupported values', () => {
    expect(normalizeLocale(null)).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('')).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('de')).toBe(DEFAULT_LOCALE)
  })

  it('distinguishes exact locale ids from supported config aliases', () => {
    expect(isSupportedLocaleValue('zh-CN')).toBe(true)
    expect(isSupportedLocaleValue('zh-TW')).toBe(true)
    expect(isSupportedLocaleValue('ja-JP')).toBe(true)
    expect(isSupportedLocaleValue('ru-RU')).toBe(true)
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('fr-FR')).toBe(false)
    expect(isLocale('fr')).toBe(true)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(true)
    expect(isLocale('ja')).toBe(true)
    expect(isLocale('ar')).toBe(true)
    expect(isLocale('ru')).toBe(true)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('fr')).toBe('fr')
    expect(localeConfigValue('zh')).toBe('zh')
    expect(localeConfigValue('zh-hant')).toBe('zh-hant')
    expect(localeConfigValue('ja')).toBe('ja')
    expect(localeConfigValue('ar')).toBe('ar')
    expect(localeConfigValue('ru')).toBe('ru')
  })
})
