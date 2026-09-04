/**
 * desktop-install-local-card.ts
 *
 * Pure presentation derivation for the setup screen's local card. The card
 * has four states, keyed off the `local` field the backend stamps on every
 * resolved backend:
 *
 *   none           — no Hermes on this machine: the card IS an install offer.
 *   installed      — a runtime resolved (PATH hermes, active root, …): clicking
 *                    starts the existing runtime; nothing downloads.
 *   bundled        — a healthy bundled install: the payload is the runtime.
 *   bundled-damaged— a bundled install whose payload failed to resolve: the
 *                    card is disabled (never an install action); reinstall the
 *                    app instead.
 *
 * Extracted so the disabled/title/desc/footer logic is unit-testable without
 * rendering the overlay.
 */

export type LocalCardState = 'none' | 'installed' | 'bundled' | 'bundled-damaged'

export interface LocalCardPresentation {
  /** i18n key into `t.install` for the card title. */
  title: 'installLocalTitle' | 'useLocalTitle' | 'bundledDamagedTitle'
  /** i18n key into `t.install` for the card body. */
  desc: 'installLocalDesc' | 'useLocalDesc' | 'bundledLocalDesc' | 'bundledDamagedDesc'
  /** When true the card cannot be clicked — there is no install to fire. */
  disabled: boolean
  /** Whether the "Will install to <root>" footer is accurate for this state. */
  showInstallTo: boolean
}

export function localCardPresentation(local: LocalCardState | undefined): LocalCardPresentation {
  switch (local) {
    case 'installed':
      return {
        title: 'useLocalTitle',
        desc: 'useLocalDesc',
        disabled: false,
        showInstallTo: false
      }

    case 'bundled':
      return {
        title: 'useLocalTitle',
        desc: 'bundledLocalDesc',
        disabled: false,
        showInstallTo: false
      }

    case 'bundled-damaged':
      return {
        title: 'bundledDamagedTitle',
        desc: 'bundledDamagedDesc',
        disabled: true,
        showInstallTo: false
      }

    case 'none':

    default:
      return {
        title: 'installLocalTitle',
        desc: 'installLocalDesc',
        disabled: false,
        showInstallTo: true
      }
  }
}
