import type { TranslationOverrides } from './define-locale'

// Provider-source copy for Capabilities > Skills, wired into fr.ts as `catalog`.
export const frCatalog = {
  provider: {
    review: (action: string) => `Vérifier ${action}`,
    confirm: (action: string) => `Confirmer ${action}`,
    actions: { install: 'l’installation', update: 'la mise à jour', uninstall: 'la désinstallation' },
    managed: 'Géré par cette source. Vérifiez les modifications ici plutôt que de modifier ou supprimer des fichiers locaux.',
    loading: 'Chargement de la source…',
    empty: 'Cette source ne contient aucune compétence correspondante.',
    signIn: 'Connectez-vous via le plugin fournisseur pour parcourir cette source.',
    unavailable: 'Cette source est indisponible. Vos compétences locales sont inchangées.',
    incomplete: 'La vérification est incomplète ou la politique du fournisseur n’autorise pas la confirmation.',
    changed: 'Cette vérification n’est plus à jour. Fermez-la et demandez une nouvelle vérification.',
    unknown: 'Le résultat n’est pas encore connu. Récupérez la confirmation d’origine avant de lancer une autre action.',
    recover: 'Récupérer l’opération d’origine',
    finalizing: 'Finalisation de l’état local…',
    setup: 'Instructions de configuration uniquement. La confirmation n’exécute pas ces instructions.',
    more: (source: string) => `Charger plus : ${source}`,
    preview: 'Aperçu de la compétence',
    status: {
      not_installed: 'Non installée', installed: 'Installée', removed: 'Supprimée hors de cette source',
      unmanaged_local: 'Copie locale non gérée', pending: 'Opération en attente', unavailable: 'Source indisponible',
      unknown: 'Résultat inconnu', completed: 'Terminée', failed: 'Opération refusée'
    }
  }
} satisfies TranslationOverrides['catalog']
