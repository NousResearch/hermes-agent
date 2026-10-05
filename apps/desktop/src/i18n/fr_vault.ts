import type { Translations } from './types'

export const frVault: Translations['settings']['vault'] = {
  title: 'Mots de passe et identifiants',
  blurb:
    "Demandez à l'agent de se connecter à un site : il vous demandera l'identifiant la première fois, puis pourra le réutiliser. Les mots de passe sont chiffrés sur cet ordinateur et remplis directement dans la page ; le modèle ne les voit jamais.",
  count: n => `${n} enregistré${n === 1 ? '' : 's'}`,
  loadFailed: 'Impossible de charger les éléments du coffre',
  empty: 'Aucun élément enregistré',
  emptyDesc:
    "Vous n'avez rien à ajouter à l'avance. Demandez à l'agent de se connecter à un site et il vous demandera l'identifiant au moment voulu.",
  add: 'Ajouter',
  addTitle: 'Ajouter un identifiant, une carte ou une adresse',
  addDescription: "Enregistré chiffré sur cet ordinateur. L'agent ne voit jamais le mot de passe.",
  added: 'Enregistré.',
  adding: 'Enregistrement…',
  addConfirm: 'Enregistrer',
  kindField: 'Type',
  kinds: {
    login: 'Identifiant',
    payment: 'Carte bancaire',
    address: 'Adresse'
  },
  labelField: 'Libellé',
  labelPlaceholder: 'par ex. compte GitHub professionnel',
  labelRequired: 'Un libellé est requis.',
  originField: 'Origine du site',
  originPlaceholder: 'https://github.com',
  originPlaceholderCheckout: 'https://shop.example.com',
  originInvalid: 'Saisissez une URL valide, par exemple https://exemple.fr.',
  originAnySiteHint:
    "Laissez vide pour utiliser cette adresse sur n'importe quel site ; l'agent vous demande de confirmer le site à chaque remplissage.",
  anySite: "N'importe quel site",
  identifierTypeField: "Type d'identifiant",
  identifierTypes: {
    email: 'E-mail',
    phone: 'Téléphone',
    username: "Nom d'utilisateur"
  },
  identifierField: 'Identifiant',
  identifierShown: identifier => identifier,
  passwordField: 'Mot de passe',
  loginFieldsRequired: "L'identifiant et le mot de passe sont requis.",
  cardNumberField: 'Numéro de carte',
  cardNameField: 'Nom sur la carte',
  expMonthField: "Mois d'expiration",
  expYearField: "Année d'expiration",
  cvcField: 'CVC',
  postalField: 'Code postal',
  addressLine1Field: 'Adresse ligne 1',
  addressLine2Field: 'Adresse ligne 2',
  cityField: 'Ville',
  stateField: 'État / région',
  countryField: 'Pays',
  optional: '(facultatif)',
  createdOn: date => `Ajouté ${date}`,
  deleteAction: "Supprimer l'élément enregistré",
  otpField: "Clé d'authentificateur",
  otpPlaceholder: 'Secret Base32 ou lien otpauth://',
  otpHint:
    "La « clé d'installation » que le site affiche lorsque vous activez 2FA. Avec elle enregistrée, Hermes génère lui-même les codes.",
  twoFactorBadge: '2FA auto',
  deleteTitle: 'Supprimer cet élément ?',
  deleteDescription: label => `« ${label} » sera supprimé définitivement.`,
  deleteConfirm: 'Supprimer',
  sources: {
    title: 'Gestionnaires de mots de passe',
    blurb:
      "Les gestionnaires de mots de passe installés sont repérés automatiquement. L'agent vous demande de déverrouiller l'un la première fois qu'il en a besoin (une fois par session) ; seul un jeton de session reste en mémoire, et l'agent ne voit jamais votre mot de passe principal ou aucun identifiant.",
    toggleFailed: 'Impossible de mettre à jour le gestionnaire de mots de passe',
    notInstalled: name =>
      `Non détecté. Installez l’outil en ligne de commande ${name} et connectez-vous ; Hermes le repère automatiquement.`,
    disabledDesc: 'Détecté mais désactivé pour Hermes.',
    lockedDesc:
      "Détecté. L'agent vous demandera de le déverrouiller lorsqu'il en a besoin, ou déverrouillez maintenant.",
    unlockedDesc:
      "Déverrouillé pour cette session. Se verrouille automatiquement après 30 minutes d'inactivité ou lorsque Hermes se ferme.",
    statusLocked: 'Verrouillé',
    statusNotDetected: 'Non détecté',
    statusOff: 'Désactivé',
    statusUnlocked: 'Déverrouillé',
    unlock: 'Déverrouiller',
    unlocking: 'Déverrouillage…',
    lock: 'Verrouiller',
    unlocked: name => `${name} est déverrouillé pour cette session.`,
    unlockTitle: name => `Déverrouiller ${name}`,
    unlockDescription:
      "Entrez votre mot de passe principal. Il est transmis au gestionnaire de mots de passe sur cette machine et jeté — il n'est jamais stocké, enregistré, ni montré à l'agent.",
    masterPasswordPlaceholder: 'Mot de passe principal'
  }
}
