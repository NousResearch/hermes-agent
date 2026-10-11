/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  fr.ts (FILE_LINES cap). */
export const frSettingsConfig = {
  keepAwakeTitle: "Garder l'ordinateur éveillé",
  keepAwakeDesc:
    "Empêcher cette machine de se mettre en veille. « Pendant le travail » ne s'applique que pendant qu'un tour est en cours : les exécutions nocturnes continuent sans garder le portable éveillé toute la semaine. L'écran peut toujours s'obscurcir.",
  keepAwakeOff: 'Désactivé',
  keepAwakeWhileWorking: 'Pendant le travail',
  keepAwakeAlways: 'Toujours',
  managedFieldHint: 'Géré par votre administrateur{source} — lecture seule',
  managedRejectedTitle: "Certains paramètres n'ont pas été enregistrés",
  managedRejectedNotice: "Non enregistré (géré par votre administrateur) : {keys}",
}
