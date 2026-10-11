/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  es.ts (FILE_LINES cap). */
export const esSettingsConfig = {
  keepAwakeTitle: 'Mantener el equipo activo',
  keepAwakeDesc:
    'Impide que este equipo entre en reposo. «Mientras trabaja» solo se aplica mientras hay un turno en curso: las ejecuciones nocturnas continúan sin mantener el portátil despierto toda la semana. La pantalla puede seguir atenuándose.',
  keepAwakeOff: 'Desactivado',
  keepAwakeWhileWorking: 'Mientras trabaja',
  keepAwakeAlways: 'Siempre',
  managedFieldHint: 'Gestionado por su administrador{source} — solo lectura',
  managedRejectedTitle: 'Algunos ajustes no se guardaron',
  managedRejectedNotice: 'No guardado (gestionado por su administrador): {keys}',
}
