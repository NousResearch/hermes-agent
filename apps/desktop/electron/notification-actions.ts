function isActionIndex(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value)
}

function notificationActionIndex(actionEvent: unknown, legacyIndex: unknown): number | null {
  if (actionEvent && typeof actionEvent === 'object' && 'actionIndex' in actionEvent) {
    const { actionIndex } = actionEvent as { actionIndex?: unknown }

    if (isActionIndex(actionIndex)) {
      return actionIndex
    }
  }

  return isActionIndex(legacyIndex) ? legacyIndex : null
}

export function resolveNotificationAction<T>(
  actions: readonly T[],
  actionEvent: unknown,
  legacyIndex: unknown
): T | null {
  const index = notificationActionIndex(actionEvent, legacyIndex)

  if (index === null || index < 0 || index >= actions.length) {
    return null
  }

  return actions[index] ?? null
}
