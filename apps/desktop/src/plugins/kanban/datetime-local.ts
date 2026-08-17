const padLocalField = (value: number) => String(value).padStart(2, '0')

export function formatLocalDateTime(date: Date): string {
  return `${date.getFullYear()}-${padLocalField(date.getMonth() + 1)}-${padLocalField(date.getDate())}T${padLocalField(date.getHours())}:${padLocalField(date.getMinutes())}`
}

export function parseLocalDateTime(value: string): Date | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value)

  if (!match) {
    return null
  }

  const parts = match.slice(1).map(Number)
  const date = new Date(parts[0], parts[1] - 1, parts[2], parts[3], parts[4], 0, 0)

  return formatLocalDateTime(date) === value ? date : null
}
