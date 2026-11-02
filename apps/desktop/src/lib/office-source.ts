export function officeSourceName(path: string | undefined, label: string): string {
  return path?.trim() || label
}
