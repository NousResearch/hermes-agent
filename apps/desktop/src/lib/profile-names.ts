const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

const RESERVED_PROFILE_NAMES = new Set(['hermes', 'default', 'test', 'tmp', 'root', 'sudo'])

interface ProfileNameCopy {
  nameHint: string
  reservedNameHint: string
}

export function hasProfileNameSyntax(name: string): boolean {
  return PROFILE_NAME_RE.test(name.trim())
}

export function isReservedProfileName(name: string): boolean {
  return RESERVED_PROFILE_NAMES.has(name.trim().toLowerCase())
}

export function isValidProfileName(name: string): boolean {
  const trimmed = name.trim()

  return hasProfileNameSyntax(trimmed) && !isReservedProfileName(trimmed)
}

export function profileNameHint(name: string, copy: ProfileNameCopy): string {
  const trimmed = name.trim()

  if (hasProfileNameSyntax(trimmed) && isReservedProfileName(trimmed)) {
    return copy.reservedNameHint
  }

  return copy.nameHint
}
