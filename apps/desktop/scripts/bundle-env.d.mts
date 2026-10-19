export function validateBundleEnvironment(values: unknown): Record<string, string | null>

export function applyBundleEnvironment(
  env: Record<string, string | undefined>,
  values: Record<string, string | null>,
): Record<string, string | undefined>

export function environmentDefaultsBanner(raw: string): string
