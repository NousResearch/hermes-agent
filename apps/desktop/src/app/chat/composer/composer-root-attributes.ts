export function composerRootAttributes(stacked: boolean) {
  return {
    'data-slot': 'composer-root',
    'data-composer-expanded': stacked ? '' : undefined
  } as const
}
