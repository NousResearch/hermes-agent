// Memory-source classification shared by the star map's UI + share codec.
// 'memory' (MEMORY.md) and 'profile' (USER.md) are Hermes-owned and mutable;
// anything else is the name of an external memory provider ('honcho', …)
// whose entries are read-only in the journey.
export const isProviderSource = (source?: null | string): source is string =>
  Boolean(source && source !== 'memory' && source !== 'profile')
