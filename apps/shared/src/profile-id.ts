// THE profile-id grammar for TypeScript: the mirror of Python's
// hermes_constants.PROFILE_ID_RE. Every Desktop (Electron + renderer), plugin
// SDK and dashboard gate reads it from here instead of re-declaring the
// literal (state-24, #155595). `default` and other aliases, trimming and
// case-folding stay with each caller.

export const PROFILE_ID_ARG_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/
