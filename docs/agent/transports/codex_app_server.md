# Codex app-server transport

`CodexAppServerClient` launches `codex app-server` with a sanitized child
environment while retaining the credentials required by the model provider.
It keeps the user's normal `HOME` unchanged and pins Codex state through
`CODEX_HOME`.

The state-directory precedence is:

1. The explicit `codex_home` constructor argument.
2. `CODEX_HOME` in the effective child environment, including a caller's
   `env` overlay.
3. The default `~/.codex` directory.

The resolved path is normalized and exposed through `client.codex_home`. The
session handshake compares that path with the app-server's reported
`codexHome` and logs a warning when they differ.
