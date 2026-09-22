# Architecture Decision Records

## 2026-09-21: Agente `hermes-ops` para administrar profiles de Hermes desde Telegram

Status: Accepted

Context:
Se quiere que, desde el bot orquestador de Telegram, se puedan crear y mantener otros
agentes Hermes (perfiles persistentes en la M1) sin tocar la CLI a mano.

Decision:
- La unidad de "agente" es el **profile** nativo de Hermes (`~/.hermes/profiles/<nombre>`),
  no un concepto paralelo. Se reutiliza `--clone` y el aislamiento existente entre profiles.
- El experto vive en **su propio profile** (`hermes-ops`), invocado **on-demand** por el
  orquestador vía subproceso (`hermes -p hermes-ops ...`) — no corre un gateway propio 24/7
  por default. El orquestador mantiene un único bot/canal de Telegram; no se crean bots
  nuevos por cada agente (routing interno, no un token por agente).
- Los agentes que `hermes-ops` crea siguen el mismo patrón: persistentes en config/estado,
  pero **sin gateway propio corriendo** salvo que el usuario pida explícitamente algo que
  requiera reacción autónoma (cron/eventos), caso en el que sí se registra como servicio
  vía `LaunchdServiceManager` (`hermes_cli/service_manager.py`).
- Toda operación de `hermes-ops` sobre otro profile pasa **exclusivamente por CLI**
  (`hermes -p <perfil> <comando>`) — nunca edición directa de archivos de otro profile.
  Si falta un comando para algo puntual, eso es un gap de CLI a resolver como ticket propio,
  no una excusa para romper el aislamiento entre profiles.
- **Gate de confirmación:**
  - Operaciones simples (agregar una skill, ajustes menores de config) → autónomo, sin preguntar.
  - **Creación de un agente nuevo** → obligatorio pasar por una entrevista estilo `grill-me`
    antes de aplicar cualquier cambio.
  - **Cambios grandes** de comportamiento en un agente existente (rol, personalidad,
    system-prompt, modelo/proveedor) → también obligatorio `grill-me`.
  - Cambios menores en agentes existentes → sin gate.
- Las notas de contexto por agente viven **del lado de `hermes-ops`, no del perfil
  administrado**: `~/.hermes/profiles/hermes-ops/notes/<nombre-agente>.md`, escritas/
  actualizadas por `hermes-ops` al cerrar cada `grill-me`, con las decisiones de diseño
  vigentes (rol, tono, límites, integraciones). Antes de volver a interrogar sobre un agente
  existente, `hermes-ops` lee esa nota primero. Se descartó guardarlas dentro de la carpeta
  del perfil administrado (`~/.hermes/profiles/<otro>/AGENT_CONTEXT.md`) porque ese mecanismo
  no existe hoy en Hermes (requeriría discovery/inyección al prompt nuevos, tocando el core) y
  porque escribir ahí directo violaría la regla de "solo CLI, nunca archivo directo" sobre
  perfiles ajenos.

Alternativas descartadas:
- `hermes-ops` como skill del orquestador (sin profile propio): más simple pero mezcla la
  identidad "recepción de mensajes" con "operación de infraestructura"; se prefirió aislarlo.
- Un bot de Telegram por agente nuevo: agrega superficie de secrets (`.env`) por cada agente
  creado, sin necesidad real dado que la invocación es on-demand vía el orquestador.
- Acceso directo a archivos de otros profiles: rompe el aislamiento por diseño de profiles y
  no queda auditado como los comandos CLI.

Consequences:
- Requiere que la CLI de profiles/gateway cubra todo lo que `hermes-ops` necesita hacer;
  cualquier gap se convierte en un ticket de CLI, no en un atajo de archivos.
- El criterio "cambio grande vs chico" para disparar `grill-me` queda definido en la skill de
  `hermes-ops`, no en este ADR — es una regla operativa que puede iterar sin nueva decisión.

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.
