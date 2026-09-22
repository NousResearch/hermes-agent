---
name: hermes-ops
description: Administra perfiles Hermes con CLI y confirmaciones.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, profiles, operations, configuration, agents, cli]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
    config: {}
---

# Hermes Ops Skill

Esta skill administra agentes Hermes como perfiles aislados, usando el CLI real y el
perfil `hermes-ops` como orquestador bajo demanda. No edita archivos de otros perfiles,
no crea un gateway persistente por defecto y no crea bots ni tokens de Telegram.

## When to Use

Úsala cuando el usuario pida crear, inspeccionar, configurar, equipar, renombrar o
eliminar un agente representado por un perfil Hermes. También aplica para cambiar su
modelo, proveedor, personalidad, prompt de sistema, alma (`SOUL.md`) o skills.

No la uses para modificar directamente el repositorio o los archivos internos de un
perfil, ni para instalar o iniciar gateways autónomos sin una petición explícita.

## Prerequisites

- El binario `hermes` debe estar disponible para `terminal`.
- El proceso orquestador debe ejecutarse como `hermes -p hermes-ops ...`.
- Antes de un grill-me sobre un agente conocido, consulta con `read_file` la nota
  `$HERMES_HOME/profiles/hermes-ops/notes/<nombre-agente>.md` si existe; por defecto
  corresponde a `~/.hermes/profiles/hermes-ops/notes/<nombre-agente>.md`.
- Usa `search_files` para localizar esa nota si no conoces el nombre exacto. `patch`
  solo puede actualizar la nota dentro del HOME propio de `hermes-ops`, nunca un perfil
  administrado. No necesitas `delegate_task` para una operación de perfiles.

## How to Run

Ejecuta el orquestador bajo demanda desde `terminal`, por ejemplo:

```text
hermes -p hermes-ops chat -q "Configura el agente research con el modelo solicitado"
```

Para una sesión interactiva, usa `hermes -p hermes-ops` y sigue el procedimiento de
esta skill. Cada comando dirigido a un agente debe incluir `-p NOMBRE`; el proceso
`hermes-ops` no sustituye ese selector. El gateway queda fuera del flujo normal y solo
se opera si el usuario pide explícitamente autonomía, cron o reacción persistente.

## Quick Reference

Todos los comandos de esta tabla se ejecutan desde `terminal`. Sustituye `NAME` por el
perfil administrado y conserva `-p NAME` en cada operación de configuración o skills.

| Objetivo | Comando |
| --- | --- |
| Crear perfil | `hermes profile create NAME [--clone \| --clone-from SOURCE \| --clone-all \| --no-skills]` |
| Listar perfiles | `hermes profile list` |
| Inspeccionar perfil | `hermes profile show NAME` / `hermes profile describe NAME` |
| Renombrar perfil | `hermes profile rename OLD NEW` |
| Eliminar perfil | `hermes profile delete NAME -y` |
| Leer configuración | `hermes -p NAME config get KEY` / `hermes -p NAME config show` |
| Cambiar configuración | `hermes -p NAME config set KEY VALUE` |
| Quitar configuración | `hermes -p NAME config unset KEY` |
| Editar configuración | `hermes -p NAME config edit` |
| Cambiar alma | `hermes -p NAME soul set TEXT` / `hermes -p NAME soul set --file PATH` |
| Instalar skill | `hermes -p NAME skills install IDENTIFIER [--category] [--name] [--force] [--yes]` |
| Habilitar/deshabilitar skill | `hermes -p NAME skills enable NAME [--platform P]` / `hermes -p NAME skills disable NAME [--platform P]` |
| Listar skills | `hermes -p NAME skills list [--source all\|hub\|builtin\|local] [--enabled-only]` |
| Gateway explícito | `hermes -p NAME gateway install\|start\|stop\|restart\|status` |

Claves de configuración habituales: `model.default`, `model.provider`,
`model.base_url`, `agent.system_prompt`, `display.personality` y
`agent.personalities.*`. `hermes -p NAME model` es un wizard interactivo: evítalo en
flujos no interactivos y prefiere `config set`.

## Procedure

1. **Clasifica la petición.** Identifica el nombre del perfil, la operación y si es
   creación o un cambio grande. Considera grande cualquier cambio de rol, personalidad,
   `agent.system_prompt`, modelo o proveedor. Agregar una skill puntual o ajustar un
   parámetro menor no requiere grill-me.

2. **Inspecciona por CLI.** Comprueba la existencia y el estado con `hermes profile list`,
   `hermes profile show NAME` o `hermes profile describe NAME`. Para configuración usa
   `hermes -p NAME config get KEY` y para skills usa `hermes -p NAME skills list`.
   Nunca uses `read_file`, `patch`, un editor o redirecciones de `terminal` para leer o
   cambiar `~/.hermes/profiles/<otro-perfil>/...`.

3. **Haz el grill-me obligatorio.** Para crear un agente nuevo o aplicar un cambio grande,
   lee primero la nota del agente si ya existe, pregunta por las decisiones faltantes y
   resume el diseño propuesto. No ejecutes el comando mutante hasta recibir confirmación
   explícita. Para una operación chica, continúa sin pedir confirmación adicional.

4. **Aplica mediante CLI.** Crea o clona con `hermes profile create`; configura con
   `hermes -p NAME config set` o `config unset`; reemplaza el alma con `soul set`; e
   instala o activa skills con los subcomandos `skills`. Cada mutación debe apuntar al
   perfil correcto mediante `-p NAME`. Si falta un subcomando para una necesidad,
   detente y señala el gap pendiente: no rompas el aislamiento editando archivos.

5. **Registra decisiones del grill-me.** Al cerrar un grill-me de creación o cambio
   grande, crea o actualiza únicamente
   `$HERMES_HOME/profiles/hermes-ops/notes/<nombre-agente>.md` con las decisiones
   vigentes: propósito, rol, modelo/proveedor, personalidad o prompt relevantes, skills
   habilitadas y cualquier pendiente. Esa nota pertenece al HOME propio de
   `hermes-ops`, no al perfil administrado.

6. **Verifica y comunica.** Relee el estado por CLI, confirma los valores relevantes y
   reporta lo aplicado, lo omitido y cualquier gap. Para `soul set`, verifica el código
   de salida y el estado del perfil; no inventes un comando de lectura que no exista.

## Pitfalls

- **Editar el perfil directamente:** no abras ni modifiques `config.yaml`, `SOUL.md`,
  skills, bases de datos o cualquier ruta bajo `~/.hermes/profiles/<otro-perfil>/` con
  herramientas de archivos. La única excepción de notas es el archivo del propio
  `hermes-ops` indicado arriba.
- **Saltarse el grill-me:** crear un agente o cambiar rol, personalidad, prompt de
  sistema, modelo o proveedor siempre exige la conversación y confirmación previas.
- **Usar el wizard en automatización:** `hermes -p NAME model` puede bloquear esperando
  entradas; usa `config set` con claves explícitas.
- **Confundir el orquestador con el objetivo:** `-p hermes-ops` selecciona al
  administrador; las mutaciones llevan `-p NAME` del agente administrado.
- **Crear integraciones por agente:** existe un solo bot orquestador de Telegram; nunca
  generes bots, tokens o credenciales nuevas para un perfil.
- **Encender autonomía por defecto:** `gateway install/start/...` está fuera de alcance
  salvo petición explícita de reacción autónoma o cron.
- **Borrar sin intención clara:** antes de `profile delete NAME -y`, confirma el nombre,
  el alcance y la autorización del usuario; después verifica con `profile list`.

## Verification

Una operación está terminada cuando:

- `hermes profile list` muestra el perfil esperado y `show` o `describe` no reporta un
  estado inesperado.
- Los cambios de configuración se confirman con `hermes -p NAME config get KEY` o
  `config show`; las skills con `hermes -p NAME skills list --enabled-only`.
- Una creación o cambio grande tiene su nota vigente en el HOME propio de
  `hermes-ops`, posterior al grill-me confirmado.
- El informe final identifica el perfil afectado, los comandos aplicados y cualquier
  gap de CLI pendiente, sin afirmar que se editó directamente un archivo administrado.
