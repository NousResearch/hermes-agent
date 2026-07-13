# Contribuir a Hermes Agent

¡Gracias por contribuir a Hermes Agent! Esta guía cubre todo lo que necesitas: configurar tu entorno de desarrollo, entender la arquitectura, decidir qué construir y conseguir que tu PR sea aceptado.

---

## Prioridades de Contribución

Valoramos las contribuciones en este orden:

1. **Correcciones de errores** — bloqueos, comportamiento incorrecto, pérdida de datos. Siempre la máxima prioridad.
2. **Compatibilidad entre plataformas** — macOS, diferentes distribuciones de Linux y WSL2 en Windows. Queremos que Hermes funcione en todas partes.
3. **Fortalecimiento de seguridad** — inyección de shell, inyección de prompts, traversal de rutas, escalada de privilegios. Ver [Consideraciones de Seguridad](#consideraciones-de-seguridad).
4. **Rendimiento y robustez** — lógica de reintento, manejo de errores, degradación elegante.
5. **Nuevas habilidades** — pero solo las ampliamente útiles. Ver [¿Debería ser una Habilidad o una Herramienta?](#debería-ser-una-habilidad-o-una-herramienta)
6. **Nuevas herramientas** — raramente necesarias. La mayoría de las capacidades deberían ser habilidades. Ver más abajo.
7. **Documentación** — correcciones, aclaraciones, nuevos ejemplos.

---

## ¿Debería ser una Habilidad o una Herramienta?

Esta es la pregunta más común para los nuevos colaboradores. La respuesta casi siempre es **habilidad**.

### Hazlo una Habilidad cuando:

- La capacidad se puede expresar como instrucciones + comandos de shell + herramientas existentes
- Envuelve una CLI externa o API que el agente puede llamar a través de `terminal` o `web_extract`
- No necesita integración personalizada de Python ni gestión de claves API integrada en el agente
- Ejemplos: búsqueda en arXiv, flujos de trabajo de git, gestión de Docker, procesamiento de PDF, email a través de herramientas CLI

### Hazlo una Herramienta cuando:

- Requiere integración de extremo a extremo con claves API, flujos de autenticación o configuración de múltiples componentes gestionada por el harness del agente
- Necesita lógica de procesamiento personalizada que debe ejecutarse con precisión en cada ocasión (no "mejor esfuerzo" de la interpretación del LLM)
- Maneja datos binarios, streaming o eventos en tiempo real que no pueden pasar por el terminal
- Ejemplos: automatización de navegador (gestión de sesiones Browserbase), TTS (codificación de audio + entrega en plataforma), análisis de visión (manejo de imágenes base64)

### ¿Debería la Habilidad estar incluida?

Las habilidades incluidas (en `skills/`) se envían con cada instalación de Hermes. Deben ser **ampliamente útiles para la mayoría de los usuarios**:

- Manejo de documentos, investigación web, flujos de trabajo de desarrollo comunes, administración de sistemas
- Usadas regularmente por una amplia gama de personas

Si tu habilidad es oficial y útil pero no universalmente necesaria (ej., una integración de servicio de pago, una dependencia pesada), ponla en **`optional-skills/`** — se envía con el repositorio pero no está activada por defecto. Los usuarios pueden descubrirla a través de `hermes skills browse` (etiquetada como "oficial") e instalarla con `hermes skills install` (sin advertencia de terceros, confianza integrada).

Si tu habilidad es especializada, contribuida por la comunidad o de nicho, es mejor para un **Skills Hub** — súbela a un registro de habilidades y compártela en el [Discord de Nous Research](https://discord.gg/NousResearch). Los usuarios pueden instalarla con `hermes skills install`.

---

## Proveedores de Memoria: Publicar como Plugin Independiente

**Ya no aceptamos nuevos proveedores de memoria en este repositorio.** El conjunto de proveedores integrados en `plugins/memory/` (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb) está cerrado. Si quieres añadir un nuevo backend de memoria, publícalo como un **repositorio de plugin independiente** que los usuarios instalen en `~/.hermes/plugins/` (o a través de un entry point de pip).

Los plugins de memoria independientes:

- Implementan el mismo ABC `MemoryProvider` (`agent/memory_provider.py`) — `sync_turn`, `prefetch`, `shutdown` y opcionalmente `post_setup(hermes_home, config)` para integración con el asistente de configuración
- Usan el mismo sistema de descubrimiento — `discover_memory_providers()` los recoge desde directorios de plugins de usuario/proyecto y entry points de pip
- Se integran con `hermes memory setup` a través de `post_setup()` — sin necesidad de tocar el código base
- Pueden registrar sus propios subcomandos CLI a través de `register_cli(subparser)` en un archivo `cli.py`
- Obtienen todos los mismos hooks de ciclo de vida y plomería de configuración que los proveedores incluidos en el árbol

Los PRs que añadan un nuevo directorio bajo `plugins/memory/` serán cerrados con un puntero para publicar el proveedor como su propio repositorio. Los proveedores en árbol existentes se mantienen; las correcciones de errores para ellos son bienvenidas.

Esto no es una barra de calidad — es una decisión de acoplamiento y mantenimiento. Los proveedores de memoria son el tipo de plugin más común y no deberían vivir todos en este árbol.

---

## Configuración del Desarrollo

### Prerequisitos

| Requisito | Notas |
|-----------|-------|
| **Git** | Con la extensión `git-lfs` instalada |
| **Python 3.11–3.13** | uv lo instalará si falta |
| **uv** | Gestor de paquetes Python rápido ([instalar](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | Opcional — necesario para herramientas de navegador y puente WhatsApp (coincide con los engines de `package.json` raíz) |

### Clonar e instalar

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Crear venv con Python 3.11
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# Instalar con todos los extras (mensajería, cron, menús CLI, herramientas de desarrollo)
uv pip install -e ".[all,dev]"

# Opcional: herramientas de navegador
npm install
```

### Configurar para desarrollo

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Añadir al menos una clave de proveedor LLM:
echo "OPENROUTER_API_KEY=***" >> ~/.hermes/.env
```

### Ejecutar

```bash
# Enlace simbólico para acceso global
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# Verificar
hermes doctor
hermes chat -q "Hola"
```

### Ejecutar tests

```bash
# Preferido — coincide con CI (entorno hermético, 4 workers xdist); ver AGENTS.md
scripts/run_tests.sh

# Alternativa (activa el venv primero). El wrapper sigue recomendándose
# para paridad con GitHub Actions antes de abrir un PR:
pytest tests/ -v
```

---

## Estructura del Proyecto

```
hermes-agent/
├── run_agent.py              # Clase AIAgent — bucle de conversación central, despacho de herramientas, persistencia de sesión
├── cli.py                    # Clase HermesCLI — TUI interactiva, integración prompt_toolkit
├── model_tools.py            # Orquestación de herramientas (capa delgada sobre tools/registry.py)
├── toolsets.py               # Agrupaciones y presets de herramientas (hermes-cli, hermes-telegram, etc.)
├── hermes_state.py           # Base de datos de sesiones SQLite con búsqueda de texto completo FTS5, títulos de sesión
├── batch_runner.py           # Procesamiento en lote paralelo para generación de trayectorias
│
├── agent/                    # Internos del agente (módulos extraídos)
│   ├── prompt_builder.py         # Ensamblaje del prompt del sistema (identidad, habilidades, archivos de contexto, memoria)
│   ├── context_compressor.py     # Auto-resumición al acercarse a los límites de contexto
│   ├── auxiliary_client.py       # Resuelve clientes OpenAI auxiliares (resumición, visión)
│   ├── display.py                # KawaiiSpinner, formateo del progreso de herramientas
│   ├── model_metadata.py         # Longitudes de contexto del modelo, estimación de tokens
│   └── trajectory.py             # Ayudantes para guardar trayectorias
│
├── hermes_cli/               # Implementaciones de comandos CLI
│   ├── main.py                   # Punto de entrada, análisis de argumentos, despacho de comandos
│   ├── config.py                 # Gestión de configuración, migración, definiciones de variables de entorno
│   ├── setup.py                  # Asistente de configuración interactivo
│   ├── auth.py                   # Resolución de proveedor, OAuth, Nous Portal
│   ├── models.py                 # Listas de selección de modelos de OpenRouter
│   ├── banner.py                 # Banner de bienvenida, arte ASCII
│   ├── commands.py               # Registro central de comandos de barra (CommandDef), autocompletado, ayudantes del gateway
│   ├── callbacks.py              # Callbacks interactivos (aclarar, sudo, aprobación)
│   ├── doctor.py                 # Diagnósticos
│   ├── skills_hub.py             # CLI del Skills Hub + comando de barra /skills
│   └── skin_engine.py            # Motor de skins/temas — personalización visual de CLI basada en datos
│
├── tools/                    # Implementaciones de herramientas (auto-registradas)
│   ├── registry.py               # Registro central de herramientas (esquemas, manejadores, despacho)
│   ├── approval.py               # Detección de comandos peligrosos + aprobación por sesión
│   ├── terminal_tool.py          # Orquestación del terminal (sudo, ciclo de vida del entorno, backends)
│   ├── file_operations.py        # read_file, write_file, búsqueda, patch, etc.
│   ├── web_tools.py              # web_search, web_extract (Paralelo/Firecrawl + resumición Gemini)
│   ├── vision_tools.py           # Análisis de imágenes a través de modelos multimodales
│   ├── delegate_tool.py          # Lanzamiento de subagentes y ejecución paralela de tareas
│   ├── code_execution_tool.py    # Python sandboxado con acceso a herramientas vía RPC
│   ├── session_search_tool.py    # Búsqueda en conversaciones pasadas con FTS5 + ventanas ancladas
│   ├── cronjob_tools.py          # Gestión de tareas programadas
│   ├── skill_tools.py            # Búsqueda, carga y gestión de habilidades
│   └── environments/             # Backends de ejecución del terminal
│       ├── base.py                   # ABC BaseEnvironment
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py, daytona.py
│
├── gateway/                  # Gateway de mensajería
│   ├── run.py                    # GatewayRunner — ciclo de vida de plataformas, enrutamiento de mensajes, cron
│   ├── config.py                 # Resolución de configuración de plataformas
│   ├── session.py                # Almacén de sesiones, prompts de contexto, políticas de reset
│   └── platforms/                # Adaptadores de plataformas
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Scripts del instalador y puente
│   ├── install.sh                # Instalador Linux/macOS
│   ├── install.ps1               # Instalador Windows PowerShell
│   └── whatsapp-bridge/          # Puente WhatsApp Node.js (Baileys)
│
├── skills/                   # Habilidades incluidas (copiadas a ~/.hermes/skills/ en la instalación)
├── optional-skills/          # Habilidades opcionales oficiales (descubribles vía hub, no activadas por defecto)
├── tests/                    # Suite de tests
├── website/                  # Sitio de documentación (hermes-agent.nousresearch.com)
│
├── cli-config.yaml.example   # Configuración de ejemplo (copiada a ~/.hermes/config.yaml)
└── AGENTS.md                 # Guía de desarrollo para asistentes de codificación IA
```

### Configuración del usuario (almacenada en `~/.hermes/`)

| Ruta | Propósito |
|------|-----------|
| `~/.hermes/config.yaml` | Configuración (modelo, terminal, toolsets, compresión, etc.) |
| `~/.hermes/.env` | Claves API y secretos |
| `~/.hermes/auth.json` | Credenciales OAuth (Nous Portal) |
| `~/.hermes/skills/` | Todas las habilidades activas (incluidas + instaladas desde hub + creadas por el agente) |
| `~/.hermes/memories/` | Memoria persistente (MEMORY.md, USER.md) |
| `~/.hermes/state.db` | Base de datos de sesiones SQLite |
| `~/.hermes/sessions/` | Índice de enrutamiento del gateway (`sessions.json`), migas de pan de solicitudes, transcripciones `*.jsonl` del gateway y (opcionalmente) snapshots JSON por sesión cuando `sessions.write_json_snapshots: true` está configurado. Los snapshots por sesión están desactivados por defecto; state.db es canónica. |
| `~/.hermes/cron/` | Datos de trabajos programados |
| `~/.hermes/whatsapp/session/` | Credenciales del puente WhatsApp |

---

## Descripción General de la Arquitectura

### Bucle Central

```
Mensaje del usuario → AIAgent._run_agent_loop()
  ├── Construir prompt del sistema (prompt_builder.py)
  ├── Construir kwargs de API (modelo, mensajes, herramientas, configuración de razonamiento)
  ├── Llamar al LLM (API compatible con OpenAI)
  ├── Si tool_calls en la respuesta:
  │     ├── Ejecutar cada herramienta a través del despacho del registro
  │     ├── Añadir resultados de herramientas a la conversación
  │     └── Volver a la llamada al LLM
  ├── Si respuesta de texto:
  │     ├── Persistir sesión en DB
  │     └── Devolver final_response
  └── Compresión de contexto si se acerca al límite de tokens
```

### Patrones de Diseño Clave

- **Herramientas auto-registradas**: Cada archivo de herramienta llama a `registry.register()` en el momento de importación. `model_tools.py` activa el descubrimiento importando todos los módulos de herramientas.
- **Agrupación en toolsets**: Las herramientas se agrupan en toolsets (`web`, `terminal`, `file`, `browser`, etc.) que pueden habilitarse/deshabilitarse por plataforma.
- **Persistencia de sesión**: Todas las conversaciones se almacenan en SQLite (`hermes_state.py`) con búsqueda de texto completo y títulos de sesión únicos.
- **Inyección efímera**: Los prompts del sistema y los mensajes de relleno se inyectan en el momento de la llamada API, nunca se persisten en la base de datos ni en los logs.
- **Abstracción de proveedor**: El agente funciona con cualquier API compatible con OpenAI. La resolución del proveedor ocurre en el momento de la inicialización.
- **Enrutamiento de proveedor**: Al usar OpenRouter, `provider_routing` en config.yaml controla la selección del proveedor.

---

## Estilo de Código

- **PEP 8** con excepciones prácticas (no imponemos longitud de línea estricta)
- **Comentarios**: Solo cuando se explica la intención no obvia, compromisos o peculiaridades de API. No narres lo que hace el código
- **Manejo de errores**: Captura excepciones específicas. Registra con `logger.warning()`/`logger.error()` — usa `exc_info=True` para errores inesperados
- **Multiplataforma**: Nunca asumas Unix. Ver [Compatibilidad Multiplataforma](#compatibilidad-multiplataforma)

---

## Añadir una Nueva Herramienta

Antes de escribir una herramienta, pregúntate: [¿debería ser una habilidad en su lugar?](#debería-ser-una-habilidad-o-una-herramienta)

Las herramientas se auto-registran en el registro central. Cada archivo de herramienta co-localiza su esquema, manejador y registro:

```python
"""my_tool — Breve descripción de lo que hace esta herramienta."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """Manejador. Devuelve un resultado en cadena (a menudo JSON)."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Qué hace esta herramienta y cuándo debería usarla el agente.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Qué es param1"},
                "param2": {"type": "integer", "description": "Qué es param2", "default": 10},
            },
            "required": ["param1"],
        },
    },
}


def _check_requirements() -> bool:
    """Devuelve True si las dependencias de esta herramienta están disponibles."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Conectar a un toolset (requerido):** Las herramientas integradas se auto-descubren: cualquier
archivo `tools/*.py` que contenga una llamada de nivel superior `registry.register(...)` es
importado por `discover_builtin_tools()` en `tools/registry.py` cuando `model_tools`
se carga. **No** hay una lista de importaciones manual en `model_tools.py` que mantener.

Todavía debes añadir el nombre de la herramienta a la lista apropiada en `toolsets.py`
(por ejemplo `_HERMES_CORE_TOOLS` o un toolset dedicado); de lo contrario la herramienta
se registra pero nunca se expone al agente.

Consulta `AGENTS.md` (sección **Adding New Tools**) para rutas conscientes del perfil y
orientación sobre plugins vs. núcleo.

---

## Añadir una Habilidad

Las habilidades incluidas viven en `skills/` organizadas por categoría. Las habilidades opcionales oficiales usan la misma estructura en `optional-skills/`:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Requerido: instrucciones principales
│       └── scripts/              # Opcional: scripts auxiliares
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### Formato de SKILL.md

```markdown
---
name: my-skill
description: Breve descripción (mostrada en los resultados de búsqueda de habilidades)
version: 1.0.0
author: Tu Nombre
license: MIT
platforms: [macos, linux]          # Opcional — restringir a plataformas de SO específicas
required_environment_variables:    # Opcional — metadatos de configuración segura al cargar
  - name: MY_API_KEY
    prompt: Clave API
    help: Dónde obtenerla
    required_for: funcionalidad completa
prerequisites:                     # Requisitos de tiempo de ejecución heredados opcionales
  env_vars: [MY_API_KEY]
  commands: [curl, jq]
metadata:
  hermes:
    tags: [Categoría, Subcategoría, Palabras clave]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]
    requires_toolsets: [terminal]
---

# Título de la Habilidad

Introducción breve.

## Cuándo Usar
Condiciones de activación — ¿cuándo debería el agente cargar esta habilidad?

## Referencia Rápida
Tabla de comandos o llamadas API comunes.

## Procedimiento
Instrucciones paso a paso que el agente sigue.

## Problemas Conocidos
Modos de fallo conocidos y cómo manejarlos.

## Verificación
Cómo confirma el agente que funcionó.
```

### Estándares de autoría de habilidades (OBLIGATORIOS)

Todo skill nuevo o modernizado — incluido, opcional o contribuido — debe cumplir estos estándares antes del merge:

1. **`description` ≤ 60 caracteres, una oración, termina con punto.** Las descripciones largas saturan la UI de listado de habilidades. Indica la capacidad, no la implementación. Sin palabras de marketing ("potente", "completo", "fluido", "avanzado").

2. **Las herramientas referenciadas en el cuerpo de SKILL.md deben ser herramientas nativas de Hermes o servidores MCP que la habilidad espere explícitamente.** Usa los nombres de herramientas en comillas invertidas: `` `terminal` ``, `` `web_extract` ``, `` `web_search` ``, `` `read_file` ``, `` `write_file` ``, etc.

3. **El campo `platforms:` auditado contra las importaciones reales del script.** Las habilidades que usen primitivos solo de POSIX deben declarar sus plataformas soportadas.

4. **`author` da crédito primero al colaborador humano.**

5. **El cuerpo de SKILL.md usa el orden moderno de secciones:** título, intro de 2-3 oraciones, luego: `## Cuándo Usar`, `## Prerequisitos`, `## Cómo Ejecutar`, `## Referencia Rápida`, `## Procedimiento`, `## Problemas Conocidos`, `## Verificación`.

6. **Los scripts van en `scripts/`, las referencias en `references/`, las plantillas en `templates/`.**

7. **Los tests viven en `tests/skills/test_<skill>_skill.py`** y usan solo stdlib + pytest + `unittest.mock`. Sin llamadas de red en vivo.

8. **Las adiciones a `.env.example` están aisladas en un bloque claramente delimitado.**

---

## Añadir una Skin / Tema

Hermes usa un sistema de skins basado en datos — no se necesitan cambios de código para añadir una nueva skin.

**Opción A: Skin de usuario (archivo YAML)**

Crea `~/.hermes/skins/<nombre>.yaml`:

```yaml
name: mitema
description: Breve descripción del tema

colors:
  banner_border: "#HEX"
  banner_title: "#HEX"
  banner_accent: "#HEX"
  banner_dim: "#HEX"
  banner_text: "#HEX"
  response_border: "#HEX"

spinner:
  waiting_faces: ["(⚔)", "(⛨)"]
  thinking_faces: ["(⚔)", "(⌁)"]
  thinking_verbs: ["forjando", "planeando"]

branding:
  agent_name: "Mi Agente"
  welcome: "Mensaje de bienvenida"
  response_label: " ⚔ Agente "
  prompt_symbol: "⚔"

tool_prefix: "╎"
```

Todos los campos son opcionales — los valores faltantes se heredan de la skin predeterminada.

**Opción B: Skin integrada**

Añade al dict `_BUILTIN_SKINS` en `hermes_cli/skin_engine.py`. Usa el mismo esquema que arriba pero como dict de Python.

**Activar:**
- CLI: `/skin mitema` o establece `display.skin: mitema` en config.yaml

---

## Compatibilidad Multiplataforma

Hermes se ejecuta en Linux, macOS y Windows nativo (además de WSL2). Al escribir código
que toca el SO, asume que *cualquier* plataforma puede alcanzar tu ruta de código.

> **Antes de hacer PR:** ejecuta `scripts/check-windows-footguns.py` para detectar
> los patrones inseguros comunes de Windows en tu diff. Es basado en grep y barato;
> CI también lo ejecuta en cada PR.

### Reglas críticas

1. **Nunca llames `os.kill(pid, 0)` para comprobaciones de liveness.** En Windows **NO es una operación sin efecto**. Usa `psutil.pid_exists(pid)` en su lugar.

2. **Usa `shutil.which()` antes de hacer shell — no asumas que Windows tiene las herramientas que tiene Linux.** `ps`, `kill`, `grep`, `awk`, etc. simplemente no existen en Windows.

3. **`termios` y `fcntl` son solo de Unix.** Siempre captura tanto `ImportError` como `NotImplementedError`.

4. **Codificación de archivos.** Windows puede guardar archivos `.env` en `cp1252`. Siempre maneja errores de codificación.

5. **Gestión de procesos.** `os.setsid()`, `os.killpg()`, `os.fork()`, `os.getuid()` y el manejo de señales POSIX difieren en Windows.

6. **Señales que no existen en Windows:** `SIGALRM`, `SIGCHLD`, `SIGHUP`, `SIGUSR1`, `SIGUSR2`, etc.

7. **Separadores de ruta.** Usa `pathlib.Path` en lugar de concatenación de cadenas con `/`.

8. **Los enlaces simbólicos necesitan privilegios elevados en Windows** (a menos que el Modo Desarrollador esté activado).

9. **Los modos de archivo POSIX (0o600, 0o644, etc.) NO se aplican en NTFS** por defecto.

10. **Los daemons de fondo desacoplados en Windows necesitan `pythonw.exe`, NO `python.exe`.**

---

## Consideraciones de Seguridad

Hermes tiene acceso al terminal. La seguridad importa.

### Protecciones existentes

| Capa | Implementación |
|------|---------------|
| **Piping de contraseña sudo** | Usa `shlex.quote()` para prevenir inyección de shell |
| **Detección de comandos peligrosos** | Patrones regex en `tools/approval.py` con flujo de aprobación del usuario |
| **Inyección de prompts en cron** | Escáner en `tools/cronjob_tools.py` bloquea patrones de anulación de instrucciones |
| **Lista de denegación de escritura** | Rutas protegidas resueltas a través de `os.path.realpath()` para prevenir bypass de enlaces simbólicos |
| **Skills Guard** | Escáner de seguridad para habilidades instaladas desde el hub (`tools/skills_guard.py`) |
| **Sandbox de ejecución de código** | El proceso hijo `execute_code` se ejecuta con claves API eliminadas del entorno |
| **Fortalecimiento de contenedor** | Docker: todas las capacidades eliminadas, sin escalada de privilegios, límites de PID, tmpfs de tamaño limitado |

### Al contribuir código sensible a la seguridad

- **Siempre usa `shlex.quote()`** al interpolar entrada del usuario en comandos de shell
- **Resuelve enlaces simbólicos** con `os.path.realpath()` antes de comprobaciones de control de acceso basadas en rutas
- **No registres secretos.** Las claves API, tokens y contraseñas nunca deben aparecer en la salida de log
- **Captura excepciones amplias** alrededor de la ejecución de herramientas para que un solo fallo no bloquee el bucle del agente
- **Prueba en todas las plataformas** si tu cambio toca rutas de archivos, gestión de procesos o comandos de shell

### Política de fijación de dependencias (fortalecimiento de la cadena de suministro)

Tras el [compromiso de la cadena de suministro de litellm](https://github.com/BerriAI/litellm/issues/24512) en marzo de 2026 y la [campaña del gusano Mini Shai-Hulud](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack) en mayo de 2026, todas las dependencias deben seguir estas reglas:

| Tipo de fuente | Tratamiento requerido | Justificación |
|---|---|---|
| **Paquete PyPI** | `>=suelo,<siguiente_mayor` | Las versiones de PyPI son inmutables una vez publicadas, pero pueden empujarse nuevas versiones en tu rango. |
| **URL de Git** | SHA completo del commit | Las ramas y etiquetas son refs mutables; el SHA está direccionado por contenido. |
| **GitHub Actions** | SHA completo del commit + comentario de versión | Las etiquetas de acción son refs mutables. Fija como `uses: owner/action@<sha>  # vX.Y.Z` |
| **Instalaciones pip solo de CI** | `==exacto` | Builds de CI herméticos; el cambio es aceptable. |

**Cada nueva dependencia de PyPI en un PR debe tener un límite superior `<siguiente_mayor`.** Los PRs que añadan especificaciones `>=X.Y.Z` sin límite superior serán rechazados.

---

## Proceso de Pull Request

### Nomenclatura de ramas

```
fix/descripcion        # Correcciones de errores
feat/descripcion       # Nuevas funcionalidades
docs/descripcion       # Documentación
test/descripcion       # Tests
refactor/descripcion   # Reestructuración de código
```

### Antes de enviar

1. **Ejecutar tests**: `scripts/run_tests.sh` (recomendado; igual que CI) o `pytest tests/ -v` con el venv del proyecto activado
2. **Probar manualmente**: Ejecuta `hermes` y ejercita la ruta de código que cambiaste
3. **Verificar impacto multiplataforma**: Si tocas E/S de archivos, gestión de procesos o manejo del terminal, considera macOS, Linux y WSL2
4. **Mantén los PRs enfocados**: Un cambio lógico por PR. No mezcles una corrección de error con una refactorización con una nueva funcionalidad.

### Descripción del PR

Incluye:
- **Qué** cambió y **por qué**
- **Cómo probarlo** (pasos de reproducción para errores, ejemplos de uso para funcionalidades)
- **Qué plataformas** probaste
- Referencia cualquier issue relacionado

### Mensajes de commit

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>(<alcance>): <descripción>
```

| Tipo | Usar para |
|------|-----------|
| `fix` | Correcciones de errores |
| `feat` | Nuevas funcionalidades |
| `docs` | Documentación |
| `test` | Tests |
| `refactor` | Reestructuración de código (sin cambio de comportamiento) |
| `chore` | Build, CI, actualizaciones de dependencias |

Alcances: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, etc.

Ejemplos:
```
fix(cli): prevenir bloqueo en save_config_value cuando el modelo es una cadena
feat(gateway): añadir aislamiento de sesión multi-usuario de WhatsApp
fix(security): prevenir inyección de shell en el piping de contraseña sudo
test(tools): añadir tests unitarios para file_operations
```

---

## Reportar Issues

- Usa [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- Incluye: SO, versión de Python, versión de Hermes (`hermes version`), traza de error completa
- Incluye pasos para reproducir
- Verifica los issues existentes antes de crear duplicados
- Para vulnerabilidades de seguridad, por favor reporta de forma privada

---

## Comunidad

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch) — para preguntas, mostrar proyectos y compartir habilidades
- **GitHub Discussions**: Para propuestas de diseño y discusiones de arquitectura
- **Skills Hub**: Sube habilidades especializadas a un registro y compártelas con la comunidad

---

## Licencia

Al contribuir, aceptas que tus contribuciones serán licenciadas bajo la [Licencia MIT](LICENSE).
