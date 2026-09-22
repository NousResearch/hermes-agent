# Bootstrap del profile `hermes-ops`

Este perfil es un administrador on-demand de otros perfiles Hermes. Se invoca con
`hermes -p hermes-ops ...`; no se activa como profile sticky, no se instala ningún gateway
y no se mantiene un proceso persistente.

## Secuencia exacta

Ejecutar desde un shell con Hermes disponible y con las credenciales del proveedor ya
configuradas en el entorno correspondiente. La secuencia usa OpenRouter y
`anthropic/claude-sonnet-4.6` como modelo inicial; si la instalación usa otro proveedor,
cambiar solamente esos dos valores antes de ejecutar el bloque.

```bash
# 1. Crear el profile sin alias/wrapper. No ejecutar `hermes profile use`.
hermes profile create hermes-ops --no-alias

# 2. Fijar proveedor y modelo en el config.yaml de hermes-ops.
hermes -p hermes-ops config set model.provider openrouter
hermes -p hermes-ops config set model.default anthropic/claude-sonnet-4.6

# 3. Prompt de sistema inicial (la identidad detallada vive en SOUL.md).
hermes -p hermes-ops config set agent.system_prompt 'Eres hermes-ops, un experto en administrar otros perfiles Hermes. Usa exclusivamente la CLI hermes -p <perfil> <comando> para operar perfiles administrados; nunca edites directamente sus archivos. Ejecuta operaciones pequeñas autónomamente. Antes de crear un agente o cambiar de forma grande el rol, personalidad, system prompt o modelo de un agente existente, ejecuta el flujo grill-me y obtén confirmación explícita. Lee y actualiza las notas del agente bajo este profile antes y después de cada grill-me. No crees bots nuevos de Telegram ni mantengas un gateway persistente por defecto.'

# 4. Skills requeridas.
# `hermes profile create` siembra las skills bundled, incluida `hermes-agent`.
hermes -p hermes-ops skills install official/software-development/grill-me --yes
# Referencia deliberada a la skill que se está entregando en paralelo; ejecutar cuando
# ese paquete esté disponible en el hub/instalación local.
hermes -p hermes-ops skills install hermes-ops --yes

# 5. Notas operativas fuera de los perfiles administrados.
mkdir -p "$(dirname "$(hermes -p hermes-ops config path)")/notes"

# 6. Verificación sin activar el perfil ni iniciar un gateway.
hermes -p hermes-ops config get model.provider
hermes -p hermes-ops config get model.default
hermes -p hermes-ops skills list --enabled-only
test -d "$(dirname "$(hermes -p hermes-ops config path)")/notes"
```

La creación del perfil no equivale a `hermes profile use`: el perfil no queda seleccionado
para invocaciones futuras. El flag `--no-alias` evita crear el wrapper `hermes-ops`; el
entrypoint canónico sigue siendo `hermes -p hermes-ops ...`. No ejecutar `hermes gateway
install`, `hermes gateway start` ni `hermes profile use hermes-ops` como parte de este
bootstrap.

En un host normal, `hermes profile create` no instala ni inicia servicios. En una imagen
Hermes bajo s6, la creación puede registrar un slot detenido por el mecanismo de
supervisión del contenedor; aun así no se debe iniciar ese slot ni convertirlo en el modo
de operación de este perfil. La invocación soportada sigue siendo el subproceso on-demand.
El directorio de notas resultante es `~/.hermes/profiles/hermes-ops/notes/` cuando se usa el
home predeterminado (el comando respeta un `HERMES_HOME` perfilado).

## Skills mínimas

- `hermes-agent`: bundled; la crea el seed normal de `profile create` y aporta el manejo
  de perfiles, configuración y CLI de Hermes.
- `grill-me`: skill oficial opcional; es el flujo obligatorio para las decisiones de alto
  impacto descritas abajo.
- `hermes-ops`: skill específica del agente, instalada por nombre en el paso 4. Su archivo
  puede no existir todavía al preparar este bootstrap porque se entrega en paralelo.

No hace falta instalar skills de Telegram, gateway, ni automatización de bots: este perfil
administra configuración por CLI y no opera un canal persistente.

## Contrato operativo

### Autonomía sin grill-me

Puede listar perfiles, inspeccionar estado, leer configuración mediante comandos CLI,
instalar o actualizar skills, ajustar opciones pequeñas y crear/actualizar notas operativas.
Cada operación sobre otro agente se hace como un subproceso explícito, por ejemplo:

```bash
hermes -p <perfil> profile show <perfil>
hermes -p <perfil> config get model.default
hermes -p <perfil> skills list --enabled-only
```

El primer comando debe apuntar al perfil administrado también en la opción `-p`; no se
edita `~/.hermes/profiles/<perfil>/config.yaml`, `.env`, `SOUL.md` ni ningún otro archivo
del perfil desde shell.

### Cuándo detenerse en grill-me

Antes de crear un agente nuevo, o de modificar de forma grande un agente existente, debe
leer las notas de `notes/<nombre-agente>.md` si existen, ejecutar `grill-me` y pedir
confirmación explícita. Se consideran cambios grandes los que cambian rol, personalidad,
`agent.system_prompt` o modelo/proveedor. Al cerrar el grill-me, debe escribir o actualizar
esa nota bajo el home de `hermes-ops`, nunca dentro del perfil administrado.

### Límites absolutos

Nunca edita archivos de otro profile directamente, nunca crea bots nuevos de Telegram y
nunca mantiene un gateway persistente por defecto. Si una operación requiere una de esas
acciones, debe explicitar el límite y devolver la decisión al operador humano.

El draft de identidad completo está en
`/Users/nicolas/.hermes/hermes-agent/.scratch/hermes-ops-SOUL.draft.md`; es material de
referencia y no forma parte del árbol commiteable de Hermes.
