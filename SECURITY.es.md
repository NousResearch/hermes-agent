# Política de Seguridad de Hermes Agent

Este documento describe el modelo de confianza de Hermes Agent, identifica el
único límite de seguridad que el proyecto trata como estructural y define el
alcance para los informes de vulnerabilidades.

## 1. Reportar una Vulnerabilidad

Reporta de forma privada a través de [GitHub Security Advisories](https://github.com/NousResearch/hermes-agent/security/advisories/new)
o **security@nousresearch.com**. No abras issues públicos para
vulnerabilidades de seguridad. **Hermes Agent no opera un programa de
recompensas por errores.**

Un informe útil incluye:

- Una descripción concisa y evaluación de severidad.
- El componente afectado, identificado por ruta de archivo y rango de líneas
  (ej. `path/to/file.py:120-145`).
- Detalles del entorno (`hermes version`, SHA del commit, SO, versión de Python).
- Una reproducción contra `main` o el último release.
- Una declaración de qué límite de confianza del §2 se cruza.

Por favor lee el §2 y el §3 antes de enviar. Los informes que demuestren
límites de una heurística en proceso que esta política no trate como un
límite serán cerrados como fuera de alcance bajo el §3 — pero consulta el §3.2:
siguen siendo bienvenidos como issues o pull requests regulares, simplemente no
a través del canal de seguridad privado.

---

## 2. Modelo de Confianza

Hermes Agent es un agente personal de un solo inquilino. Su postura es
por capas, y las capas no tienen el mismo peso. Los reportadores y
operadores deben razonar sobre ellas en los mismos términos.

### 2.1 Definiciones

- **Proceso del agente.** El intérprete Python que ejecuta Hermes Agent,
  incluyendo cualquier módulo Python que haya cargado (habilidades, plugins,
  manejadores de hooks).
- **Backend de terminal.** Un objetivo de ejecución conectado para la
  herramienta `terminal()`. El predeterminado ejecuta comandos directamente en el host.
  Otros backends ejecutan comandos dentro de un contenedor, sandbox en la nube o
  host remoto.
- **Superficie de entrada.** Cualquier canal a través del cual el contenido entra en el
  contexto del agente: entrada del operador, fetches web, email, mensajes del gateway,
  lecturas de archivos, respuestas del servidor MCP, resultados de herramientas.
- **Envolvente de confianza.** El conjunto de recursos a los que un operador ha otorgado
  implícitamente acceso a Hermes Agent al ejecutarlo — típicamente, todo lo que
  la propia cuenta de usuario del operador puede alcanzar en el host.
- **Postura.** Una declaración explícita en la documentación o código de Hermes Agent
  sobre cómo una capa consumidora (adaptador, UI, escritor de archivos,
  shell) debe tratar la salida del agente — ej. "el dashboard renderiza
  la salida del agente como HTML inerte."

### 2.2 El Límite: Aislamiento a Nivel de SO

**El único límite de seguridad contra un LLM adversario es el
sistema operativo.** Nada dentro del proceso del agente constituye
contención — ni la puerta de aprobación, ni la redacción de salida, ni ningún
escáner de patrones, ni ninguna lista de herramientas permitidas. Cualquier componente dentro
del proceso que filtre la salida del LLM es una heurística operando sobre una
cadena influenciada por el atacante, y esta política lo trata como tal.

Hermes Agent admite dos posturas de aislamiento a nivel de SO. Abordan
diferentes amenazas y un operador debe elegir deliberadamente.

#### Aislamiento del backend de terminal

Un backend de terminal no predeterminado ejecuta comandos de shell emitidos por el LLM
dentro de un contenedor, host remoto o sandbox en la nube. Las herramientas de archivos
(`read_file`, `write_file`, `patch`) también se ejecutan a través de este backend,
ya que están implementadas sobre el contrato del shell — no pueden
alcanzar rutas que el backend no exponga.

Lo que confina: todo lo que el agente hace emitiendo operaciones de shell o
de archivos. Lo que **no** confina: todo lo que el agente hace en su propio
proceso Python. Eso incluye la herramienta de ejecución de código (lanzada como
subproceso del host), subprocesos MCP (lanzados desde el entorno del agente),
carga de plugins, despacho de hooks y carga de habilidades (todos importados en el
intérprete del agente).

El aislamiento del backend de terminal es la postura correcta cuando la preocupación es
que el LLM emita comandos de shell destructivos o escrituras de herramientas de archivo no deseadas, y el
operador es de confianza.

#### Envoltura del proceso completo

La envoltura del proceso completo ejecuta todo el árbol de procesos del agente dentro de un
sandbox. Cada ruta de código — shell, ejecución de código, MCP, herramientas de archivos,
plugins, hooks, carga de habilidades — está sujeta a la misma política de sistema de archivos,
red, proceso e (donde sea aplicable) inferencia.

Hermes Agent admite esto de dos maneras:

- **La propia imagen Docker de Hermes Agent y la configuración de Compose.** Más
  liviana; el agente se ejecuta en un contenedor estándar con montajes y
  política de red configurados por el operador.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)**.
  OpenShell proporciona sandboxes por sesión con política declarativa
  a través de capas de sistema de archivos, red (egreso L7), proceso/syscall e
  enrutamiento de inferencia. Las políticas de red e inferencia son
  recargables en caliente. Las credenciales se inyectan desde un almacén de Proveedor
  y nunca tocan el sistema de archivos del sandbox.

Bajo una envoltura de proceso completo, las heurísticas en proceso de Hermes Agent
(§2.4) funcionan como prevención de accidentes en capas sobre un límite real.
Esta es la postura soportada cuando el agente ingiere contenido de superficies
que el operador no controla — la web abierta, email entrante, canales de
múltiples usuarios, servidores MCP no confiables — y para despliegues en
producción o compartidos.

Los operadores que ejecuten el backend local predeterminado con superficies de entrada
no confiables, o que ejecuten un sandbox de backend de terminal esperando que contenga
rutas de código que no pasan por el shell, están operando fuera de la postura de
seguridad soportada.

### 2.3 Alcance de Credenciales

Hermes Agent filtra el entorno que pasa a sus componentes en proceso de
menor confianza: subprocesos de shell, subprocesos MCP y el proceso hijo
de ejecución de código. Las credenciales como las claves API del proveedor y los
tokens del gateway se eliminan por defecto; las variables declaradas explícitamente
por el operador o por una habilidad cargada se pasan.

Esto reduce la exfiltración casual. No es contención. Cualquier
componente que se ejecute dentro del proceso del agente (habilidades, plugins, manejadores
de hooks) puede leer lo que el agente mismo puede leer, incluidas las
credenciales en memoria. La mitigación contra un componente en proceso comprometido
es la revisión del operador antes de instalar (§2.4, §2.5), no el
saneamiento del entorno.

### 2.4 Heurísticas en Proceso

Los siguientes componentes filtran o advierten sobre el comportamiento del LLM. Son
útiles. No son límites.

- La **puerta de aprobación** detecta patrones de shell destructivos comunes
  y le pide al operador confirmación antes de la ejecución. El shell es Turing-
  completo; una lista de denegación sobre cadenas de shell es estructuralmente
  incompleta. La puerta detecta errores en modo cooperativo, no salidas
  adversariales.
- **La redacción de salida** elimina patrones similares a secretos de la visualización.
  Un productor de salida motivado la evitará.
- **Skills Guard** escanea el contenido de habilidades instalables en busca de patrones
  de inyección. Es una ayuda de revisión; el límite para habilidades de terceros
  es la revisión del operador antes de instalar. Revisar una habilidad significa
  leer su código Python y scripts, no solo su descripción SKILL.md —
  las habilidades ejecutan Python arbitrario en el momento de importación.

### 2.5 Modelo de Confianza de Plugins

Los plugins se cargan en el proceso del agente y se ejecutan con todos los privilegios
del agente: pueden leer las mismas credenciales, llamar a las mismas
herramientas, registrar los mismos hooks e importar los mismos módulos que
cualquier cosa incluida en el árbol. El límite para los plugins de terceros es
la revisión del operador antes de instalar — la misma regla que las habilidades (§2.4),
mencionado por separado porque los plugins son arquitectónicamente más pesados
y a menudo incluyen sus propios servicios en segundo plano, oyentes de red
y dependencias.

Un plugin malicioso o con errores no es una vulnerabilidad en Hermes Agent
en sí mismo. Los errores en la ruta de instalación o descubrimiento de plugins de Hermes Agent
que impidan al operador ver lo que está instalando están en alcance bajo el §3.1.

### 2.6 Superficies Externas

Una **superficie externa** es cualquier canal fuera del proceso del agente local
a través del cual un llamador puede despachar trabajo del agente, resolver
aprobaciones o recibir salida del agente. Cada superficie tiene su propio
modelo de autorización, pero las reglas a continuación se aplican uniformemente.

**Superficies en Hermes Agent:**

- **Adaptadores de plataforma del gateway.** Integraciones de mensajería —
  principalmente `plugins/platforms/<name>/adapter.py`, más adaptadores legacy
  directos y tipos compartidos en `gateway/platforms/` (Telegram, Discord,
  Slack, email, SMS, etc. se distribuyen como plugins; `base.py` y algunas
  rutas legacy permanecen en el árbol).
- **Superficies HTTP expuestas en red.** El adaptador del servidor API, el
  plugin del dashboard, los endpoints HTTP del plugin kanban, y cualquier
  otro plugin que vincule un socket de escucha.
- **Adaptadores de Editor / IDE.** El adaptador ACP (`acp_adapter/`) e
  integraciones equivalentes que aceptan solicitudes de un proceso cliente local.
- **El gateway TUI (`tui_gateway/`).** Backend JSON-RPC para la
  UI de terminal Ink, alcanzado a través de IPC local.

**Reglas uniformes:**

1. **Se requiere autorización en cada superficie que cruce un límite de confianza.** Para
   superficies de mensajería y HTTP en red, el límite es la red: la autorización
   significa una lista de llamadores permitidos configurada por el operador. Para superficies
   de editor e IPC local (ACP, gateway TUI), el límite es la cuenta de usuario del host:
   la autorización significa depender del control de acceso a nivel de SO (permisos
   de archivos, vinculaciones solo a loopback) y no exponer la superficie más allá
   del usuario local sin una capa de autenticación de red explícita.
2. **Se requiere una lista de permitidos para cada adaptador de red habilitado.**
   Los adaptadores deben rechazar despachar trabajo del agente, resolver
   aprobaciones o transmitir salida hasta que se establezca una lista de permitidos. Las rutas
   de código que fallan de forma abierta cuando no hay lista de permitidos configurada son errores de código en
   alcance bajo el §3.1.
3. **Los identificadores de sesión son manejadores de enrutamiento, no límites de autorización.**
   Conocer el ID de sesión de otro llamador no otorga acceso a sus aprobaciones o salida;
   la autorización siempre se vuelve a verificar contra la lista de permitidos (o equivalente
   a nivel de SO).
4. **Dentro del conjunto autorizado, todos los llamadores tienen la misma confianza.**
   Hermes Agent no modela capacidades por llamador dentro de un único adaptador.
   Los operadores que necesiten separación de capacidades deben ejecutar instancias
   de agente separadas con listas de permitidos separadas.
5. **Vincular una superficie solo local a una interfaz no-loopback es una decisión de
   operador de emergencia (§3.2).** El dashboard y otros servidores HTTP de plugins
   son predeterminados a loopback; exponerlos a través de `--host 0.0.0.0` o equivalente
   hace que el fortalecimiento de exposición pública (§4) sea responsabilidad del operador.

---

## 3. Alcance

### 3.1 En Alcance

- Escape de una postura de aislamiento a nivel de SO declarada (§2.2): una
  ruta de código controlada por el atacante alcanzando estado que la postura
  afirmó confinar.
- Acceso no autorizado a superficie externa: un llamador fuera del conjunto de
  autorización configurado (lista de permitidos, o equivalente a nivel de SO
  para superficies de IPC local) despachando trabajo, recibiendo salida o
  resolviendo aprobaciones (§2.6).
- Exfiltración de credenciales: filtración de credenciales del operador o
  material de autorización de sesión a un destino fuera del envolvente de
  confianza, a través de un mecanismo que debería haberlo prevenido
  (error de saneamiento de entorno, registro del adaptador, error de transporte
  que vacía credenciales a un upstream, etc.).
- Violaciones de la documentación del modelo de confianza: código que se comporta
  contrariamente a lo que esta política, la propia documentación de Hermes Agent o
  las expectativas razonables del operador predecirían — incluyendo casos donde
  Hermes Agent ha documentado una postura sobre cómo su salida debe ser
  renderizada por una capa consumidora (dashboard, adaptador de gateway,
  escritor de archivos, shell) y una ruta de código rompe esa postura.

### 3.2 Fuera de Alcance

"Fuera de alcance" aquí significa "no es una vulnerabilidad de seguridad bajo esta
política." No significa "no vale la pena reportarlo." Las mejoras a las
heurísticas en proceso, ideas de fortalecimiento y correcciones de UX son bienvenidas como
issues o pull requests regulares — la puerta de aprobación siempre puede detectar
más patrones, la redacción puede volverse más inteligente, el comportamiento del adaptador
puede apretarse siempre. Estos elementos simplemente no van a través del canal de
divulgación privada y no reciben avisos.

- **Bypasses de heurísticas en proceso (§2.4)** — bypasses de regex de la puerta de aprobación,
  bypasses de redacción, bypasses de patrones de Skills Guard, e informes
  análogos contra heurísticas futuras. Estos componentes no son límites;
  vencerlos no es una vulnerabilidad bajo esta política.
- **Inyección de prompts per se.** Hacer que el LLM emita salida inusual
  — a través de contenido inyectado, alucinación, artefactos de entrenamiento,
  o cualquier otra causa — no es en sí mismo una vulnerabilidad. "Logré
  inyección de prompts" sin un resultado encadenado del §3.1 no es un informe
  procesable bajo esta política.
- **Consecuencias de una postura de aislamiento elegida.** Los informes de que
  una ruta de código que opera dentro del alcance de su postura puede hacer lo que esa
  postura permite no son vulnerabilidades. Ejemplos: herramientas de shell o archivos
  que alcanzan estado del host bajo el backend local; subprocesos de ejecución de código
  o MCP que alcanzan estado del host bajo aislamiento de backend de terminal que solo
  sandboxea el shell; informes cuyas precondiciones requieren acceso de escritura preexistente
  a archivos de configuración o credenciales propiedad del operador (esos ya están dentro
  del envolvente de confianza).
- **Configuraciones documentadas de emergencia.** Compensaciones seleccionadas por el operador
  que deshabilitan explícitamente protecciones: `--insecure` y flags equivalentes
  en el dashboard u otros componentes, aprobaciones deshabilitadas,
  backend local en producción, perfiles de desarrollo que evitan
  la seguridad de hermes-home, y similares. Los informes contra esas
  configuraciones no son vulnerabilidades — eso es el trabajo del flag.
- **Habilidades y plugins contribuidos por la comunidad.** Las habilidades de terceros
  (incluyendo el repositorio de habilidades de la comunidad) y los plugins de terceros
  están en la superficie de revisión del operador, no en la superficie de confianza de Hermes Agent
  (§2.4, §2.5). Una habilidad o plugin que haga algo
  malicioso es el modo de falla esperado de uno que no fue
  revisado, no una vulnerabilidad en Hermes Agent. Los errores en la ruta de
  instalación de habilidades o plugins de Hermes Agent que impidan al
  operador ver lo que está instalando están en alcance bajo el §3.1.
- **Exposición pública sin controles externos.** Exponer el
  gateway o la API a la internet pública sin autenticación,
  VPN o firewall.
- **Restricciones de lectura/escritura a nivel de herramienta en una postura donde el shell está
  permitido.** Si una ruta es alcanzable a través de la herramienta terminal, los informes
  de que otras herramientas de archivos pueden alcanzarla no añaden nada.

---

## 4. Fortalecimiento del Despliegue

La decisión de fortalecimiento más importante es hacer coincidir el aislamiento
(§2.2) con la confianza del contenido que el agente ingerirá. Más allá de eso:

- Ejecuta el agente como usuario no-root. La imagen de contenedor proporcionada
  hace esto por defecto.
- Mantén las credenciales en el archivo de credenciales del operador con permisos
  estrictos, nunca en la configuración principal, nunca en control de versiones.
  Bajo OpenShell, usa el almacén de Proveedores en lugar de un archivo de
  credenciales en disco.
- No expongas el gateway o la API a la internet pública sin
  VPN, Tailscale o protección de firewall. Bajo OpenShell, usa la
  capa de política de red para restringir el egreso.
- Configura una lista de llamadores permitidos para cada adaptador de red expuesto
  que habilites (§2.6).
- Revisa las habilidades y plugins de terceros antes de instalar (§2.4,
  §2.5). Para las habilidades, esto significa leer el Python y los scripts,
  no solo SKILL.md. Los informes de Skills Guard y el registro de auditoría
  de instalación son la superficie de revisión.
- Hermes Agent incluye guardias de cadena de suministro para lanzamientos de servidores
  MCP y para cambios de dependencias / paquetes incluidos en CI; consulta
  `CONTRIBUTING.es.md` para más detalles.

---

## 5. Divulgación

- **Ventana de divulgación coordinada:** 90 días desde el informe, o hasta que se
  publique una corrección, lo que ocurra primero.
- **Canal:** el hilo GHSA o correspondencia por email con
  security@nousresearch.com.
- **Crédito:** los reportadores reciben crédito en las notas de versión a menos que
  se solicite anonimato.
