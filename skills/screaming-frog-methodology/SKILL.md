---
name: screaming-frog-methodology
description: "Metodología detallada de auditoría SEO con Screaming Frog SEO Spider, extraída de 4 tutoriales oficiales. Cubre: auditoría de XML Sitemaps, URLs canónicas, web scraping de datos on-page, y visualizaciones de arquitectura del sitio (Force-Directed Graph, Tree Graph)."
category: seo
created: "2026-07-09"
updated: "2026-07-10"
tags: [screaming-frog, seo-spider, sitemaps, canonicals, web-scraping, site-architecture, crawl-analysis]
---

# Screaming Frog SEO Spider — Metodología Detallada de Auditoría

Extraído de 4 tutoriales oficiales de Screaming Frog.

---

## 1. How to Audit XML Sitemaps

**URL original:** https://www.screamingfrog.co.uk/seo-spider/tutorials/how-to-audit-xml-sitemaps/
**Propósito:** Auditar y validar sitemaps XML para asegurar que todas las URLs importantes son indexables, no contienen errores y están optimizadas para el crawling de Google.

### Por qué es importante para SEO
- Los sitemaps XML son la principal vía para que Google descubra URLs nuevas en un sitio
- Sitemaps mal configurados pueden provocar que páginas importantes no sean indexadas
- URLs no canónicas, redirigidas o bloqueadas en sitemaps desperdician crawl budget
- Google puede ignorar sitemaps con más de 50.000 URLs o archivos >50MB sin comprimir

### Pasos Detallados en Screaming Frog

#### 1. Configurar el proyecto
1. Abrir Screaming Frog SEO Spider
2. Ir a **Configuration > Spider >** asegurarse que la configuración es correcta:
   - **Crawl Internal URLs**: ON
   - **Check External Links**: OFF (a menos que se necesiten)
   - **Crawl Canonicalised URLs**: OFF (para evitar duplicados)
3. Ir a **Configuration > Sitemaps** y verificar:
   - **Discover sitemaps from robots.txt**: ON
   - **Check Sitemaps**: ON

#### 2. Cargar el sitemap
1. En la barra de URL, pegar la URL del sitio web (ej. `https://ejemplo.com`)
2. Click **Start** para iniciar el crawl
3. SF automáticamente buscará el `robots.txt` y los sitemaps referenciados
4. También se puede cargar un sitemap directamente:
   - **Sitemaps > Open Sitemap** desde el menú superior
   - Pegar URL directa del sitemap (ej. `https://ejemplo.com/sitemap.xml`)

#### 3. Ir al tab de sitemaps
1. Click en **Sitemaps** en la barra de pestañas superior (junto a Internal, External, etc.)
2. Aquí se ven:
   - **URL**: la URL listada en el sitemap
   - **Sitemap**: el archivo sitemap de origen
   - **Status**: HTTP status code de la URL
   - **Status Code**: código de respuesta
   - **Indexability**: indexable o no indexable
   - **Reason**: por qué no es indexable (si aplica)

#### 4. Filtrar y analizar sitemaps

Filtros clave en el panel superior del tab Sitemaps:

**Por Status Code:**
- Filtro **Not Found (404)**: URLs en sitemap que devuelven 404
- Filtro **Redirect (3xx)**: URLs redirigidas en sitemap
- Filtro **Client Error (4xx)**: errores 4xx
- Filtro **Server Error (5xx)**: errores 5xx

**Por Indexability:**
- Filtro **Indexable**: URLs que Google puede indexar
- Filtro **Non-Indexable**: URLs bloqueadas (noindex, bloqueadas por robots.txt, etc.)
- Columna **Indexability** muestra el estado exacto

#### 5. Exportar resultados
1. **Sitemaps > Export > All Sitemaps URLs**
2. Formatos disponibles:
   - **Excel (.xlsx)**
   - **Google Sheets**
   - **CSV**
3. También: **Sitemaps > Export > Filtered Sitemaps URLs** para exportar solo las URLs filtradas

### Checklist de Hallazgos — XML Sitemaps

- [ ] URLs en sitemap que devuelven **404 Not Found**
- [ ] URLs en sitemap con **redirecciones (301/302)**
- [ ] URLs en sitemap marcadas como **No-Index** (meta robots noindex)
- [ ] URLs en sitemap **bloqueadas por robots.txt**
- [ ] URLs en sitemap que son **canonicalizadas a otra URL**
- [ ] URLs en sitemap con **errores 5xx**
- [ ] Sitemaps con más de **50.000 URLs** (límite de Google)
- [ ] Sitemaps con **tamaño de archivo >50MB** (sin comprimir)
- [ ] URLs en sitemap que **no tienen enlaces internos** desde el sitio
- [ ] Sitemaps que **no están referenciados en robots.txt**
- [ ] URLs en sitemap que son **páginas etiqueta/categoría vacías** (thin content)
- [ ] **Sitemaps de imágenes/vídeos** que faltan si el sitio tiene mucho contenido multimedia
- [ ] Sitemaps con **fechas `lastmod` incorrectas o futuras**
- [ ] Múltiples sitemaps index (sub-sitemaps) desbalanceados

### Interpretación de Resultados

| Hallazgo | Significado | Acción |
|----------|-------------|--------|
| URL en sitemap con 404 | Google perdiendo tiempo intentando indexar URLs muertas | Eliminar del sitemap o redirigir |
| URL redirigida en sitemap | Indica que la URL ha cambiado, pero el sitemap no se actualizó | Reemplazar con la URL canónica final |
| URL no-indexable en sitemap | Contradicción: pides indexar pero bloqueas la indexación | Quitar noindex o quitar del sitemap |
| URL bloqueada por robots.txt | Google no puede crawlear la URL aunque esté en sitemap | Ajustar robots.txt o quitar del sitemap |
| URL canonicalizada a otra | El sitemap lista una URL no canónica como principal | Cambiar a la URL canónica en el sitemap |

### Configuraciones Específicas de SF para Sitemaps

```
Configuration > Sitemaps > Discover sitemaps from robots.txt: ON
Configuration > Sitemaps > Check Sitemaps: ON
Sitemaps > Open Sitemap (carga directa)
Sitemaps > Export > All Sitemaps URLs
Sitemaps > Export > Filtered Sitemaps URLs
```

---

## 2. How to Audit Canonicals

**URL original:** https://www.screamingfrog.co.uk/seo-spider/tutorials/how-to-audit-canonicals/
**Propósito:** Auditar todas las etiquetas canónicas (rel=\"canonical\") del sitio para detectar configuraciones incorrectas que puedan diluir la autoridad SEO o causar problemas de contenido duplicado.

### Por qué es importante para SEO
- Las canónicas son la señal principal para Google de qué URL debe ser la versión canónica/principal
- Errores en canónicas pueden causar que la URL incorrecta sea indexada
- Cadenas de canónicas (A→B→C) desperdician autoridad
- Canónicas inconsistentes entre HTTP/HTTPS, www/no-www pueden fragmentar rankings
- Canónicas en páginas no-indexables son contradictorias con Google

### Pasos Detallados en Screaming Frog

#### 1. Configurar el proyecto
1. Abrir SF SEO Spider
2. **Configuration > Spider >**
   - **Crawl Canonicalised URLs**: ON (para crawlear también las URLs que son canónicas a otras)
   - **Check External Links**: según necesidad

#### 2. Iniciar el crawl
1. Ingresar URL del sitio en la barra de direcciones
2. Click **Start**
3. Esperar a que el crawl se complete (SF crawlea todas las URLs internas)

#### 3. Ir al tab de canónicas
1. Click en la pestaña **Canonicals** en la barra superior
2. Vista principal muestra:
   - **URL**: la URL actual
   - **Canonical URL**: la URL que señala como canónica
   - **Canonical Status**: el tipo de configuración canónica

#### 4. Filtrar por tipo de canónica

En el dropdown **Canonical Status** del panel de filtros:

**Opciones de filtro:**
1. **All URLs with Canonical**: todas las URLs que tienen etiqueta canónica
2. **Canonicalised**: URLs que son canónicas a otra URL diferente
3. **Self-Referencing**: URLs con canónica apuntando a sí mismas (correcto)
4. **Canonicalised (to external URL)**: canónicas que apuntan a otro dominio (peligroso)
5. **No Canonical**: URLs sin ninguna etiqueta canónica

#### 5. Revisar cadenas y bucles de canónicas

Para inspeccionar **Canonical Chains** y **Canonical Loops**, usar la ruta dedicada:

1. Ir a **Reports > Canonicals > Canonical Chains**
2. Aquí SF muestra:
   - **Source URL**: la URL origen de la cadena
   - **Canonical URL**: el destino canónico
   - **Redirect or Canonical Chain depth**: número de hops en la cadena
   - **Final URL**: la URL de destino final después de todas las redirecciones/canónicas
3. Las cadenas aparecen como A→B→C (múltiples hops)
4. Los bucles aparecen como A→B→A (circular)

**Canonical a External URL:**
1. Volver al tab Canonicals y filtrar por **Canonicalised (to external URL)**
2. Revisar que no haya canónicas apuntando a otros dominios por error

#### 6. Ver canónicas junto con otros datos

Usar el panel **Details** inferior:
1. Seleccionar una URL en el tab Canonicals
2. En el panel inferior, ir a **Canonical** tab
3. También revisar **Meta Robots** tab para ver si la página es indexable
4. Revisar **Headers** tab para ver si hay canónica en HTTP header

#### 7. Exportar datos de canónicas
1. **Canonicals > Export > All**
2. Formatos: Excel, Google Sheets, CSV
3. Columnas exportadas: URL, Canonical URL, Canonical Status, Status Code, etc.

### Checklist de Hallazgos — Canonicals

- [ ] URLs con **cadena de canónicas** (A → B → C) — más de 1 hop
- [ ] URLs en **bucle de canónicas** (A → B → A)
- [ ] Canónicas que apuntan a **URLs externas** (cross-domain)
- [ ] Canónicas que apuntan a **URLs 404**
- [ ] Canónicas que apuntan a **URLs redirigidas** (3xx)
- [ ] Canónicas que apuntan a **URLs no-indexables**
- [ ] Páginas indexables **sin etiqueta canónica**
- [ ] Canónica en **HTTP header** vs canónica en **HTML** — conflicto
- [ ] Canónicas con **protocolo inconsistente** (HTTP vs HTTPS)
- [ ] Canónicas con **www vs no-www inconsistente**
- [ ] Canónicas con **trailing slash inconsistente**
- [ ] Canónicas con **parámetros de URL** innecesarios
- [ ] Canónicas en páginas **paginadas** (ej. /page/2/ canónica a /page/1/)
- [ ] Canónicas **rel='canonical'** en páginas AMP

### Interpretación de Resultados

| Hallazgo | Significado | Acción |
|----------|-------------|--------|
| Self-referencing canónica | Correcto: la URL se señala a sí misma | Ninguna |
| Canonical chain (A→B→C) | La autoridad se diluye a través de múltiples hops | Acortar a 1 hop: A→C |
| Canonical loop (A→B→A) | Error: las URLs se señalan circularmente sin resolver | Romper el bucle, definir canónica correcta |
| Canónica a URL externa | Posible fuga de autoridad a otro dominio | Cambiar a URL interna a menos que sea intencional |
| Canónica a URL 404 | La URL destino no existe, la canónica no tiene efecto | Actualizar canónica a URL válida |
| Canónica a URL redirigida | Señal débil, Google seguirá la redirección | Apuntar a la URL final directamente |
| Sin canónica en página indexable | Google debe inferir la canónica, puede equivocarse | Añadir self-referencing canonical |
| Conflicto header vs HTML | Señales contradictorias para Google | Unificar ambas fuentes |

### Configuraciones Específicas de SF para Canonicals

```
Configuration > Spider > Crawl Canonicalised URLs: ON (para ver el destino final)
Canonicals tab > Canonical Status filter > [seleccionar tipo]
Reports > Canonicals > Canonical Chains (para cadenas y bucles)
Canonicals > Export > All
Tab Details > Canonical (por URL individual)
Tab Details > Meta Robots
Tab Details > Headers
```

---

## 3. Web Scraping

**URL original:** https://www.screamingfrog.co.uk/seo-spider/tutorials/web-scraping/
**Propósito:** Extraer datos personalizados de cualquier elemento HTML de las páginas crawleadas usando el motor de web scraping integrado de SF, para auditorías SEO avanzadas como extracción de H1, meta descripciones, schema markup, etc.

### Por qué es importante para SEO
- Permite extraer datos a escala que SF no captura por defecto
- Ideal para auditorías de contenido: extraer H1, H2, meta descriptions, schema JSON-LD, etc.
- Automatiza la revisión de elementos SEO on-page en miles de URLs
- Permite crear configuraciones de crawling altamente personalizadas
- Los datos extraídos se integran en los reportes de SF

### Pasos Detallados en Screaming Frog

#### 1. Acceder al Web Scraping Configuration
1. Ir a **Configuration > Access > Web Scraping**
2. O también: **Configuration > Custom > Extraction**
3. Se abre el panel **Web Scraping Configuration**

#### 2. Crear una nueva extracción
1. Click **+ Add** en la esquina inferior izquierda
2. Configurar cada campo:
   - **Name**: nombre descriptivo (ej. \"Meta Description\", \"H1 Tag\")
   - **CSS Path (CSS Selector)**: selector CSS del elemento a extraer
     - Ej: `meta[name=\"description\"]` para meta description
     - Ej: `h1` para primer H1
     - Ej: `script[type=\"application/ld+json\"]` para schema JSON-LD
   - **Attribute**: si se quiere extraer un atributo específico (ej. `content`, `href`, `src`)
     - Dejar vacío para extraer texto del elemento
   - **Extract First Only**: ON para solo el primer match, OFF para todos
     - ON para H1 (solo debe haber 1)
     - OFF para imágenes (puede haber múltiples)

#### 3. Tipos de selectores CSS soportados

| Selector | Ejemplo | Extrae |
|----------|---------|--------|
| Tag | `h1` | Texto del primer H1 |
| Atributo | `meta[name=\"description\"]` | Atributo `content` |
| Clase | `.product-title` | Texto del elemento |
| ID | `#main-content` | Texto del elemento |
| JSON-LD | `script[type=\"application/ld+json\"]` | Contenido JSON completo |
| Imagen | `img` | Atributo `src` y/o `alt` |
| Link | `a` | Atributo `href` |

#### 4. Atributos comunes para extraer

| Atributo | Uso |
|----------|-----|
| `content` | Meta tags (description, robots, etc.) |
| `href` | Enlaces |
| `src` | Imágenes, scripts |
| `alt` | Texto alternativo de imágenes |
| `data-*` | Atributos data personalizados |
| `class` | Clases CSS |
| (vacío) | Texto interno del elemento |

#### 5. Probar el scraping
1. Click **Test** en la parte inferior del panel
2. Ingresar una URL de prueba
3. SF muestra el resultado de la extracción
4. Ajustar el selector CSS si no devuelve lo esperado

#### 6. Aplicar la configuración
1. Click **OK** para guardar la configuración
2. SF preguntará: \"Do you want to recrawl all URLs?\" — **Yes** para aplicar a todo el crawl
3. SF recrawleará el sitio aplicando las nuevas extracciones

#### 7. Ver los datos extraídos
1. Ir al tab **Custom Extraction** (nueva pestaña que aparece)
2. Las columnas muestran los nombres de las extracciones configuradas
3. Cada fila = una URL, cada columna = el valor extraído

#### 8. Exportar datos extraídos
1. **Custom Extraction > Export > All**
2. Formatos: **Excel**, **Google Sheets**, **CSV**
3. Los datos se exportan con todas las columnas personalizadas

### Ejemplos Prácticos de Web Scraping para SEO

#### Extraer Meta Descriptions
```
Name: Meta Description
CSS Path: meta[name=\"description\"]
Attribute: content
Extract First Only: ON
```

#### Extraer H1 Tags
```
Name: H1 Tag
CSS Path: h1
Attribute: (vacío — texto interno)
Extract First Only: ON
```

#### Extraer Schema JSON-LD
```
Name: JSON-LD Schema
CSS Path: script[type=\"application/ld+json\"]
Attribute: (vacío — contenido JSON)
Extract First Only: OFF (por si hay múltiples)
```

#### Extraer Open Graph Tags
```
Name: OG Title
CSS Path: meta[property=\"og:title\"]
Attribute: content
Extract First Only: ON
```

#### Extraer H2 Tags
```
Name: H2 Tags
CSS Path: h2
Attribute: (vacío)
Extract First Only: OFF (múltiples H2s)
```

#### Extraer Atributos Alt de Imágenes
```
Name: Image Alt
CSS Path: img
Attribute: alt
Extract First Only: OFF
```

#### Extraer Canonical desde HTML
```
Name: Canonical HTML
CSS Path: link[rel=\"canonical\"]
Attribute: href
Extract First Only: ON
```

### Checklist de Web Scraping

- [ ] Probar selectores CSS complejos con el botón **Test** antes de aplicarlos
- [ ] Configurar **Extract First Only** correctamente según si se espera 1 o múltiples valores
- [ ] Si se extraen múltiples valores, los datos se concatenan separados por delimitador
- [ ] Verificar que el sitio no bloquea el scraping con JS (SF no ejecuta JS por defecto)
- [ ] Para sitios con JavaScript, activar **Configuration > Spider > JavaScript**: ON
- [ ] Las extracciones se guardan en el **Spider Configuration** y persisten entre sesiones
- [ ] Los datos extraídos aparecen también en el tab **Custom Extraction** de la barra inferior

### Interpretación de Resultados

**Para Meta Descriptions:**
- Vacío = falta meta description → escribir description única
- Duplicadas = múltiples páginas con misma meta description → personalizar
- Muy cortas (<120 chars) → expandir
- Muy largas (>160 chars) → acortar

**Para H1 Tags:**
- Vacío = falta H1 → añadir H1 descriptivo
- Múltiples H1s → reducir a 1 por página
- H1 no descriptivo → mejorar con keywords relevantes
- H1 duplicados en múltiples URLs → personalizar por página

**Para Schema JSON-LD:**
- Ausente = falta schema markup → implementar
- Múltiples schemas → revisar que no haya conflicto
- Errores de sintaxis JSON → validar con validador de schema.org

**Para Imágenes:**
- Alt vacío = falta texto alternativo → añadir alt descriptivo
- Alt genérico (\"imagen.jpg\") → mejorar con descripción relevante

### Configuraciones Específicas de SF para Web Scraping

```
Configuration > Access > Web Scraping
Configuration > Custom > Extraction
Configuration > Spider > JavaScript: ON (si el sitio carga contenido con JS)
Tab Custom Extraction (aparece tras configurar extracciones)
Custom Extraction > Export > All
Botón Test (dentro del panel de configuración)
```

---

## 4. Site Architecture Crawl & Visualisations

**URL original:** https://www.screamingfrog.co.uk/seo-spider/tutorials/site-architecture-crawl-visualisations/
**Propósito:** Visualizar la arquitectura del sitio mediante gráficos de crawl (mapas de sitio) para entender la estructura de enlaces, detectar páginas huérfanas, profundidad de crawl, distribución de PageRank interno e identificar problemas de navegación.

### Por qué es importante para SEO
- La arquitectura del sitio impacta directamente en cómo Google distribuye el crawl budget
- Páginas a más de 3-4 clicks de la homepage reciben menos autoridad y se indexan peor
- Las visualizaciones permiten detectar silos rotos, páginas huérfanas y cuellos de botella
- Entender el flujo de link juice interno es crítico para la indexación de contenido importante
- Google prioriza páginas con buena arquitectura de enlaces internos

### Pasos Detallados en Screaming Frog

#### 1. Configurar el proyecto
1. Abrir SF SEO Spider
2. **Configuration > Spider >**
   - **Crawl Internal URLs**: ON
   - **Check External Links**: OFF (para centrarse en la arquitectura interna)
   - **Crawl Canonicalised URLs**: OFF (opcional, según necesidad)
3. **Configuration > Crawl Depth >** establecer límite (ej. 20 para sitios grandes)

#### 2. Iniciar el crawl
1. Ingresar URL del sitio
2. Click **Start**
3. Esperar a que el crawl se complete

#### 3. Generar visualización
1. **Crawl Analysis > Visualisations** (menú superior izquierdo)
2. Se abre el **Visualisations** popup/diálogo
3. SF ofrece los siguientes tipos de visualización:

#### 3a. Force-Directed Graph
- **Qué muestra:** Nodos (URLs) conectados por enlaces entre ellos
- **Layout:** Los nodos se posicionan mediante algoritmo de fuerzas
- **Útil para:** Vista general de la estructura del sitio, identificación de clusters
- **Configuración:**
  - **Max Nodes**: límite de nodos a mostrar (por defecto 500)
  - **Node Size**: basado en número de enlaces entrantes (inlinks)
  - **Node Colour**: por tipo de contenido, status code, etc.
  - **Group Similar URLs**: agrupa URLs por patrón

#### 3b. Tree Graph (Jerárquico)
- **Qué muestra:** Estructura jerárquica desde la homepage hacia abajo
- **Layout:** Árbol con la homepage en la raíz
- **Útil para:** Ver profundidad de cada URL, identificar páginas enterradas
- **Configuración:**
  - **Start URL**: URL raíz del árbol
  - **Max Depth**: profundidad máxima a mostrar
  - **Show All URLs**: ON para mostrar todas, OFF para mostrar solo las conectadas

#### 4. Interactuar con la visualización
- **Click en nodo**: abre detalles de la URL
- **Zoom/Pan**: navegar por el gráfico
- **Hover**: muestra URL y métricas básicas
- **Selección múltiple**: Shift+Click para seleccionar varios nodos
- **Right-click**: opciones de exportación y filtrado

#### 5. Filtrar la visualización
- **Inlinks**: filtrar nodos por número de enlaces entrantes
- **Outlinks**: filtrar por número de enlaces salientes
- **Status Code**: mostrar solo ciertos códigos de estado
- **Content Type**: filtrar por tipo de contenido
- **Depth**: filtrar por rango de profundidad
- **Directory**: filtrar por subdirectorio específico

#### 6. Exportar visualización
1. **Export >** opciones:
   - **Image (.png, .jpg, .svg)**
   - **GraphML** (para Gephi u otras herramientas de análisis de redes)
   - **GEXF** (formato para análisis de redes)
   - **CSV edges/nodes** (datos tabulares de la red)

#### 7. Análisis de arquitectura desde los datos

**View Crawl Depth:**
1. En el tab **Internal** (o **Response Codes**)
2. Hacer scroll hasta columna **Crawl Depth**
3. Ordenar por profundidad descendente
4. Identificar URLs que requieren muchos clicks desde la homepage

**View Inlinks:**
1. Tab **Internal** > columna **Inlinks**
2. Ordenar descendente para ver URLs con más enlaces internos
3. Identificar URLs con 0 inlinks (huérfanas)

**View Outlinks:**
1. Tab **Internal** > columna **Outlinks**
2. Identificar páginas con pocos enlaces salientes (posibles dead ends)

#### 8. Combinar visualizaciones con filtros

1. Aplicar filtros en el tab **Internal** (ej. mostrar solo 4xx)
2. **Crawl Analysis > Visualisations** con filtros aplicados
3. La visualización solo mostrará los datos filtrados
4. Útil para ver arquitectura de páginas con errores

### Checklist de Hallazgos — Site Architecture

- [ ] Páginas con **profundidad > 4** desde la homepage
- [ ] Páginas **huérfanas** (0 inlinks internos)
- [ ] **Silos rotos**: páginas de una categoría que enlazan a otra categoría incorrecta
- [ ] **Páginas sin enlaces salientes** (dead ends que no transmiten link juice)
- [ ] URLs con **más de 100 enlaces salientes** (dilución de link juice)
- [ ] **Clusters desconectados**: grupos de páginas que no reciben enlaces del resto del sitio
- [ ] **Excesiva profundidad** en contenido importante (blog posts a 6+ clicks)
- [ ] **Desequilibrio en distribución de inlinks** (pocas páginas acaparan todo el link juice)
- [ ] **Páginas importantes** (productos, servicios) con pocos enlaces internos
- [ ] **Navegación inconsistente**: menú que no está presente en todas las páginas
- [ ] **Páginas con parámetros de sesión** que crean URLs infinitas (spider traps)
- [ ] **Múltiples paths** para llegar al mismo contenido (canibalización de estructura)
- [ ] **Contenido thin** a mucha profundidad

### Interpretación de Resultados

| Hallazgo | Significado | Acción |
|----------|-------------|--------|
| Profundidad > 4 | Páginas en niveles profundos reciben poca autoridad | Acercar a la homepage con mejores enlaces internos |
| 0 inlinks (huérfana) | URL invisible para Google (no descubrible) | Añadir enlaces internos desde páginas relevantes |
| Dead end (0 outlinks) | Página que no transmite link juice a otras | Añadir enlaces internos relevantes |
| Cluster desconectado | Grupo de URLs que forman un sub-sitio aislado | Añadir enlaces cruzados con el resto del sitio |
| Spider trap | URLs infinitas que desperdician crawl budget | Bloquear con robots.txt o corregir parámetros |
| >100 outlinks por página | Dilución excesiva de link juice | Reducir número de enlaces por página |
| Silos rotos | Estructura temática inconsistente | Reorganizar enlaces para reforzar silos |
| Inlinks muy concentrados | Pocas URLs reciben casi todo el link juice | Distribuir enlaces internos más equitativamente |

### Configuraciones Específicas de SF para Visualizaciones

```
Configuration > Spider > Crawl Internal URLs: ON
Configuration > Crawl Depth > límite (ej. 20)
Crawl Analysis > Visualisations > Force-Directed Graph
Crawl Analysis > Visualisations > Tree Graph

Filtros en visualización:
- Max Nodes (forzado: 500 default)
- Node Size: por inlinks
- Node Colour: por status code o tipo
- Group Similar URLs: ON/OFF
- Max Depth

Exportación:
- Export > Image (.png, .jpg, .svg)
- Export > GraphML
- Export > GEXF
- Export > CSV edges/nodes

Datos tabulares:
Tab Internal > Crawl Depth
Tab Internal > Inlinks
Tab Internal > Outlinks
```

---

## Resumen de Herramientas y Comandos Clave de SF

### Menú Configuration (configuración general)

| Ruta | Función |
|------|---------|
| `Configuration > Spider` | Configuración general de crawling |
| `Configuration > Sitemaps` | Configuración de detección de sitemaps |
| `Configuration > Access > Web Scraping` | Configuración de extracción personalizada |
| `Configuration > Custom > Extraction` | Alternativa a Web Scraping |
| `Configuration > Crawl Depth` | Límite de profundidad de crawling |

### Tabs Principales

| Tab | Función |
|-----|---------|
| **Internal** | URLs internas del sitio |
| **Canonicals** | Estado de todas las URLs canónicas |
| **Sitemaps** | URLs encontradas en sitemaps XML |
| **Custom Extraction** | Datos extraídos mediante web scraping |
| **Response Codes** | Códigos de respuesta HTTP |

### Exportación

| Ruta | Formato |
|------|---------|
| `Sitemaps > Export > All` | Excel, CSV, Google Sheets |
| `Canonicals > Export > All` | Excel, CSV, Google Sheets |
| `Custom Extraction > Export > All` | Excel, CSV, Google Sheets |
| `Export > Image` | PNG, JPG, SVG (visualizaciones) |
| `Export > GraphML / GEXF` | Datos de red (visualizaciones) |

### Pipeline de Auditoría Recomendado

1. **Configurar** el proyecto con las opciones adecuadas
2. **Crawlear** el sitio completamente
3. **Revisar Sitemaps** primero — URLs no deseadas en sitemaps
4. **Auditar Canonicals** — errores en señales de canonicalización
5. **Web Scraping** — extraer datos on-page personalizados
6. **Visualizar arquitectura** — entender estructura y detectar problemas
7. **Exportar** reportes para documentar hallazgos
8. **Priorizar** acciones basadas en severidad de hallazgos

---

_Documentación extraída de los tutoriales oficiales de Screaming Frog SEO Spider._
