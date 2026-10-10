# Referência das Ferramentas MCP

O servidor MCP oficial do draw.io (`npx @drawio/mcp`) expõe estas ferramentas.
Mencione o nome da ferramenta explicitamente nos prompts para que o agente use
o MCP em vez de escrever um arquivo manualmente (*"Always use the draw.io MCP
tools to create diagrams."*).

## `open_drawio_xml`

Abre o editor do draw.io com XML nativo draw.io/mxGraph. Esta é a ferramenta
principal para diagramas de arquitetura.

| Param | Tipo | Obrigatório | Notas |
|-------|------|-------------|-------|
| `content` | string | Sim | XML `<mxfile>…</mxfile>` completo |
| `lightbox` | boolean | Não | Visão somente-leitura (padrão `false`) |
| `dark` | string | Não | `"auto"` / `"true"` / `"false"` (padrão `"auto"`) |
| `routing` | string | Não | `"libavoid"` redireciona conectores ao redor das formas antes de abrir |

**Passes de layout (só MCP):**
- `routing: "libavoid"` — mantém as posições dos nós; redireciona arestas ao redor das formas.
- `postLayout: "elk"` — re-layout ELK completo; adicione `direction: "horizontal"` para LR.
  **Não** combine ambos.

## `open_drawio_mermaid`

Abre o editor com um diagrama **Mermaid.js**. Melhor para diagramas tipados
padrão (flowchart, sequence, class, state, ER, gantt, mindmap, sankey…) sem
necessidade de estilo/ícone customizado.

| Param | Tipo | Obrigatório | Notas |
|-------|------|-------------|-------|
| `content` | string | Sim | Sintaxe Mermaid.js |
| `lightbox` | boolean | Não | Visão somente-leitura |
| `dark` | string | Não | `"auto"` / `"true"` / `"false"` |

Fluxogramas Mermaid complexos (≥ ~20 nós, ≥ 3 diamantes, arestas de retorno ou
≥ 3 pontas) precisam de `postLayout: "elk"`.

## `open_drawio_csv`

Abre o editor com **CSV** convertido em diagrama (org charts, tabelas).

| Param | Tipo | Obrigatório | Notas |
|-------|------|-------------|-------|
| `content` | string | Sim | Conteúdo CSV |
| `lightbox` | boolean | Não | Visão somente-leitura |
| `dark` | string | Não | `"auto"` / `"true"` / `"false"` |

## `search_shapes`

Pesquisa a biblioteca de ~10.000 formas do draw.io (AWS, Azure, GCP, Cisco,
Kubernetes, P&ID, elétrico, BPMN…) e retorna `style` prontos para uso em
`open_drawio_xml`. Complementado pelo serviço de ícones do draw.io para logos
de marca.

| Param | Tipo | Obrigatório | Notas |
|-------|------|-------------|-------|
| `query` | string | Sim | Palavras-chave separadas por espaço, ex. `aws lambda`, `cisco router` |
| `limit` | number | Não | Máx. resultados (padrão 10, máx. 50) |

> Use `search_shapes` só para diagramas que precisam de ícones setoriais/de
> marca. Pule para fluxogramas/UML/ERD/org charts/mind maps.

## `list_pages` / `get_page` / `set_page`

Acesso por página a um arquivo `.drawio`/`.xml` multi-página local. Páginas são
endereçadas por índice 0-based, nome exato ou id.

| Ferramenta | Params | Resultado |
|------------|--------|-----------|
| `list_pages` | `path` | `[{index, id, name, approxSizeBytes}]` |
| `get_page` | `path`, `page` | O XML `<mxGraphModel>` da página |
| `set_page` | `path`, `page`, `content` | Substitui o conteúdo da página (um `<mxGraphModel>`); outras páginas intactas |

## Ferramentas helper locais (deste repo)

Estas **não** são ferramentas MCP — são scripts Python que você executa:

- `scripts/setup_drawio_mcp.py` — detecta plataforma + escreve config MCP / gera
  URL de viewer no browser.
- `scripts/validate_drawio.py` — lint XML, score de legibilidade, reparo IEND de
  PNG `-e`.
