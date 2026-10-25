# Exemplos de Uso

Exemplos ponta a ponta mostrando como o agente usa a skill. Em todo exemplo MCP
o agente retorna (ou abre) uma URL do editor do draw.io; nenhum arquivo é
escrito a menos que você peça.

## 0. Configure uma vez

```bash
python3 scripts/setup_drawio_mcp.py --target claude-code
```
Reinicie o cliente. Em seguida adicione uma instrução permanente: *"Always use
the draw.io MCP tools to create diagrams."*

---

## 1. Diagrama de arquitetura (app web de 3 camadas) — `open_drawio_xml`

**Prompt:** *"Use o MCP do draw.io para criar um diagrama de arquitetura de um
app web de 3 camadas: uma camada web (2 instâncias), uma camada de app (2
serviços atrás de um load balancer) e um banco PostgreSQL. Roteie as arestas ao
redor das formas."*

O agente authora o XML e chama:

```
open_drawio_xml(
  content="<mxfile ...><diagram ...><mxGraphModel ...><root>
    <mxCell id='0'/><mxCell id='1' parent='0'/>
    <!-- tiers como swimlanes, serviços dentro, LB, cilindro DB -->
    ... (veja references/architecture-patterns.md) ...
  </root></mxGraphModel></diagram></mxfile>",
  routing="libavoid"
)
```

Resultado: o editor abre com um diagrama de fios limpos, contornando formas.

## 2. Sequência / fluxograma — `open_drawio_mermaid`

**Prompt:** *"Use open_drawio_mermaid para criar um diagrama de sequência de
login OAuth2: User → App → Auth Provider → App → User."*

```
open_drawio_mermaid(
  content="sequenceDiagram
    participant U as User
    participant A as App
    participant P as Auth Provider
    U->>A: Login
    A->>P: Authorize
    P-->>A: Token
    A-->>U: Session"
)
```

## 3. Org chart — `open_drawio_csv`

**Prompt:** *"Use open_drawio_csv para criar um org chart: CEO → CTO, CFO; CTO →
3 Engenheiros."*

```
open_drawio_csv(
  content="CEO,,
CTO,CEO,
CFO,CEO,
Eng1,CTO,
Eng2,CTO,
Eng3,CTO,"
)
```

## 4. Ícones de nuvem — `search_shapes`

**Prompt:** *"Desenhe uma arquitetura AWS com Lambda, API Gateway e DynamoDB.
Use os ícones AWS corretos."*

O agente primeiro chama:
```
search_shapes(query="aws lambda")
search_shapes(query="aws api gateway")
search_shapes(query="aws dynamodb")
```
…cola os `style` retornados nos vértices do `open_drawio_xml`. (Cheatsheet
manual: `references/cloud-icons.md`.)

## 5. Export local + validar (sem MCP)

Quando você precisa de um arquivo de imagem:
```bash
# escreva architecture.drawio (author seguindo SKILL.md)
drawio -x -f png --width 2000 -o architecture.png architecture.drawio      # preview
drawio -x -f png -e -s 2 -o architecture.drawio.png architecture.drawio   # final
python3 scripts/validate_drawio.py architecture.drawio.png --repair-iend # corrige IEND
python3 scripts/validate_drawio.py architecture.drawio --score           # lint + score
```

## 6. Edição multi-página — `list/get/set_page`

```
list_pages(path="architecture.drawio")
get_page(path="architecture.drawio", page="Container")
set_page(path="architecture.drawio", page="Container", content="<mxGraphModel>...</mxGraphModel>")
```

---

Mais XML pronto: `references/architecture-patterns.md` (layered, microservices
com event bus, client→API→DB, C4-lite). Estilo/paleta/legenda:
`references/style-guide.md`. Ícones de nuvem: `references/cloud-icons.md`.
