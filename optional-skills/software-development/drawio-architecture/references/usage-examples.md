# Usage Examples

End-to-end examples showing how the agent uses the skill. In every MCP example
the agent returns (or opens) a draw.io editor URL; no file is written unless you
ask for one.

## 0. Configure once

```bash
python3 scripts/setup_drawio_mcp.py --target claude-code
```
Restart the client. Then add a standing instruction: *"Always use the draw.io
MCP tools to create diagrams."*

---

## 1. Architecture diagram (3-tier web app) — `open_drawio_xml`

**Prompt:** *"Use the draw.io MCP to make an architecture diagram of a 3-tier
web app: a web tier (2 instances), an app tier (2 services behind a load
balancer), and a PostgreSQL database. Route edges around shapes."*

The agent authors XML and calls:

```
open_drawio_xml(
  content="<mxfile ...><diagram ...><mxGraphModel ...><root>
    <mxCell id='0'/><mxCell id='1' parent='0'/>
    <!-- tiers as swimlanes, services inside, LB, DB cylinder -->
    ... (see references/architecture-patterns.md) ...
  </root></mxGraphModel></diagram></mxfile>",
  routing="libavoid"
)
```

Result: the editor opens with a clean, obstacle-avoiding wiring diagram.

## 2. Sequence / flowchart — `open_drawio_mermaid`

**Prompt:** *"Use open_drawio_mermaid to create a sequence diagram of OAuth2
login: User → App → Auth Provider → App → User."*

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

**Prompt:** *"Use open_drawio_csv to create an org chart: CEO → CTO, CFO; CTO →
3 Engineers."*

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

## 4. Cloud icons — `search_shapes`

**Prompt:** *"Draw an AWS architecture with Lambda, API Gateway, and DynamoDB.
Use the proper AWS icons."*

The agent first calls:
```
search_shapes(query="aws lambda")
search_shapes(query="aws api gateway")
search_shapes(query="aws dynamodb")
```
…pastes the returned `style` strings into `open_drawio_xml` vertices. (Hand
cheatsheet: `references/cloud-icons.md`.)

## 5. Local export + validate (no MCP)

When you need an image file:
```bash
# write architecture.drawio (author per SKILL.md rules)
drawio -x -f png --width 2000 -o architecture.png architecture.drawio      # preview
drawio -x -f png -e -s 2 -o architecture.drawio.png architecture.drawio   # final
python3 scripts/validate_drawio.py architecture.drawio.png --repair-iend # fix IEND
python3 scripts/validate_drawio.py architecture.drawio --score           # lint + score
```

## 6. Multi-page edit — `list/get/set_page`

```
list_pages(path="architecture.drawio")
get_page(path="architecture.drawio", page="Container")
set_page(path="architecture.drawio", page="Container", content="<mxGraphModel>...</mxGraphModel>")
```

---

More worked XML: `references/architecture-patterns.md` (layered, microservices
with event bus, client→API→DB, C4-lite). Style/palette/legend:
`references/style-guide.md`. Cloud icons: `references/cloud-icons.md`.
