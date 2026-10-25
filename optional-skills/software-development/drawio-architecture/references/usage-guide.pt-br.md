# Usando a Skill

Como conduzir a `drawio-architecture` após configurada. Esta skill é
**proativa**: o agente deve carregá-la sempre que um pedido implicar um
diagrama com 3+ componentes, fluxo de dados complexo ou explicação visual de um
sistema.

## Quando invocá-la

- "Desenhe um diagrama de arquitetura / sistema / serviço / rede / nuvem"
- "Faça um fluxograma / ER / sequência / UML class / modelo C4"
- "Mostre um diagrama de como X conversa com Y"
- Qualquer visualização que deva ser editável no draw.io/diagrams.net

## Dois caminhos de entrega

| Caminho | O que acontece | Melhor para |
|---------|----------------|-------------|
| **A. MCP** | Agente chama `open_drawio_xml`/_`mermaid`/_`csv` → editor do draw.io abre em aba do browser. Sem arquivo em disco. | Agentes de chat (Claude, OpenCode, Devin, AGY, Copilot, Cursor). Fluxos "mostre o diagrama". |
| **B. CLI local** | Agente escreve um `.drawio` e então `drawio -x` exporta PNG/SVG/PDF. | Headless/CI, entregáveis em imagem, sem MCP. |

Ambos compartilham as **mesmas regras de authoring XML** abaixo — só muda a
entrega.

## Fluxo de trabalho recomendado

1. **Esclareça** (1–3 perguntas se faltar): tipo de diagrama, modo de entrega
   (abrir via MCP vs arquivo+CLI vs imagem), escopo/fidelidade, tecnologias.
   Pule se o pedido já é específico.
2. **Configure** (caminho A) ou resolva o binário draw.io (caminho B).
3. **Planeje** formas, relações, layout (LR/TB), agrupamento (tier/container),
   necessidade de ícones.
4. **Author** o XML (ou escreva Mermaid para um diagrama tipado padrão).
5. **Entregue**: MCP → chama a ferramenta; CLI → escreve `.drawio`, exporta,
   opcionalmente self-check por visão.
6. **Itere** com edições XML direcionadas até aprovado.

## Regras de authoring (obrigatórias)

- `id="0"` e `id="1"` são células-raiz obrigatórias; formas do usuário começam
  em `id="2"` com ids únicos.
- Toda `mxCell` de aresta precisa de `<mxGeometry relative="1" as="geometry" />`
  — nunca feche arestas sozinhas.
- Use a grade rígida: `x = col*180+40`, `y = row*120+40`; retângulo `140×60`,
  diamante `140×80`, cilindro `100×70`.
- Não roteie arestas manualmente — declare `source`/`target` e deixe o router
  trabalhar, ou use `routing:"libavoid"` (MCP) para fios que contornam formas.
- **Nunca** coloque comentários XML (`<!-- -->`) na saída do diagrama.
- Inclua sempre `html=1` nos estilos das células; use `&#xa;` para quebras.
- Escape `&` `<` `>` `"` em valores de atributos.

## Cheatsheet de iteração

| Pedido | Edição XML |
|--------|------------|
| Mudar cor de X | ache `mxCell` por `value`, atualize `fillColor`/`strokeColor` |
| Adicionar nó | anexe `mxCell` vertex com próximo id perto dos nós relacionados |
| Remover nó | delete o vertex + arestas com `source`/`target` correspondentes |
| Mover/redimensionar | atualize `x`/`y` ou `width`/`height` no `mxGeometry` |
| Adicionar seta A→B | anexe aresta `mxCell` com `source`/`target` |
| Mudar rótulo | atualize `value` |
| Trocar LR↔TB | regeneração completa |

## Valide antes de entregar

```bash
python3 scripts/validate_drawio.py diagram.drawio --score
```

Referência completa de authoring: `SKILL.md` e `references/style-guide.md`,
`references/architecture-patterns.md`, `references/cloud-icons.md`.
