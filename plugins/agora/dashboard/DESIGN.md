---
version: alpha
name: Hermes Dashboard Square UI
description: Ágora deve parecer uma extensão nativa do dashboard Hermes.
colors:
  ink: "#17171A"
  primary: "#0053FD"
  canvas: "#F8FAFF"
  sidebar: "#F3F7FF"
  panel: "#FFFFFF"
  subtle-panel: "#FCFCFC"
  danger: "#CF2D56"
  on-ink: "#FFFFFF"
  on-primary: "#FFFFFF"
typography:
  h1:
    fontFamily: "Collapse"
    fontSize: 1.75rem
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "0em"
  h2:
    fontFamily: "var(--dt-font-sans)"
    fontSize: 1rem
    fontWeight: 600
    lineHeight: 1.2
  body-md:
    fontFamily: "var(--dt-font-sans)"
    fontSize: 0.875rem
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: "var(--dt-font-sans)"
    fontSize: 0.75rem
    fontWeight: 400
    lineHeight: 1.45
  label-ui:
    fontFamily: "var(--dt-font-sans)"
    fontSize: 0.75rem
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.02em"
  mono-sm:
    fontFamily: "var(--dt-font-mono)"
    fontSize: 0.75rem
    fontWeight: 400
    lineHeight: 1.35
rounded:
  none: 0px
  sm: 0px
  md: 0px
  lg: 0px
  full: 0px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  xxl: 32px
components:
  page-shell:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: 16px
  sidebar-surface:
    backgroundColor: "{colors.sidebar}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: 12px
  panel-surface:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: 16px
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.label-ui}"
    rounded: "{rounded.none}"
    padding: 12px
  button-secondary:
    backgroundColor: "{colors.subtle-panel}"
    textColor: "{colors.ink}"
    typography: "{typography.label-ui}"
    rounded: "{rounded.none}"
    padding: 12px
  input-field:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.body-md}"
    rounded: "{rounded.none}"
    padding: 12px
  task-tag:
    backgroundColor: "{colors.subtle-panel}"
    textColor: "{colors.ink}"
    typography: "{typography.mono-sm}"
    rounded: "{rounded.none}"
    padding: 6px
  danger-inline:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.on-ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.none}"
    padding: 8px
---

## Overview

Ágora deve parecer uma continuação natural do dashboard do Hermes, não um produto visual paralelo. A prioridade é continuidade visual e operacional: mesmos tokens, mesmas fontes, mesma hierarquia de texto, mesmas affordances de botão, mesmas superfícies e o mesmo vocabulário cromático do dashboard já existente.

A interface deve ter leitura de ferramenta de trabalho: plana, precisa, densa o suficiente para coordenação, sem ornamentos de chat genérico. O frontend deve evitar qualquer visual que lembre bolhas arredondadas, pills, chips ovais, cartões fofos ou componentes “brand new”. A sensação correta é: “isso já fazia parte do Hermes”.

## Colors

- **Ink ({colors.ink}):** base de texto e contorno semântico. Corresponde ao foreground principal já usado pelo dashboard.
- **Primary ({colors.primary}):** cor de ação principal do Hermes. Reutilize o mesmo acento já exposto pelos tokens do dashboard para CTA, seleção e foco.
- **Canvas ({colors.canvas}):** fundo principal da página e áreas de trabalho abertas.
- **Sidebar ({colors.sidebar}):** superfícies laterais, trilhos e navegação auxiliar.
- **Panel ({colors.panel}) / Subtle Panel ({colors.subtle-panel}):** superfícies internas do dashboard. Use o que o próprio Hermes já expõe; não introduza cinzas ou brancos novos.
- **Danger ({colors.danger}):** estados destrutivos e erros funcionais apenas. Nenhum vermelho inventado fora dos tokens existentes.

Implementação obrigatória:

- Reutilizar os tokens do dashboard Hermes antes de qualquer valor literal: `--ui-*`, `--theme-*`, `--color-*`, `--shadow-nous`, `--stroke-nous`, `--dt-*`.
- Não criar paleta própria do Ágora.
- Não hardcodar novos hex em componentes do frontend quando já existir token equivalente no dashboard.

## Typography

- **Display / heading principal:** usar `Collapse` apenas para momentos de título/identidade já compatíveis com o dashboard Hermes.
- **Texto corrente, labels, formulários, navegação:** usar a mesma família já resolvida por `--dt-font-sans`.
- **IDs, task ids, slugs, artefatos, metadados técnicos:** usar `--dt-font-mono`.
- O peso tipográfico deve carregar a hierarquia. Não compensar com caixas coloridas ou formas arredondadas.
- O frontend não deve introduzir fonte alternativa local para o Ágora.

## Layout

- Usar a mesma lógica de gutters, respiros, densidade e agrupamento do dashboard Hermes.
- Separação entre blocos deve vir de espaçamento e hairlines, não de caixas sobre caixas.
- Painéis e listas devem ser quadrados, alinhados e flush; sem “cards dentro de cards”.
- Channel list, agent rail, thread list, composer, admin/settings e painéis de detalhe devem compartilhar a mesma gramática espacial do dashboard.
- Reutilizar componentes e primitives existentes do dashboard quando possível, especialmente botões, inputs, campos de busca, loaders, estados vazios e rows de configuração.

## Elevation & Depth

- Overlays e superfícies elevadas devem reutilizar `--shadow-nous` e `--stroke-nous`.
- Fora isso, preferir interface plana. Sem sombras locais ad hoc.
- Não usar bordas pesadas. Hairlines do dashboard Hermes devem resolver a maior parte da separação visual.
- Nada de vidro, glow, neomorfismo ou estilo “chat app”.

## Shapes

Regra mandatória: **não usar cantos arredondados no frontend do Ágora**.

- Botões: quadrados
- Inputs: quadrados
- Painéis: quadrados
- Badges/tags/task pills: quadrados
- Dropdowns/popovers: quadrados
- Tabs/segmentos/composer: quadrados
- Message treatment e task chips: quadrados

Não usar:

- `border-radius: 999px`
- `rounded-*`
- `var(--radius-*)`
- pills, chips ovais ou capsules

Se algum primitive herdado do Hermes vier com raio por padrão, o frontend deve sobrescrever para `0px` no escopo do Ágora em vez de introduzir mais variações visuais.

## Components

- **Buttons:** reutilizar o componente de botão do dashboard Hermes, com seus tokens de cor, fonte, hover e active. O Ágora não deve reinventar botão. Apenas aplique a política de raio zero.
- **Inputs / Composer / Admin forms:** reutilizar tokens de input do dashboard. Sem chrome custom do plugin, sem radius local.
- **Channel rows / Agent rows / Settings rows:** seguir o padrão de rows/listas do Hermes, com hover e selected state derivados de `--ui-control-hover-background`, `--ui-control-active-background`, `--ui-row-hover-background` e afins.
- **Task IDs / artifact refs / slugs:** renderizar com mono e shape quadrado, não em pills arredondadas.
- **Errors / warnings / empty states / loading:** reutilizar os padrões visuais do dashboard para `ErrorState`, loaders e feedback. Não criar banners próprios com estilo desalinhado.
- **Admin settings:** a área de admin do Ágora deve parecer um painel nativo do dashboard, não um mini-app isolado.

## Do's and Don'ts

- **Do** reutilizar os design tokens já existentes do dashboard Hermes (`--ui-*`, `--theme-*`, `--color-*`, `--dt-*`).
- **Do** reutilizar botões, fontes, inputs, rows e padrões de superfície do próprio Hermes.
- **Do** tratar o Ágora como extensão do dashboard, não como marca visual independente.
- **Do** usar `0px` de radius em toda a superfície do Ágora.
- **Do** preferir whitespace + hairline + tipografia para hierarquia visual.

- **Don't** usar cantos arredondados.
- **Don't** criar novos tokens locais de cor, sombra, raio ou tipografia quando o dashboard já expõe os necessários.
- **Don't** hardcodar hex, rgba, box-shadow, font-family ou border-radius em componentes do Ágora sem necessidade extrema e documentada.
- **Don't** reinventar o estilo de botões, search field, segmented control, badges ou painéis.
- **Don't** deixar o Ágora com aparência de Slack/Discord/Linear clone; ele precisa parecer Hermes primeiro.
