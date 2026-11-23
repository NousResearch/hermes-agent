[Read in English](README.md)

# Tema One Dark Pro e Ajustes do Terminal Integrado (Hermes Desktop / Webapp)

Este documento resume o preset do tema Atom One Dark Pro e os ajustes do terminal integrado mantidos na branch local `theme/one-dark-sync` do Hermes Agent (`apps/desktop/`).

## Visão Geral

A skin de backend `one-dark` é mapeada para um preset exato do Atom One Dark Pro no renderer (`apps/desktop/src/themes/one-dark.ts` + `apps/desktop/src/themes/one-dark.css`), protegido por proveniência em runtime (`Symbol('one-dark-pro')` e `data-hermes-palette='one-dark'`), preservando temas instalados pelo usuário e garantindo limpeza completa ao trocar de tema.

## Principais Mudanças

### Tema e Tipografia (`apps/desktop/src/themes/`)
- **Superfícies principais:** Chrome/sidebar `#282c34`, editor/cards/popovers/terminal `#21252b`, inputs/composer `#1d1f23`, muted/balão do usuário `#2c313a`, bordas contínuas `#181a1f`.
- **Acentos e Markdown:** Títulos e acento principal `#e06c75`, negrito `#d19a66`, links `#61afef`, marcadores de lista (`li::marker`, `--tw-prose-bullets`, `--tw-prose-counters`) `#abb2bf`, checkboxes `#c678dd`.
- **Seleção de texto (`::selection`):** `background: rgba(97, 175, 239, 0.4)` e `color: #e6e6e6`.
- **Chrome da interface:** Fio contínuo entre sidebar e conteúdo (`--sidebar-edge-border: #181a1f`), sublinhado ativo de 2px nas abas (`pane-tab.tsx`), hairlines na sidebar (`sidebar/index.tsx`, `session-row.tsx`) e efeito "Thinking" (`od-sun` em `message-parts.tsx` com fallback para `prefers-reduced-motion`).

### Terminal Integrado (`apps/desktop/src/app/right-sidebar/terminal/`)
- **Paleta xterm de alto contraste:** Paletas `terminal` e `darkTerminal` explícitas em `one-dark.ts` sobre `--ui-terminal-surface-background: #21252b` (texto principal `#d7dae0` com contraste WCAG **10.99:1**; todas as cores ANSI legíveis >= **4.82:1**).
- **Alinhamento sob CSS zoom:** `PersistentTerminal` (`persistent.tsx`) e `terminalSelectionAnchor` (`selection.ts`) desescalam `getBoundingClientRect()` por `currentCSSZoom` (fallback `documentElement.style.zoom || 1`), usando `Math.floor` na borda direita para o overlay fixo ficar colado abaixo da aba `TERMINAL` sem cobrir o `TerminalRail` nem o botão `+`.
- **Rail e inicialização:** Borda vertical do `TerminalRail` até o final do painel (`chrome.tsx`, `rail.tsx`), espaçamento sem sobreposição dos ícones (`shrink-0`, `gap-1.5`), fit de altura completa (`instance.tsx`, `use-terminal-session.ts`) e ancoragem no topo para shells novos/expirados (`terminal-session-attempt.ts`).

## Build e Testes

Execute a partir da raiz do repositório (`hermes-agent`), usando o runtime Node/pnpm compatível com `pm/lock.json` (`--no-install` reutiliza `.build/webapp-workspace`):

```bash
export PATH="$HOME/.hermes/tools/pnpm-12.15.0-linux-x64:$HOME/.hermes/tools/node-26.12.1-linux-x64/bin:$PATH"

# Build de produção do webapp (sem instalar dependências)
node apps/desktop/scripts/build-webapp.mjs --no-install

# Testes unitários/CSSOM do tema e do terminal integrado
cd .build/webapp-workspace/apps/desktop
./node_modules/.bin/vitest run \
  src/themes \
  src/app/right-sidebar/terminal/persistent.test.tsx \
  src/app/right-sidebar/terminal/selection.test.ts \
  src/app/right-sidebar/terminal/use-terminal-session.test.tsx \
  src/app/right-sidebar/terminal/instance.test.tsx \
  src/app/right-sidebar/terminal/rail.test.tsx
```

## Documentação do Projeto
- Decisões arquiteturais (ADRs): [`.ai/adr/`](../../.ai/adr/)
- Armadilhas e comportamentos conhecidos: [`.ai/pitfalls.md`](../../.ai/pitfalls.md)
- Regras técnicas e limites de escopo: [`.ai/regras.md`](../../.ai/regras.md), [`.ai/teto.md`](../../.ai/teto.md)
- Skills disponíveis para o agente: [`.agents/skills/README.md`](../../.agents/skills/README.md)
