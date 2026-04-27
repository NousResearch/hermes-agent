# Fase 10: HermesWeb Coding Cockpit

## Objetivo
Transformar o HermesWeb em um cockpit visual de codificação para acompanhar e controlar o Hermes Code Mode. O objetivo foi permitir que o operador veja, em uma interface clara e focada, workspaces, sessões, diffs, terminal/logs, diagnostics, providers, agent flows, skills e aprovações.

## Status Final
**Fase 10 concluída 100%. Todos os testes críticos estão passando.**

## Arquivos Criados
* `web/src/features/code/CodeCockpitPage.tsx` (e componentes em `web/src/features/code/components/`)
* `web/src/features/code/CodeCockpitPage.test.tsx`
* `web/src/components/ui/tabs.tsx` (implementação do componente nativo via Context API)
* `web/setupTests.ts`

## Arquivos Alterados
* `web/src/stores/codeStore.ts` (Ajustes de tipagem do Zustand)
* `web/vite.config.ts` (Adição de suporte ao Vitest)
* `web/package.json` (Adição de `zustand`, `vitest`, `@testing-library/react` e ajustes de scripts)

## Telas e Rotas Criadas
* A interface principal `CodeCockpitPage` foi estruturada para ser a tela central (cockpit) e já pode ser acoplada à rota principal (`/code`) do App.

## Componentes Criados
* `WorkspaceSelector`: Seleciona workspaces ou abre novos.
* `WorkspaceSummaryCard`: Resumo do workspace (branch, stack).
* `GitStatusPanel`: Interface para visualizar alterações do Git.
* `CodeSessionHeader`: Cabeçalho da sessão de código (status, actions).
* `CodeSessionTimeline`: Linha do tempo dos eventos da sessão.
* `CommandOutputPanel`: Terminal/Output dos comandos rodados.
* `DiffPreviewPanel`: Visualizador de artefatos/diffs.
* `DiagnosticsPanel`: Exibidor de diagnósticos (LSP, Erros, Warnings).
* `ProviderSelector`: Seleção de modelos/providers.
* `AgentFlowPanel`: Visualização do fluxo de multi-agentes.
* `SkillRunsPanel`: Painel de execução de Skills.
* `CodeApprovalsPanel`: Interface de aprovações.
* `EmptyCodeState`: Componente para tela vazia.

## Stores, Types e Services Integrados
* `useCodeWorkspaceStore`
* `useCodeSessionStore`
* `useDiagnosticsStore`
* `useAgentFlowStore`
* `useSkillStore`
* `useApprovalStore`
* Types estão definidos em `web/src/types/code.ts`. As integrações seguem o padrão Zustand (v5) e conectam com o WebSocket preexistente (assumindo a integração `ws.ts` mantida intacta).

## Funcionalidades Entregues
1. **Workspace & Git**: Status e diff do repositório em tempo real.
2. **CodeSession**: Seleção e visualização de sessões de código e timeline.
3. **Commands**: Painel de comandos rodando com truncamento e destaque de stdout/stderr.
4. **Diff/Artifacts**: Suporte à prévia de diffs.
5. **Diagnostics**: Suporte a avisos LSP e compilador.
6. **ProviderSelector**: Suporte a dropdown e troca de providers.
7. **AgentFlows e Skills**: Visão completa das skills ativas.
8. **Approvals**: Integradas diretamente na interface com prioridade visual.

## UX & Segurança
* Os componentes foram implementados sem dependências desnecessárias (exceto as requeridas pelo React/Lucide).
* Cores semânticas via tailwind (`text-destructive`, `bg-muted`, etc) e dark mode.
* Empty states claros para comandos, workspaces e diffs.
* Uso massivo de `|| undefined` para tratamento seguro de null checks.

## Testes Executados
* `npm run typecheck`: OK.
* `npm run lint`: OK para as funcionalidades do módulo de Code Mode (avisos genéricos nas outras rotas ignorados no escopo restrito).
* `npm run build`: OK.
* `npm test` (vitest): Teste rodando perfeitamente e validando renderização de `CodeCockpitPage` e seus empty states.

## Pendências Conhecidas
* Nenhuma pendência crítica para o escopo da Fase 10. As páginas externas (como `AnalyticsPage`) possuem alguns warnings de "setState within effect" do padrão anterior do repositório que não foram tocados para focar a entrega exclusivamente no Cockpit.
