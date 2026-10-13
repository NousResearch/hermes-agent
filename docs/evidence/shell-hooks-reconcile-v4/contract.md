# V4 — diagnóstico operacional focal e contrato de correção

## Resultado

**Defeito persiste no código em disco e a coexistência V3/V4 foi novamente observada no processo operacional. V4 exclusivo NÃO comprovado.** Não houve reinício, instalação, edição de configuração/runtime nem criação de card.

Sessão nativa `20261012_230127_daac0b`, modelo efetivo `gpt-6-astra`, provider `openai-codex`: `~/.hermes/logs/agent.log:16035,16053`, também registrados em `v4-runtime-proveniencia.json`. Nenhum launcher externo ou provedor adicional foi chamado. O log confirma a rota, não audita faturamento da assinatura.

## Prova retomada e repetida

Origem: `atlas/wt/lovable-inventario-20261012/docs/evidencia/rag-lovable/GATE-V4-ESTADO.md:15–23`; transcript `~/.hermes/cache/delegation/live/deleg_27e64ba2/task-0.log:129–146`.

Processo atual: PID **1394**, pai **1119**, iniciado em **2026-10-10 22:54:22**, comando `/Users/farantes/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main --profile default serve --host 127.0.0.1 --port 0`. Pai é o Desktop em `apps/desktop/release/mac-arm64/Hermes.app`.

Durante oito chamadas inofensivas `terminal(true)`, amostragem `ps` registrou filhos do mesmo PID 1394:

- V3: PID 81768 e 81971, comando `python3 /Users/farantes/.local/share/atlas/hooks/dfba147efdcecaa3b427a266b67a1538ae9c92e8/hook-padrao-card.py`.
- V4: PID 81774 e 81975, comando `python3 /Users/farantes/.local/share/atlas/hooks/33bc07cfd94cd732e07319bf51281814e9d40103/ferramentas/agentes/hook-padrao-card.py`.

Recibo: `v4-processos-observados.json`. Não é atribuição exclusiva a esta sessão: outras sessões compartilham o backend. É prova suficiente de coexistência atual no processo, não de qual callback julgou um create desta sessão — nenhum create foi emitido.

Repetição standalone por stdin, com corpo previamente existente e controles de um único campo, sem executar ferramenta/CLI de criação:

| Hook | Caso | rc | Resultado |
|---|---|---:|---|
| V4 | V4 completo | 0 | sem bloqueio |
| V3 | mesmo V4 completo | 2 | `Padrao deve ser v3` |
| V4 | somente `Padrao=v3` | 2 | esperado v4 |
| V4 | somente `Parar=3 runs` | 2 | V4-01: faltam rate_limited, spawn_failed, crashed |

Fonte final: `v4-vereditos-finais.json`. A primeira tentativa de fixture esqueceu `[doc]` no título e foi corretamente recusada; recibo preservado em `v4-vereditos-stdin.json`, **não** serve como positivo. Título corrigido, reexecutado; todos os payloads finais têm `max_runtime_seconds=300` para não disparar a recusa independente de teto de execução.

## Repositório e responsabilidade reais

- Checkout consumidor: `/Users/farantes/.hermes/hermes-agent`, branch `t78-c28-pin`, HEAD `db72e58229639b80d68a53cdc1d081f6511457f5`, árvore limpa na consulta.
- `origin`: `https://github.com/NousResearch/hermes-agent.git` — dono upstream do runtime.
- `fork`: `https://github.com/cytracks-git/hermes-agent.git` — fork local de integração.
- A correção pertence ao runtime Hermes, **não** aos PRs Atlas 1490/1491 nem ao pacote V4 já instalado. Pessoa/assignee mantenedora não foi identificada; não atribuir o trabalho a um autor do git como se fosse aceite de ownership.
- Nenhum CODEOWNERS foi listado entre arquivos versionados consultados. HEAD do checkout não prova qual revisão Python o processo iniciado anteriormente importou; a coexistência foi medida ao vivo independentemente disso.

## Causa no código vigente

Caminhos abaixo relativos ao checkout consumidor:

- `agent/shell_hooks.py:141–176`: `register_from_config()` só acrescenta callbacks. Chave `(home, event, matcher, command)` em 158; `append` em 171. Trocar caminho V3→V4 produz outra chave e mantém o anterior.
- `agent/shell_hooks.py:151–153`: configuração sem specs retorna sem retirar callbacks antigos.
- `agent/shell_hooks.py:158–161`: timeout/fail_closed não entram na chave; alterações desses atributos podem manter callback obsoleto.
- `tui_gateway/server.py:2393–2396`: construir outro agente carrega config e chama registro novamente. Conversa/delegação nova não garante manager novo.
- `agent/shell_hooks.py:184–200`: recuperação após force-reload pressupõe limpeza do manager; não é reconciliação por alteração normal de config.
- `agent/shell_hooks.py:567–572`: revoke documenta explicitamente que callbacks vivos ficam até restart. Não usar como recuperação disfarçada.
- `tests/agent/test_shell_hooks.py:346–393`: cobertura de idempotência e matchers distintos, não prova substituição V3→V4.
- `tests/tui_gateway/test_profile_shell_hooks.py:11–53`: teste existente de construção/consentimento por perfil; ponto de extensão para regressão do consumidor.
- Configuração atual `~/.hermes/config.yaml:265–273`: somente V4 para o gate de card, matcher terminal/kanban_create, timeout 10, fail_closed true.

## Contrato mínimo de correção versionada — ainda não executado

1. Em worktree próprio do repositório Hermes, produzir RED para sequência config V3 → config V4 → repetição V4 no mesmo manager/home. Após troca, somente um callback de card e comando V4; callback V3 ausente.
2. Reconciliar callbacks **de propriedade da configuração** por manager/home de modo sincronizado; substituir removidos/alterados, incluindo timeout e fail_closed; não apagar hooks de plugins ou de outros perfis. Config vazia precisa remover os seus callbacks. Dois matchers legitimamente declarados continuam válidos.
3. Não executar callbacks sob o lock de reconciliação. Definir snapshot seguro para chamadas em voo; demonstrar que a troca não deixa janela sem política. Falha/recusa de consentimento não pode virar liberação silenciosa. Preservar semântica de force-reload e consentimento.
4. Estender teste do consumidor `_make_agent`, não só parser/linter: V3→V4 no mesmo processo; controles de perfil B/plugin preservados; repetição sem duplicação; negativos continuam bloqueando. Sem mockar o algoritmo que se pretende provar.
5. Commit pequeno, revisão independente, evidência RED/GREEN e integração no checkout realmente servido. Deploy/reinício requer janela separada após preservar trabalhos vivos; não é autorizado por este relatório.

## Mecanismo suportado de prova sem lixo

Documentação oficial consultada: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks — `hermes hooks test <event> --for-tool X --payload-file F`. `--help` local confirmou a sintaxe.

Comandos para os três payloads finais, **ainda não executados nesta coleta**:

```sh
/Users/farantes/.local/bin/hermes --profile default hooks test pre_tool_call --for-tool kanban_create --payload-file /Users/farantes/atlas/evidencia/v4-retomada-20261012/payload-v4-valido.json
/Users/farantes/.local/bin/hermes --profile default hooks test pre_tool_call --for-tool kanban_create --payload-file /Users/farantes/atlas/evidencia/v4-retomada-20261012/payload-v3.json
/Users/farantes/.local/bin/hermes --profile default hooks test pre_tool_call --for-tool kanban_create --payload-file /Users/farantes/atlas/evidencia/v4-retomada-20261012/payload-v4-incompleto.json
```

`hermes_cli/hooks.py:178–206` usa os hooks configurados e `run_once`; **não executa kanban_create**. Payload CLI recebe `args`, convertido para `tool_input` em `agent/shell_hooks.py:79–93`. Não passar o envelope stdin como se fosse payload CLI. Examinar `exit=` e `parsed` de cada hook, não só o exit code externo (`hermes_cli/hooks.py:210–228`). O CLI dispara todos os hooks correspondentes, que podem produzir sua própria telemetria: não é sandbox de código arbitrário. O outro gate exige teto positivo (`~/.hermes/hooks/gate-revisao-independente.mjs:563–571`), já incluído nos fixtures.

**Limite decisivo:** CLI e stdin provam julgamento em subprocesso, não o registro do PID 1394. Fonte do CLI lê config (`hermes_cli/hooks.py:194`); não inspeciona callbacks vivos. Não encontrei porta de validação do manager operacional no recorte `tui_gateway`; `--dry-run` em promote/dispatch não se transfere ao create. Portanto não inventar endpoint nem declarar o aceite operacional integral cumprido.

Para satisfazer o aceite estrito da mesma instância, o contrato do mantenedor deve escolher e documentar um modo seguro de avaliação dos hooks carregados, sem invocar o corpo da ferramenta (com perfil, PID e inventário efetivo de callbacks), **ou** declarar explicitamente uma prova de consumidor isolado e manter a prova operacional pendente até uma operação legítima autorizada. Não expor uma nova API genérica de execução só para este teste. Não usar criação seguida de delete, board alternativo que tire Atlas da jurisdição, downgrade V3 ou override.

## Próximo ato exato

Encaminhar este contrato e os recibos ao mantenedor/integrador do fork `cytracks-git/hermes-agent`, solicitando atribuição explícita de Writer/Reviewer e worktree próprio para a regressão V3→V4. Primeiros arquivos: `agent/shell_hooks.py`, `tests/agent/test_shell_hooks.py` e `tests/tui_gateway/test_profile_shell_hooks.py`. Não reinstalar o pacote V4 e não reinvestigar RAG/Runner/Rel. Nenhum card foi aberto por esta frente.
