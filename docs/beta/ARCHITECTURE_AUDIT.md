# Auditoria de Arquitetura — Hermes para Beta

## Objetivo

Mapear os componentes existentes do Hermes Agent que serão reutilizados, adaptados ou restringidos para transformar o agente principal no Beta Orquestrador.

O Beta será a interface única do chefe. Ele interpreta intenção, classifica risco, escolhe especialistas, delega, acompanha e consolida resultados. A execução técnica deve permanecer nos agentes especialistas.

## Resumo executivo

O Hermes já possui a maior parte da infraestrutura necessária:

- loop de agente com tool calling;
- registro modular de ferramentas;
- subagentes com contexto isolado;
- execução paralela de delegações;
- perfis e papéis de subagentes;
- Kanban compartilhado;
- modo explícito de orquestrador no Kanban;
- callbacks e filas de aprovação;
- memória por providers, incluindo Hindsight;
- gateway para Telegram, Discord, Slack e outros canais;
- skills, cron e recuperação de sessões.

A estratégia correta não é substituir o runtime. Devemos criar uma camada de política do Beta sobre o runtime existente.

## 1. Pontos de entrada

### `hermes_cli.main:main`

Entrada principal do comando `hermes`. Deve continuar existindo durante a primeira fase para manter compatibilidade com o Hermes original.

### `run_agent:main`

Entrada direta do agente. O arquivo `run_agent.py` contém a classe `AIAgent` e coordena:

- ciclo de mensagens;
- chamadas ao modelo;
- despacho de ferramentas;
- histórico e persistência;
- memória;
- compactação de contexto;
- guardrails;
- execução paralela de ferramentas.

Decisão: não reescrever esse loop. Introduzir a política do Beta por composição e configuração.

## 2. Construção do prompt

Arquivo principal: `agent/prompt_builder.py`.

Pontos relevantes:

- `DEFAULT_AGENT_IDENTITY` define atualmente o Hermes como assistente geral e executor;
- `AIAgent._build_system_prompt()` combina identidade, skills, contexto, memória e prompts efêmeros;
- arquivos `.hermes.md`, `HERMES.md`, `SOUL.md` e outros contextos podem alterar o comportamento por projeto;
- existe varredura contra prompt injection antes de carregar arquivos de contexto.

Mudança planejada:

1. Criar uma identidade própria `BETA_AGENT_IDENTITY`.
2. Permitir selecionar o modo `hermes` ou `beta` por configuração.
3. Injetar a política do Beta no início do system prompt.
4. Manter o restante da montagem de prompt do Hermes.

Não devemos renomear globalmente Hermes para Beta no primeiro ciclo. Isso dificultaria acompanhar atualizações upstream.

## 3. Registro e despacho de ferramentas

Arquivos principais:

- `model_tools.py`;
- `tools/registry.py`;
- `toolsets.py`;
- módulos individuais dentro de `tools/`.

O registro é modular. Cada ferramenta registra schema, handler e metadados. `model_tools.py` apenas dispara descoberta e fornece a API pública.

Decisão:

- preservar o registro atual;
- criar um filtro de ferramentas do Beta;
- separar ferramentas de coordenação das ferramentas de execução;
- impedir que o agente principal use ferramentas técnicas mutantes diretamente;
- permitir leitura direta apenas quando a política autorizar;
- manter ferramentas de execução disponíveis para agentes especialistas.

## 4. Delegação e subagentes

Arquivo principal: `tools/delegate_tool.py`.

O Hermes já implementa:

- subagentes com conversa nova e sem histórico do pai;
- task ID e terminal isolados;
- toolsets restritos;
- execução única e em lote paralelo;
- papéis `leaf` e `orchestrator`;
- profundidade de delegação configurável;
- observabilidade dos subagentes ativos;
- interrupção individual;
- bloqueio padrão de ferramentas perigosas em filhos;
- callback de aprovação seguro por padrão.

Ferramentas bloqueadas nos filhos incluem memória compartilhada, interação com usuário, agendamento e mensagens externas.

Decisão:

O `delegate_task` será o primeiro mecanismo de especialistas do Beta. Não será criado outro barramento de agentes no MVP.

Adaptações:

1. Criar catálogo de perfis especialistas.
2. Fazer o roteador retornar um ou mais perfis.
3. Usar `delegate_task` em modo paralelo quando não houver dependência.
4. Exigir retorno estruturado de cada especialista.
5. Consolidar as respostas no agente principal.
6. Manter recursão desabilitada por padrão para especialistas comuns.

## 5. Kanban e tarefas persistentes

O prompt já define um protocolo Kanban e contém um modo de orquestrador explícito:

- o planejador cria cartões filhos;
- cada cartão recebe um especialista real;
- dependências são representadas por pais;
- o orquestrador não executa o trabalho;
- agentes completam ou bloqueiam tarefas com handoff estruturado.

Decisão:

Existirão dois caminhos:

- `delegate_task`: trabalhos curtos e síncronos dentro de uma conversa;
- Kanban: projetos, tarefas longas, dependências, revisão e continuidade entre execuções.

## 6. Aprovação e risco

O Hermes já possui callbacks de aprovação no terminal e fila de aprovação em sessões do gateway. Subagentes negam comandos perigosos por padrão, salvo configuração explícita de autoaprovação.

Mudança planejada:

Criar uma política central do Beta com três níveis:

- baixo: leitura, análise, planejamento;
- médio: geração ou preparação de alterações;
- alto: alteração de estado, produção, exclusão, deploy, firewall, banco, usuários e permissões.

O Beta poderá autorizar automaticamente apenas operações de baixo risco. Operações de alto risco devem retornar ao chefe antes de serem delegadas para execução.

## 7. Memória

Arquivo principal: `agent/memory_manager.py`.

O `MemoryManager` já fornece um ponto único para:

- registrar provider externo;
- injetar contexto no system prompt;
- prefetchar memória antes do turno;
- sincronizar memória após o turno;
- expor ferramentas do provider;
- usar Hindsight como provider opcional.

Decisão:

No MVP:

- Beta usa memória do chefe e decisões organizacionais;
- subagentes não escrevem na memória compartilhada;
- memória técnica detalhada será separada por perfil em uma fase posterior;
- Hindsight permanece como provider do Beta;
- somente o Beta decide o que vira memória durável.

## 8. Componentes a preservar

Preservar sem fork interno desnecessário:

- providers de modelos;
- transports de chat completion;
- gateway de mensageria;
- TUI e dashboard;
- skills;
- cron;
- sessão e compactação de contexto;
- terminal backends;
- MCP;
- plugins;
- segurança de contexto;
- sistema de aprovação existente.

## 9. Componentes novos do Beta

Estrutura proposta:

```text
beta/
├── identity.py
├── policy.py
├── router.py
├── risk.py
├── specialist_registry.py
├── delegation.py
├── consolidation.py
└── schemas.py
```

Responsabilidades:

### `identity.py`

Identidade do Beta e regras absolutas.

### `policy.py`

Decide se o Beta responde, consulta, delega ou pede autorização.

### `router.py`

Classifica a intenção e escolhe especialistas.

### `risk.py`

Classifica risco da tarefa e das ferramentas solicitadas.

### `specialist_registry.py`

Catálogo de perfis, capacidades, toolsets e restrições.

### `delegation.py`

Adaptador sobre `delegate_task` e Kanban.

### `consolidation.py`

Valida e combina respostas dos especialistas.

### `schemas.py`

Contratos estruturados para plano, tarefa, evidência e resultado.

## 10. Primeira implementação

### Marco BETA-001 — Modo de identidade

- adicionar configuração `agent.mode: hermes|beta`;
- criar identidade do Beta;
- carregar identidade sem alterar o comportamento padrão do Hermes;
- criar testes de seleção de identidade.

### Marco BETA-002 — Registro de especialistas

Perfis iniciais:

- infra;
- dba;
- security;
- monitoring;
- dev;
- devops;
- memory;
- qa.

Cada perfil deve definir:

- descrição;
- capacidades;
- toolsets permitidos;
- ferramentas bloqueadas;
- permissão de escrita;
- limite de profundidade;
- formato esperado de retorno.

### Marco BETA-003 — Roteador inicial

Entrada:

- mensagem do chefe;
- contexto resumido;
- especialistas disponíveis.

Saída estruturada:

```json
{
  "intent": "diagnosis",
  "risk": "low",
  "needs_delegation": true,
  "specialists": ["dba", "infra", "monitoring"],
  "parallel": true,
  "needs_approval": false,
  "tasks": []
}
```

### Marco BETA-004 — Primeiro cenário ponta a ponta

Pedido:

`Verifique por que o PostgreSQL está lento.`

Fluxo esperado:

1. Beta classifica como diagnóstico de baixo risco.
2. Seleciona DBA, Infra e Monitoring.
3. Cria delegações somente leitura.
4. Executa em paralelo.
5. Recebe evidências estruturadas.
6. Consolida causa provável e divergências.
7. Recomenda próximo passo.
8. Solicita autorização antes de qualquer alteração.

## 11. Riscos técnicos

### Acoplamento em `run_agent.py`

O arquivo continua grande e concentra vários pontos. Modificações diretas devem ser mínimas.

Mitigação: criar módulos `beta/*` e adicionar hooks estreitos.

### Atualizações upstream

Renomeações e alterações globais dificultariam sincronização com o Hermes oficial.

Mitigação: manter o runtime e a marca Hermes internamente durante o MVP; Beta entra como modo e camada de política.

### Delegação síncrona

`delegate_task` bloqueia o pai até os filhos terminarem.

Mitigação: usar para tarefas curtas; usar Kanban para trabalhos longos.

### Memória contaminada

Misturar memória do chefe com detalhes técnicos de todos os especialistas aumenta custo e ruído.

Mitigação: escrita centralizada pelo Beta e separação futura por perfil.

## 12. Decisão arquitetural

O Beta será implementado como uma camada de orquestração sobre o Hermes, e não como uma reescrita.

Princípio:

```text
Hermes Runtime
    +
Beta Identity
    +
Beta Policy
    +
Specialist Registry
    +
delegate_task / Kanban
    +
Approval Gate
    =
Beta Orchestrador
```

## Próximo passo

Implementar o marco BETA-001: modo configurável `agent.mode`, identidade própria do Beta e testes, preservando o Hermes como comportamento padrão.