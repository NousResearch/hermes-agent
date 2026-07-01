# RCA — Hermes recusa usar `terminal` em DMs do Telegram mesmo com `platform_toolsets.telegram` incluindo `terminal`

Status: **investigação concluída — sem fix aplicado** (por escopo).
Branch: `investigation/telegram-dm-terminal-tool-refusal`.

## 1. Resumo executivo

A exposição de tools para uma sessão de gateway (Telegram incluso) é recalculada do zero, a cada
construção/reuso de `AIAgent`, a partir de **duas fontes de config estáticas**: `platform_toolsets.<plataforma>`
e uma lista **global** `agent.disabled_toolsets`. Essa segunda lista é aplicada como uma subtração
incondicional e final, **mesmo quando o toolset já foi habilitado explicitamente pela plataforma**
(`model_tools.py:395-399`, citando o issue #17309 no próprio comentário do código). Se `"terminal"`
estiver presente em `agent.disabled_toolsets` — por qualquer caminho, incluindo o fluxo oficial de
"Blank Slate" documentado em `hermes_cli/cli_agent_setup_mixin.py:3021-3032` — o Telegram (e qualquer
outra plataforma) perde `terminal` permanentemente, sem nenhum log que diferencie "excluído pela
config da plataforma" de "excluído pelo override global". Como isso é recalculado puramente a partir
do `config.yaml` a cada turno (a tabela `sessions` do SQLite não tem NENHUMA coluna de tools/toolset —
verificado em `hermes_state.py:640-683`), o comportamento é 100% determinístico e reaparece
identicamente em sessões novas. Os logs de `FOREIGN KEY constraint failed`, `Persisted transcript
lagged` e `rebuilding from scratch` são sintomas de um problema de durabilidade de sessão **não
relacionado** (mesma classe do bug já corrigido em `investigation/telegram-gateway-self-restart-freeze`
/ PR #56036) — coincidentes no tempo, não causais.

## 2. Fluxo de criação/rebuild de sessão Telegram (ASCII diagram)

```
Mensagem Telegram (DM) chega no gateway
        │
        ▼
gateway/run.py: handler de mensagem (~L15479 / ~L16400+)
        │
        ├─ user_config = _load_gateway_config()                (yaml cacheado por mtime_ns+size)
        ├─ platform_key = _platform_config_key(source.platform)  → "telegram" (sempre; sem variante DM)
        │
        ├─ enabled_toolsets = sorted(_get_platform_tools(user_config, "telegram"))
        │        │
        │        ├─ lê platform_toolsets.telegram (explícito) OU default_toolset "hermes-telegram"
        │        ├─ resolve nomes → conjunto de toolset-keys configuráveis (inclui "terminal" se listado)
        │        └─ NÃO aplica nenhuma restrição de plataforma a "terminal"
        │                (_TOOLSET_PLATFORM_RESTRICTIONS só tem discord/discord_admin)
        │
        ├─ disabled_toolsets = user_config["agent"]["disabled_toolsets"]   ← GLOBAL, não por plataforma
        │
        ├─ _sig = _agent_config_signature(model, runtime, enabled_toolsets, ...)
        │
        ├─ cache hit? ──sim──► reusa AIAgent cacheado (agent.tools já fixado na 1ª construção)
        │        │
        │        não
        │        ▼
        └─ AIAgent(... enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets ...)
                 │
                 ▼
        agent_init.py:1037  agent.tools = get_tool_definitions(enabled_toolsets, disabled_toolsets)
                 │
                 ▼
        model_tools.py:_compute_tool_definitions
                 ├─ tools_to_include = resolve(enabled_toolsets)        → inclui "terminal"
                 └─ SEMPRE (independente de enabled_toolsets):          ← ***PONTO DA PERDA***
                      tools_to_include -= resolve(disabled_toolsets)    → remove "terminal" se lá estiver
                 │
                 ▼
        agent.tools (schema enviado à API)  ──►  NÃO contém "terminal"
                 │
                 ▼
        agent/system_prompt.py: guidance de tool é condicional a
        "terminal" ∈ agent.valid_tool_names → nenhuma menção é injetada
                 │
                 ▼
        Modelo nunca vê function "terminal" no `tools=[...]` da chamada de API
                 │
                 ▼
        Modelo responde em texto puro: "I can't access a terminal tool..."
        (tool_turns=0 — nenhuma tool_call foi de fato tentada)
```

Em paralelo, e **sem interseção com o caminho acima**, existe o mecanismo de sessão/DB:

```
agent/conversation_loop.py: restore_or_build_system_prompt()
        │
        ├─ lê sessions.system_prompt (coluna TEXT, hermes_state.py:650)
        ├─ NULL/empty/stale → loga "Stored system prompt ... rebuilding from scratch"
        └─ SÓ reconstrói o TEXTO do system prompt (cache de prefixo de custo),
           nunca toca em agent.tools / enabled_toolsets

gateway/run.py (~L16846-16864): compara message_count persistido vs. cache em memória
        └─ diverge → "Persisted transcript lagged live cached history" (guarda de corrupção de FTS,
           preserva o histórico em memória — também não toca em tools)

hermes_state.py: messages.session_id REFERENCES sessions(id)
        └─ append_message em session_id inexistente/evictado → "FOREIGN KEY constraint failed"
           (schema de sessions/messages não tem NENHUMA coluna de tools/toolset)
```

## 3. Ponto exato onde a tool `terminal` é perdida

`model_tools.py:395-399`, dentro de `_compute_tool_definitions`:

```python
# Always apply disabled toolsets as a subtraction step at the end.
# This ensures that even if a composite toolset (like hermes-cli)
# is enabled, any tools belonging to a disabled toolset are strictly
# stripped out. See issue #17309.
if disabled_toolsets:
    for toolset_name in disabled_toolsets:
        ...
        tools_to_include.difference_update(resolved)
```

Isso contradiz a docstring pública da função vizinha `get_tool_definitions` (linha 292):

```python
disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
```

Ou seja: a **documentação da API interna diz** que `disabled_toolsets` só importa quando
`enabled_toolsets` é `None`; o **código real** aplica `disabled_toolsets` incondicionalmente, sempre,
por cima de qualquer `enabled_toolsets` — inclusive um `enabled_toolsets` que veio de
`platform_toolsets.telegram` explicitamente contendo `"terminal"`.

## 4. Causa raiz (mecanismo, não sintoma)

**A resolução de tools do gateway trata `agent.disabled_toolsets` (lista global, seção `agent:` do
`config.yaml`, não escopada por plataforma) como um veto absoluto e silencioso sobre
`platform_toolsets.<plataforma>`.** Isso é uma decisão de design explícita e documentada — não é um
bug de digitação isolado:

- `model_tools.py:395-399` implementa a subtração incondicional, citando o próprio issue #17309 como
  justificativa ("mesmo se um toolset composto está habilitado, tools de um toolset desabilitado devem
  ser removidas").
- `hermes_cli/cli_agent_setup_mixin.py:3025-3031` documenta o mesmo comportamento do ponto de vista do
  operador: *"`agent.disabled_toolsets` — a global hard-suppression list (applied last in
  `_get_platform_tools`, overriding every other path ...)"* e instrui que o único jeito de reverter é
  "editing `agent.disabled_toolsets`" diretamente — **não** basta ajustar `platform_toolsets` pela
  plataforma.
- `hermes_cli/setup.py` (fluxo de "Blank Slate") é um caminho **real e já existente no produto** que
  escreve em `agent.disabled_toolsets` a partir de uma ação com escopo aparente de uma única
  plataforma (`platform_toolsets["cli"]`), mas cujo efeito colateral (a lista de disabled) é
  **global** e alcança todas as demais plataformas, Telegram incluído.

Se `"terminal"` está nessa lista global — por edição manual, por um "Blank Slate" anterior (rodado
para `cli` ou qualquer outro contexto onde o operador não quis manter terminal), ou por qualquer
outro escritor de `agent.disabled_toolsets` — toda sessão de toda plataforma perde `terminal` de forma
permanente e silenciosa, **independente do que `platform_toolsets.telegram` diga**. Como o cálculo é
100% função do `config.yaml` atual (sem nenhum estado de sessão no meio), o resultado é determinístico:
mesma entrada → mesma saída, sempre.

**Confirma-se que isso não é um efeito de estado de sessão “presa”**: a tabela `sessions` (
`hermes_state.py:640-683`) e a tabela `messages` (`hermes_state.py:685-705`) não têm nenhuma coluna
relacionada a tools/toolsets — `agent.tools` nunca é persistido, é recalculado do zero em toda
construção de `AIAgent` (`agent_init.py:1037`), a partir do config vigente naquele instante.

## 5. Por que persiste mesmo em sessão fresh (não é só cache antigo)

Uma sessão "fresh" limpa apenas o histórico de mensagens/transcript e o `system_prompt` persistido —
nunca toca em `agent.disabled_toolsets` nem em `platform_toolsets`, que vivem em `config.yaml`,
completamente fora do ciclo de vida da sessão. A cada construção de `AIAgent` (cache hit ou miss —
`enabled_toolsets`/`disabled_toolsets` fazem parte da assinatura de cache em
`gateway/run.py:_agent_config_signature`, então uma mudança neles já invalidaria o cache
corretamente), o mesmo `agent.disabled_toolsets` global é lido do mesmo arquivo e produz o mesmo
resultado. Isso descarta por completo as classes de hipótese "cache de sessão corrompido" e "rebuild
usa fallback obsoleto" — o comportamento nem depende de rebuild nenhum: a subtração de `disabled_toolsets`
roda de forma idêntica tanto no caminho de "sessão nova" quanto no de "sessão recuperada".

## 6. Conexão com FK / transcript lag / rebuilding from scratch: causa ou sintoma colateral?

**São sintomas de uma causa completamente diferente — não têm relação causal com a perda de `terminal`.**
Evidências estruturais:

- `restore_or_build_system_prompt` (`agent/conversation_loop.py:~250-336`) só lê/escreve a coluna
  `sessions.system_prompt` (uma string). O rebuild que ela dispara (`agent._cached_system_prompt =
  agent._build_system_prompt(...)`) reconstrói apenas o **texto** do system prompt para fins de
  cache de prefixo de custo (ver comentário longo em `agent/turn_context.py:146-152` sobre o
  issue #45499). Em nenhum ponto esse caminho toca `agent.tools`, `enabled_toolsets` ou
  `disabled_toolsets` — esses já foram fixados antes, na construção do `AIAgent`.
- `"FOREIGN KEY constraint failed"` vem de `messages.session_id REFERENCES sessions(id)`
  (`hermes_state.py:687`) — um `append_message` para uma sessão que não existe (ou foi evictada) na
  tabela `sessions`. Não existe FK nem coluna relacionada a tools em nenhuma das duas tabelas.
- `"Persisted transcript lagged live cached history"` (`gateway/run.py:16846-16864`) é uma guarda
  contra corrupção de gravação por FTS5 — ela decide qual histórico de **mensagens** usar, não qual
  lista de **tools**.

Ambos os sintomas (FK e lag) são, isso sim, indícios de um problema real e independente de
durabilidade/concorrência na sessão do gateway do Telegram — da mesma família do bug de reconexão já
identificado e corrigido em `investigation/telegram-gateway-self-restart-freeze` (PR #56036). Eles
devem continuar sendo monitorados, mas **não explicam** a ausência de `terminal` nas tools do modelo.

## 7. Por que o cronfallback funciona (caminho diferente de tools)

`cron/scheduler.py` resolve tools por um par de funções **totalmente separado** do caminho do
gateway de mensagens: `_resolve_cron_enabled_toolsets` / `_resolve_cron_disabled_toolsets`
(`cron/scheduler.py:115-188`), chamado a partir de um `AIAgent(...)` próprio (`cron/scheduler.py:2478+`).
Note que esse caminho **também** lê e aplica `agent.disabled_toolsets` global
(`_resolve_cron_disabled_toolsets`, linha ~123-130: *"User-level `agent.disabled_toolsets` ... is
layered on top so per-job `enabled_toolsets` cannot bypass policy"*) — ou seja, se `"terminal"`
estivesse de fato na lista global, um job de cron que dependesse do modelo chamar a tool `terminal`
seria igualmente afetado.

A evidência do relator ("Cronfallback consegue executar `whoami` na mesma máquina") foi descrita como
prova de que **o backend do terminal funciona** — não como prova de que um agente LLM do cron chamou
a tool com sucesso. Isso é consistente com o mecanismo aqui descrito: `check_terminal_requirements()`
(`tools/terminal_tool.py:2749+`) testa apenas a *capacidade* do backend (local/docker/ssh/modal) de
executar comandos, e essa checagem independe inteiramente de `platform_toolsets`/`disabled_toolsets`.
Ou seja, "o backend responde" e "o schema `terminal` chega até o modelo em uma sessão de chat" são
duas coisas diferentes, verificadas por dois mecanismos diferentes — não há contradição entre
"cron prova que o backend funciona" e "o Telegram nunca recebe o schema".

## 8. Alcance: é genérico a outros gateways, não específico do Telegram

`plugins/platforms/telegram/adapter.py` não contém nenhuma referência a toolsets/tools — a resolução
inteira acontece em código compartilhado (`gateway/run.py` + `hermes_cli/tools_config.py` +
`model_tools.py`), chamado da mesma forma para qualquer `source.platform` (Discord, Slack, WhatsApp,
Mattermost, etc., via `_platform_config_key`/`_get_platform_tools` genéricos). **Qualquer plataforma
cujo operador tenha, em algum momento, deixado `"terminal"` (ou qualquer outro toolset) entrar em
`agent.disabled_toolsets` sofre exatamente o mesmo efeito.** O relato ter sido observado "só" no
Telegram muito provavelmente reflete que foi a única plataforma onde o operador testou explicitamente
`platform_toolsets.<plataforma>` contendo `"terminal"` — não que exista tratamento especial de Telegram
no código.

## 9. Soluções possíveis (ordenadas por impacto) — não implementadas nesta investigação

**Baixo impacto**
- Logar, no momento em que `_compute_tool_definitions` faz a subtração de `disabled_toolsets`
  (`model_tools.py:399-426`), quais tool-names foram removidos **e por qual toolset desabilitado**,
  em nível INFO (hoje só imprime via `print()` quando `quiet_mode=False` — o caminho do gateway roda
  com `quiet_mode=True`, então essa informação nunca chega a `agent.log`).
- Ao construir `agent.tools` em `agent_init.py:1037`, logar (nível INFO, sempre, mesmo em quiet_mode)
  a lista final de nomes de tools resolvida para a sessão, junto com `enabled_toolsets` e
  `disabled_toolsets` de entrada — permitindo correlacionar diretamente "o que a plataforma pediu"
  vs. "o que efetivamente chegou ao modelo".
- Corrigir a docstring de `get_tool_definitions` (`model_tools.py:292`) para refletir o
  comportamento real (subtração sempre aplicada), evitando que o próprio texto do código continue
  induzindo a leitura errada de que `disabled_toolsets` é ignorado quando `enabled_toolsets` está
  presente.

**Médio impacto**
- Adicionar um aviso explícito (log WARNING ou saída de `hermes tools` / `hermes doctor`) quando
  `platform_toolsets.<plataforma>` lista um toolset que também aparece em `agent.disabled_toolsets`
  global — hoje esse conflito é resolvido silenciosamente a favor do override global, sem qualquer
  sinalização ao operador no momento da configuração.
- Expor, em `hermes tools show <platform>` (ou comando equivalente), o resultado *pós-subtração*
  (o que realmente vai para o modelo) lado a lado com o `platform_toolsets` bruto, para que a
  divergência fique visível sem precisar ler logs.

**Alto impacto (só se estritamente necessário)**
- Revisitar a decisão de design do issue #17309 de que `disabled_toolsets` deve sempre vencer
  `enabled_toolsets`/`platform_toolsets`, avaliando se persiste a necessidade de veto global absoluto
  ou se ele deveria ser explicitamente escopável por plataforma (o que exigiria migração de schema de
  config e revisão de todos os chamadores que hoje dependem do "vence sempre").

## 10. Pontos onde adicionar logs para validar/descartar a hipótese (requer confirmação)

Esta causa raiz é a mais bem sustentada pelas evidências de código encontradas, mas **depende de um
dado que não está disponível nesta investigação**: o conteúdo real de `agent.disabled_toolsets` no
`config.yaml` do ambiente onde o bug foi reportado. Recomenda-se, antes de qualquer fix:

1. Inspecionar diretamente `~/.hermes/config.yaml` do ambiente afetado — chave `agent.disabled_toolsets`
   — e verificar se `"terminal"` (ou um toolset composto que o inclua, como `"hermes-cli"`, cuja
   subtração de bundle é tratada à parte em `model_tools.py:402-421`) está presente.
2. Adicionar temporariamente um log em `model_tools.py:399` (antes do loop de subtração) imprimindo
   `disabled_toolsets` e `tools_to_include` antes/depois — isso confirma ou descarta o mecanismo em
   minutos, sem precisar reproduzir a falha de ponta a ponta.
3. Adicionar log em `agent_init.py:1037` com o `agent.tools` final resolvido por sessão Telegram,
   comparando com o `enabled_toolsets` de entrada vindo de `gateway/run.py:15483`/`16594` — uma
   divergência ali (`enabled_toolsets` contém "terminal" mas `agent.tools` não) confirma
   definitivamente o ponto exato descrito na Seção 3.

## Anexos — trechos relevantes

**`model_tools.py:279-302` (contrato documentado, incorreto na prática):**
```python
def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    ...
    Args:
        enabled_toolsets: Only include tools from these toolsets.
        disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
```

**`model_tools.py:395-399` (comportamento real):**
```python
# Always apply disabled toolsets as a subtraction step at the end.
# This ensures that even if a composite toolset (like hermes-cli)
# is enabled, any tools belonging to a disabled toolset are strictly
# stripped out. See issue #17309.
if disabled_toolsets:
    ...
```

**`hermes_cli/cli_agent_setup_mixin.py:3021-3031` (o próprio produto documenta o override global):**
```python
1. ``platform_toolsets["cli"] = ["file", "terminal"]`` — an explicit list of
   configurable keys, which the resolver treats as authoritative
   (``has_explicit_config``) so default toolsets aren't re-expanded.
2. ``agent.disabled_toolsets`` — a global hard-suppression list (applied last
   in ``_get_platform_tools``, overriding every other path including the
   non-configurable platform-toolset recovery that would otherwise re-add
   toolsets like ``kanban``).
```

**`hermes_state.py:640-705` (schema — nenhuma coluna de tools em `sessions`/`messages`):**
```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    ...
    system_prompt TEXT,
    parent_session_id TEXT,
    ...
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    ...
);
```

**`cron/scheduler.py:115-130` (mesmo override global, caminho independente do gateway):**
```python
def _resolve_cron_disabled_toolsets(cfg: dict) -> list[str]:
    """
    User-level ``agent.disabled_toolsets`` from config.yaml is layered on top
    so per-job ``enabled_toolsets`` cannot bypass policy that applies to
    ordinary agent runs (#25752 — LLM-supplied enabled_toolsets was widening
    ...
    """
```
