# Dependência invertida — evidência do executor

Card: `t_19aa0a8b`. Candidato para revisão, NÃO aprovado, NÃO instalado no runtime.
Base upstream: `8ae1dc11a0b0197a057459f21fcbd8d4175acfe7`.

## Recorte e decisão

Roteamento: gpt-6-astra/openai-codex, executor nativo; assinatura/OAuth,
USD/BRL NÃO MEDIDO. Nenhum agente externo iniciado pelo shell.
RACI: executor implementa; revisor independente responde por QA/revisão;
orquestrador/PM decide escopo; Rel congela depois da revisão; H1 aceita.
Não houve alteração de schema, dispatcher, configuração, credencial ou aresta real.
A cláusula final do card exclui corrigir em massa as suspeitas encontradas.

Aviso `suspected_inverted_dependency`, `warning`, um por aresta, no filho.
O grafo existente agora carrega assignee/body para o motor, sem consulta por aresta.
Precedência: assignee exato revisor/reviewer/executor/implementer; título;
corpo se título não indica papel. PT/EN, sem acentos/case, palavras inteiras.
Texto misturando os dois papéis não decide. O aviso expõe ids e sinais utilizados,
ressalva auditoria prévia legítima e sugere conferir a intenção — nunca inverter
automaticamente. Perfis renomeados e frases fora do vocabulário podem escapar;
menções incidentais e mudanças de assignee para revisão podem gerar suspeitas.
Ausência de aviso NÃO é prova de topologia correta ou de inexistência de deadlock.

## Prova focal e alcance

Dois testes parametrizados, 9 casos: quatro atravessam SQLite real e o handler de
`kanban diagnostics --json` (título PT, título EN, corpo, assignee); cinco protegem
silêncio quando o papel/grafo não sustenta o aviso. Cada integração cria dois cards,
inverte a aresta, mede o aviso, corrige a ordem e mede silêncio. `iterdump()` antes
/depois de cada diagnóstico exige que nem tabelas nem eventos mudem.

Justificativa: o risco é sensor cego ou alerta que acusa tudo. Por isso controle
positivo + negativo e mutante, não bateria de segurança/rede alheia ao recorte.
Os 5 testes existentes do motor também foram executados. NÃO MEDIDO: suíte global,
Desktop/browser, produção instalada, precisão semântica universal. O alvo solicitado
é a CLI do Hermes; atlas.local e TLS não fazem parte desta mudança.

Saídas reais:

    BASE: 4 failed, 5 passed in 3.53s; rc=1
    assert len(warnings) == 1
    E assert 0 == 1

    NOVO: 2 files, 14 tests passed, 0 failed (100% complete) in 4.4s
    REGENERAÇÃO python:3.12-slim: 14 tests passed, 0 failed ... in 5.7s
    MUTANTE (motor antigo montado sobre o novo grafo):
    1 files, 5 tests passed, 4 failed ... in 6.2s; rc=1

O mutante usa o arquivo de diagnóstico da base, não uma imitação defeituosa.
Os mesmos quatro casos de aviso falham; os cinco casos de silêncio passam.
Os containers rodam não-root (`-u 501:20`) e são removidos com `--rm`.
Restauração final: `14 tests passed, 0 failed ... in 2.7s`, rc=0.
Gitleaks pré-commit no diff staged: `scanned ~20374 bytes (20.37 KB) in 94.6ms`,
`no leaks found`, rc=0. `git diff --check` sem saída, rc=0.

### Duas execuções pelo handler real, sem mock

Prova manual adicional executa o mesmo handler para mostrar o JSON (o runner
canônico resume a saída dos testes verdes). Extratos literais da saída:

    INVERTED
    "task_id": "t_8a84f9d2",
    "title": "Executar a matriz E2E",
    "status": "todo",
    "kind": "suspected_inverted_dependency",
    "severity": "warning",
    "data": {
      "parent_id": "t_eeb0583a",
      "child_id": "t_8a84f9d2",
      "parent_signal": "assignee=revisor",
      "child_signal": "assignee=executor"
    }
    CORRECTED
    [
      {
        "task_id": null,
        "dispatch_profiles": "any",
        "diagnostics": []
      }
    ]

A linha `task_id: null` é metadado da CLI, não card. Nenhum card sintético
foi criado no board atlas. Banco de teste morreu junto com o container.

## Receita de regeneração

No checkout candidato (não no pin em uso), container novo e venv novo:

```sh
docker run --rm -u "$(id -u):$(id -g)" \
  -e HOME=/tmp/proof-home -e HERMES_TEST_FILE_RETRIES=0 \
  -v "$PWD":/work -w /work python:3.12-slim bash -c '
set -eu
python -m venv /tmp/proof-venv
/tmp/proof-venv/bin/pip install -q pytest==8.3.4 pytest-asyncio==1.3.0 pytest-timeout==2.4.0 python-dotenv==1.2.2 PyYAML==6.0.3 rich==14.3.3 prompt_toolkit==3.0.52 httpx==0.28.1 croniter==6.0.0 psutil==7.2.2 pathspec==1.1.1 openai==2.24.0 tiktoken==0.12.0 websockets==15.0.1 requests==2.33.0
HERMES_PYTHON=/tmp/proof-venv/bin/python bash scripts/run_tests.sh tests/hermes_cli/test_kanban_inverted_dependencies.py tests/hermes_cli/test_kanban_diagnostics.py
'
```

Receita executada, rc=0. O runner avisou `git: command not found` no passo
opcional de pré-compilação; a coleta/execução dos 14 casos ocorreu. Na imagem
local `hermes-launch:t_e9aedb24` a mesma etapa avisa sobre o ponteiro `.git` do
worktree fora do mount; não foi usado como prova de integridade Git.
A imagem local também avisa sobre SQLite 3.46.1 e adota `journal_mode=DELETE`;
não foi suprimido o aviso nem alterado o runtime do usuário.

Mutante: extrair `git show 8ae1dc11a0:hermes_cli/kanban_diagnostics.py` para um
arquivo fora do checkout e repetir o runner com esse arquivo montado read-only
sobre `/work/hermes_cli/kanban_diagnostics.py`. Remover o mount restaura o novo;
nenhum arquivo do checkout é substituído. O runner é sempre `scripts/run_tests.sh`.

## Varredura do atlas

Fonte real: `~/.hermes/kanban/boards/atlas/kanban.db`. Snapshot consistente por
`sqlite3 -readonly <fonte> '.backup <workspace>/atlas-snapshot.db'`.
O sensor foi executado em Docker não-root sobre o snapshot montado `:ro`,
URI `file:/board.db?mode=ro&immutable=1`. `immutable` SOMENTE na cópia fechada,
nunca no banco vivo com WAL. Uma tentativa inicial sem `immutable` falhou com
`attempt to write a readonly database`; não foi contada como varredura verde.

Consulta e chamada efetivamente utilizadas:

```python
import sqlite3
from hermes_cli import kanban_db as kb, kanban_diagnostics as kd
c = sqlite3.connect('file:/board.db?mode=ro&immutable=1', uri=True)
c.row_factory = sqlite3.Row
rows = c.execute('SELECT * FROM tasks ORDER BY id').fetchall()
graphs = kb.task_graph_contexts(c, [r['id'] for r in rows])
for row in rows:
    for diagnostic in kd.compute_task_diagnostics(row, [], [], graph=graphs[row['id']]):
        if diagnostic.kind == 'suspected_inverted_dependency':
            print(diagnostic.data)
```

Resultado: 326 cards, 294 arestas, 55 pares suspeitos em 0,0848 s de consulta+
classificação (não inclui startup Docker). Filhos: archived=27, done=20,
triage=5, todo=3. Varredura inclui histórico para não esconder pares; a CLI
fleet existente exclui filhos archived por contrato. Suspeita NÃO é inversão
confirmada. Em especial, revisão antes de merge e auditoria antes de implementação
podem ser corretas. A lista inteira segue; JSON com títulos/sinais preservado
como artefato do card (`board-suspects.json`).

| Pai | Estado pai | Filho | Estado filho |
|---|---|---|---|
| t_37dc47cd | done | t_07ee5746 | archived |
| t_dbdcb368 | done | t_0a573073 | done |
| t_d2f26990 | done | t_133b9955 | done |
| t_e1e5b78b | done | t_133b9955 | done |
| t_07314874 | done | t_16dd0e9b | triage |
| t_72836bf7 | done | t_16dd0e9b | triage |
| t_aff2c258 | done | t_16dd0e9b | triage |
| t_c3d0361f | done | t_16dd0e9b | triage |
| t_18962b68 | triage | t_1764ae5a | todo |
| t_8df36343 | done | t_1764ae5a | todo |
| t_c061bb8d | done | t_1f1baecb | archived |
| t_9b5cc8c2 | done | t_23540d11 | done |
| t_f598827e | done | t_26a3e7f2 | archived |
| t_60e0d393 | archived | t_2da1ef88 | done |
| t_ad767a89 | done | t_2da1ef88 | done |
| t_fc9133bc | done | t_319551c5 | archived |
| t_9b5cc8c2 | done | t_3ad80919 | done |
| t_4242ee93 | done | t_450f4511 | archived |
| t_fdb17481 | done | t_496af165 | archived |
| t_83a2613b | done | t_4ac41f27 | done |
| t_3c03fb9b | done | t_4eb2a1e9 | triage |
| t_1a8c64b2 | done | t_71816911 | archived |
| t_83a2613b | done | t_7d26372d | done |
| t_161df850 | done | t_817f1a5b | archived |
| t_fdb17481 | done | t_8523c45f | archived |
| t_e5fe1106 | done | t_93621643 | done |
| t_11511bbd | done | t_95203088 | archived |
| t_26d48111 | done | t_95203088 | archived |
| t_03aac2e3 | blocked | t_a4c2bfbd | todo |
| t_8407f1e5 | archived | t_a5ccb7b3 | archived |
| t_18962b68 | triage | t_afa62da5 | done |
| t_8df36343 | done | t_afa62da5 | done |
| t_00542312 | done | t_b26c1871 | done |
| t_0fe17f2b | archived | t_b2d66bdb | archived |
| t_568c0860 | archived | t_b62bcb1e | archived |
| t_123bbe10 | done | t_b97e7a5e | archived |
| t_161df850 | done | t_b97e7a5e | archived |
| t_1a8c64b2 | done | t_b97e7a5e | archived |
| t_3a825571 | done | t_b97e7a5e | archived |
| t_b2cd6abd | done | t_b97e7a5e | archived |
| t_bc4daef5 | done | t_b97e7a5e | archived |
| t_ff978f17 | archived | t_b97e7a5e | archived |
| t_dbdcb368 | done | t_c6356162 | done |
| t_00b18862 | done | t_c7bee8cc | done |
| t_af9e17d8 | done | t_c7bee8cc | done |
| t_e1178d05 | archived | t_c92b96ec | archived |
| t_161df850 | done | t_caefba67 | archived |
| t_e010322c | archived | t_cb49f647 | archived |
| t_1a7ea237 | done | t_d0643701 | archived |
| t_daa12d87 | done | t_dd690534 | archived |
| t_aff2c258 | done | t_ddba4966 | done |
| t_c3d0361f | done | t_ddba4966 | done |
| t_daa12d87 | done | t_e1249364 | archived |
| t_2353075c | done | t_e9aedb24 | done |
| t_9b5cc8c2 | done | t_e9aedb24 | done |

Entre filhos abertos, só dois pais sinalizados ainda estão abertos:
`t_18962b68 -> t_1764ae5a` (arbitragem de integração antes da matriz) e
`t_03aac2e3 -> t_a4c2bfbd` (canal de entrega antes do aviso). Ambos pedem decisão
semântica do orquestrador, não inversão automática. Os demais pais dos oito pares
com filho aberto já estão done: não são bloqueio vivo por essa aresta.
O par de origem está na direção corrigida `t_1764ae5a -> t_2824a052` e não acusa.

## Operação e régua erguida

Pin medido: branch `t78-c28-pin`, SHA `587e13cbf9c0fdb1725416dd2cf7f8c5259acbb1`;
`git rev-list --count HEAD..origin/main` = 1730. Busca no arquivo do pin:
0 ocorrências de `suspected_inverted_dependency` e `stranded_in_triage`.
NÃO ativo no runtime. Instalação/restart pertence a release, não a este card.

Limite medido da heurística pedida: assignee atual também pode ser revisor por
handoff de implementação; 55 suspeitas não significam 55 erros. Próxima decisão
sugerida ao orquestrador: conferir apenas os dois pares com ambos os lados abertos
antes de discutir modelo de papel estável. Custo estimado: leitura de dois pares;
risco de auto-inversão: alto. Não foi aberta frente adicional nem alterada aresta.

DGAP + Brain: Discover lista pares/sinais; Govern exige interpretação humana;
Act não muda dependências; Prove exerce CLI/SQLite, ordem correta e mutante;
Brain preserva receita, saídas e limites. Isso é candidato revisável, não freeze
nem declaração de segurança/qualidade global.
