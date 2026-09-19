# Orcamento de repeticao por cota

## Escopo e autoria

Autor: Hermes/GPT-6 Astra. Mudanca local solicitada pelo H1 para encerrar horas
de repeticao. QA independente e aceite humano NAO sao substituidos por estes
testes. Nenhum worker existente deve ser interrompido para aplicar a mudanca.

## Regra

- Primeira falha de cota: cooldown ja existente, uma repeticao automatica.
- Segunda falha consecutiva: `blocked/capability`, motivo visivel, sem nova
  tentativa e sem consumir o contador de falhas do trabalho.
- O limite continua valendo com cooldown configurado como zero.
- `ready` e `review` preservam sua fase na retomada.
- `unblock` explicito permite uma sonda nova. Nao desbloquear automaticamente
  sem evidencia de cota restabelecida ou rota elegivel.
- Um resultado diferente de `rate_limited` interrompe a sequencia.
- Historico, anexos, workspace e claims concorrentes sao preservados. Nao se
  sintetiza uma tentativa para registrar o estacionamento.

## Prova real no Docker

Imagem `hermes-quota-test:local`, Python 3.11, sem rede nos testes e sem montar
nenhum banco/credencial do operador. Codigo copiado para diretorio efemero;
runner canonico `scripts/run_tests.sh`, dois processos, sem retry de teste.

- `positive.txt`: 76 passaram, 0 falharam, 1 exclusivo Windows ignorado em Linux.
- `negative.txt`: elevar intencionalmente o limite a 999 em copia descartavel
  fez 4 testes falharem; a copia instalada nao foi sabotada.
- Testes cobrem dry-run sem escrita, estacionamento, continuidade de outro
  card, preservacao de historico, desbloqueio explicito, progresso real,
  idempotencia e recusa de tocar claims/run/PID concorrentes.

Comando: `bash scripts/run_tests.sh tests/hermes_cli/test_kanban_quota_budget.py tests/hermes_cli/test_kanban_quota_safety.py tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_review_lifecycle.py -j 2 --file-retries 0 -q`.

## Aplicacao e reversao

Aplicar apenas `hermes_cli/kanban_quota.py` e os pontos de integracao em
`hermes_cli/kanban_db_dispatch.py`, verificando que a base viva nao mudou.
Gateway le codigo no boot: recarga suave, sem escalada para kill; inventario
de PIDs antes/depois e leitura dos limites 6/3 no novo boot.
Reverter o commit de codigo nao autoriza desbloquear cards; a decisao sobre
cota continua explicita. Nenhuma migracao de banco foi criada.
