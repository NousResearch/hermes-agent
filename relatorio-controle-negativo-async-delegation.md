# Relatório: controle negativo da Async Delegation View

Data: 24 de julho de 2026

## Objetivo

Controle negativo verifica se provedor rejeita uma estrutura sabidamente
inválida. Ele não representa falha acidental do Codex, Hermes ou Anthropic.

## Cenário válido

Fluxo normal de ferramenta:

```text
user
  → assistant com tool_use(id="toolu_demo")
  → tool_result(tool_use_id="toolu_demo")
  → assistant
```

Cada `tool_result` deve apontar para um `tool_use` anterior. No steering
cache-safe, correção é anexada ao conteúdo desse resultado válido:

```text
scan complete
[USER STEER]: inspect only *.py files
```

Isso mantém vínculo entre chamada e resultado e evita inserir mensagem `user`
sintética no meio da iteração.

## Cenário negativo proposital

Teste enviou:

```text
user
  → tool_result(tool_use_id="missing_tool_call")
```

Não existia `tool_use` com ID `missing_tool_call`. Resultado ficou órfão.

Anthropic respondeu:

- HTTP `400`;
- `BadRequestError`;
- requisição rejeitada antes de resposta útil do modelo.

## O que resultado comprova

- Provedor valida relacionamento entre `tool_use` e `tool_result`.
- Payload inválido usado no controle é rejeitado.
- Fluxo cache-safe deve preservar estrutura válida.
- Anexar steering ao resultado existente evita criar resultado órfão.

## O que resultado não comprova

- Não garante zero erros em todas as APIs.
- Não prova comportamento de OpenAI ou outros provedores.
- Não indica defeito na conta Anthropic.
- Não indica falha inesperada do Codex.
- Não mede qualidade da resposta do modelo.

## Segurança e privacidade

Teste não gravou:

- token OAuth;
- chave de API;
- e-mail;
- nome da conta;
- dados pessoais;
- cabeçalhos de autenticação.

Evidência registrou somente provedor, modelo, métricas de uso, latência, custo
estimado, HTTP 400 e tipo sanitizado do erro.

## Custo

Controle negativo foi rejeitado. Três chamadas válidas de medição tiveram
custo estimado total de US$0,0102052. Cobrança exata depende do plano
Anthropic associado ao OAuth.

## Conclusão

HTTP 400 foi resultado esperado e intencional. Controle negativo demonstrou
que estrutura incorreta é bloqueada, enquanto cenário de steering usa
`tool_result` válido e preserva prefixo cacheável.
