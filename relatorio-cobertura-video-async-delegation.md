# Relatório de cobertura do vídeo Async Delegation View

Data: 24 de julho de 2026  
Vídeo: `async-delegation-demo.mp4`  
Branch: `feat/async-delegation-view`

## Resultado

Vídeo cobre:

1. painel acoplado de agentes;
2. live steering por ID;
3. entrega cache-safe na fronteira de ferramenta;
4. telemetria real de cache, tokens, custo e latência;
5. controle negativo real com HTTP 400.

## Evidência visual da TUI

- Painel fica acima do compositor.
- Conversa permanece disponível durante trabalho background.
- Contadores `running` e `done`.
- Objetivo, ferramenta recente, duração e `result ready`.
- Transição de execução para conclusão.
- Painel recolhível.
- Steering `@deleg_b7c2 ...` altera abordagem sem abortar delegação.

## Evidência estrutural

Fluxo mostrado:

`steer(text) → _pending_steer → término da ferramenta → último tool_result`

Conteúdo mostrado:

`tool.content += "\n[USER STEER]: inspect only *.py files"`

Invariantes:

- steering espera fronteira segura;
- nenhuma mensagem `user` sintética entra no meio do lote;
- mensagens anteriores não são mutadas;
- sequência lógica permanece `user → assistant → tool → assistant`;
- prefixo anterior permanece byte-estável e elegível para cache.

## Evidência real do provedor

Provedor: Anthropic direto  
Modelo: `claude-haiku-4-5-20251001`  
Credencial: OAuth; segredo nunca registrado  
Arquivo bruto redigido: `async-delegation-provider-evidence.json`

| Cenário | Cache criado | Cache lido | Latência | Custo estimado |
|---|---:|---:|---:|---:|
| Aquecimento | 6.896 | 0 | 1.992 ms | US$0,0086560 |
| Prefixo repetido | 0 | 6.896 | 944 ms | US$0,0007306 |
| Steering cache-safe | 0 | 6.896 | 1.097 ms | US$0,0008186 |

Custo total estimado: **US$0,0102052**.  
Orçamento autorizado: **US$1,00**.  
Limite preventivo do script: **US$0,05**.

Resultados comprovam, nesta execução Anthropic:

- cache write real;
- cache read real;
- prefixo continuou reutilizado após steering anexado ao `tool_result`;
- custo estimado caiu no cache hit;
- latência observada caiu de 1.992 ms para 944 ms na repetição;
- chamada cache-safe seguinte manteve 6.896 tokens lidos do cache.

Latência é observação de amostra única, não benchmark estatístico.

## Controle negativo

Payload com `tool_result` órfão foi enviado ao Anthropic.

Resultado:

- HTTP `400`;
- `BadRequestError`;
- nenhuma inferência aceita.

Isso comprova que payload estruturalmente inválido é rejeitado. Não comprova
ausência absoluta de erros para todos os provedores.

## Escopo não coberto

- OpenAI direto: não testado, pois nenhuma credencial OpenAI API foi detectada.
- GitHub Copilot não foi tratado como OpenAI API.
- Métricas Anthropic não garantem comportamento idêntico em outros provedores.
- Não houve repetição suficiente para alegar percentual estatístico de redução
  de latência.
- Steering não interrompe ferramenta atômica em andamento; entrega ocorre na
  próxima fronteira segura.
- Economia por “abortar e refazer” não foi simulada, pois depende da tarefa.
  Vídeo mede economia de cache no fluxo equivalente.

## Reprodutibilidade e segurança

Script: `scripts/async_delegation_provider_evidence.py`

Proteções:

- aborta antes da chamada se estimativa exceder US$0,05;
- limita respostas a oito tokens;
- não imprime nem grava token;
- trunca IDs de resposta;
- grava somente métricas, modelo, tipo de credencial e erro sanitizado;
- não lê nem mostra valor da credencial.

## Conclusão

Três pilares estão demonstrados. Terceiro pilar agora possui prova estrutural e
telemetria real Anthropic. Única lacuna de provedor: OpenAI direto, bloqueado
por ausência de credencial; vídeo declara isso explicitamente.
