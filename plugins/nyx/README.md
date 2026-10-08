# Nyx — avatar de partículas do Hermes

Nyx é um busto feito só de pontos de luz que mostra o que o Hermes está fazendo. Em repouso fica só
o busto. Quando ele começa a pensar, uma galáxia de três braços nasce em volta da cabeça. Cada
ferramenta chamada solta um cometa que orbita a cabeça enquanto ela roda, e o cometa fica âmbar se
a ferramenta deu erro. Quando a resposta começa a sair, a galáxia recolhe e o dourado do rosto
pulsa. Subagentes ganham uma galáxia-satélite menor.

## Instalar

Num Hermes que não tem o Nyx (instala direto deste repositório, já ativado):

```bash
hermes plugins install julinhodailha/hermes-agent/plugins/nyx --enable
hermes dashboard          # aba "Nyx", na seção de plugins da barra lateral
```

Enquanto o Nyx não estiver na `main`, fixe o commit da branch com `--ref <sha de 40 caracteres>`.
Neste fork o plugin já vem junto: basta `hermes plugins enable nyx`.

Funciona com qualquer superfície que rode o agente (CLI, TUI, desktop, gateway de mensagens): os
hooks escrevem no barramento do perfil e o painel lê de lá. O botão **ensaio** manda pelo mesmo
barramento uma sequência de exemplo (pensar → duas ferramentas → responder), sem precisar de uma
conversa de verdade.

## Como funciona

| Peça | O que faz |
|---|---|
| `__init__.py` | registra os hooks `pre_llm_call`, `pre_tool_call`, `post_tool_call`, `on_stream_delta`, `post_llm_call`, `subagent_start`, `subagent_stop` |
| `barramento.py` | cada hook vira uma linha em `$HERMES_HOME/nyx/eventos.jsonl` (só-acréscimo, encurtado a 200 linhas quando passa de 256 KB) |
| `dashboard/plugin_api.py` | `GET /api/plugins/nyx/eventos` (SSE que segue o arquivo) e `POST /api/plugins/nyx/ensaio` |
| `dashboard/dist/index.js` | a aba: monta a cena e liga o SSE |
| `dashboard/dist/nyx.js` | a cena Three.js: silhueta, orelhas e queixo medidos na foto de referência; busto de ~1,2 milhão de grãos (fios, contas e névoa, gerados em ~1 s); galáxia, satélite, cometas e o redutor de estado (`reduzir`, `alvoGalaxia`: funções puras) |
| `dashboard/dist/vendor/` | Three.js r181 (MIT) e os addons usados, com os imports reescritos pra caminhos relativos (sem npm, sem CDN) |

Os hooks só observam: nenhum devolve diretiva, nenhuma ferramenta é adicionada ao modelo, e nada do
conteúdo da conversa (argumentos, resultados, texto) vai pro barramento. Só vão o tipo do evento,
o nome da ferramenta, os ids e o status.

## Eventos

`pensando` · `ferramenta {ferramenta, id}` · `ferramenta_fim {ferramenta, id, erro, ms}` ·
`falando` (uma vez por iteração, no primeiro trecho de texto) · `pronto` · `subagente` ·
`subagente_fim`. Todos levam `t` (epoch) e `sessao`.
