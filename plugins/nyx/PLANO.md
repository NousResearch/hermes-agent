# Nyx — plano e estado

**Nome:** Nyx (deusa grega da noite; combina com Hermes, que também é grego, e com um avatar feito
de estrelas). Substitui "Jarvis".

**Forma de integração:** plugin do Hermes (`plugins/nyx/`), sem tocar no core. Hooks observam o
turno, um barramento por perfil leva os eventos ao dashboard, e uma aba do dashboard desenha o
avatar. Nenhuma ferramenta nova vai pro modelo.

Regra do produto: **a galáxia só aparece quando o Hermes pensa e chama ferramentas.** Em repouso,
só o busto.

---

## Feito (v0.1.0)

| Bloco | Onde | Como foi verificado |
|---|---|---|
| Hooks → eventos (`pensando`, `ferramenta`, `ferramenta_fim`, `falando`, `pronto`, `subagente`, `subagente_fim`) | `__init__.py` | `tests/plugins/test_nyx_plugin.py`: plugin carregado pela descoberta real, turno completo chega ao leitor em ordem e sem conteúdo da conversa |
| Barramento por perfil (`$HERMES_HOME/nyx/eventos.jsonl`) | `barramento.py` | teste do leitor depois do arquivo encurtado; E2E com evento vindo de OUTRO processo aparecendo no painel |
| API do painel: SSE `/eventos` + `POST /ensaio` | `dashboard/plugin_api.py` | dashboard real (`hermes dashboard`), aba "ao vivo", ensaio percorre todos os estados |
| Aba "Nyx" no dashboard | `dashboard/manifest.json`, `dist/index.js`, `dist/style.css` | screenshot no dashboard real, sem erro de console |
| Cena 3D: busto em pontilhismo (~3 milhões de pontos; a luz decide quantos aparecem), galáxia de 3 braços, satélite, cometas por ferramenta (âmbar no erro), dourado pulsando ao falar | `dashboard/dist/nyx.js` | harness com eventos injetados: repouso → pensando → ferramentas → falando → pronto; contorno comparado com a foto linha a linha (erro médio ~2,5 px) |
| Three.js r181 vendorizado (MIT), imports reescritos pra caminhos relativos | `dashboard/dist/vendor/` | carrega no dashboard sem npm nem CDN |

Detalhes que já estão no código:
- **Histerese:** a galáxia só nasce se pensar durar mais de 0,3 s, e fica no mínimo 1,2 s.
- **Exposição:** o brilho se ajusta pela altura do painel.
- **Cor do cometa:** sai de uma tabela por ferramenta.
- **Estado em funções puras:** `reduzir` e `alvoGalaxia`.

---

## Próximos blocos

1. **Voz no painel.** Mic → `POST /api/audio/transcribe` → enviar o prompt → `WS /api/audio/speak-stream`
   → o áudio do TTS dirige o dourado da boca (hoje o pulso é sintético). Tudo isso já existe no
   servidor.
2. **Conversar pela aba.** Uma caixa de texto na aba que manda o prompt pela sessão do dashboard
   (`WS /api/ws`, JSON-RPC do `tui_gateway`). Assim a aba vira o "rosto" de uma conversa, e não só
   espelho das outras superfícies.
3. **Filtro de sessão.** O painel hoje mostra todas as sessões do perfil. Falta um seletor
   "acompanhar esta sessão" (os eventos já trazem `sessao`).
4. **Modo foto.** Portar o avatar 2.5D por vistas (`apps/jarvis-avatar/`) como segundo modo do
   painel, ou aposentá-lo. Decisão do usuário.
5. **Testes JS.** Vitest de `reduzir`/`alvoGalaxia`, as invariantes do estado: "`falando` sempre
   recolhe", "evento desconhecido não muda nada", "`pronto` fecha todos os cometas".
6. **Desktop.** Painel "Nyx" no app Electron apontando pra mesma aba.
7. **Pitch/aceno.** Usar `pose.js` do avatar antigo pra olhar pro mouse e acenar.

---

## Riscos e decisões em aberto

- **Profile multiplex:** o dashboard lê o barramento do perfil em escopo na requisição. Se o
  dashboard servir vários perfis, a aba mostra o perfil ativo. Validar A→B→A antes de chamar de
  pronto pra multi-perfil.
- **`on_stream_delta` liga o observador de streaming do agente** (fila assíncrona, fora do caminho
  dos tokens). O custo é baixo, mas existe; dá pra trocar por `post_llm_call` se incomodar.
- **Tamanho:** ~735 KB de Three.js na pasta do plugin. Carrega só quando a aba é aberta.
