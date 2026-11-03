# B-3PO — regras de marca

Fonte da verdade: `brand.json` (nomes e caminhos) + `palette.json` (cores). Nada de hex ou nome de
marca hardcoded fora desta pasta — `scripts/brand_check.py` reprova.

## Marca

| Elemento | Valor |
|---|---|
| Nome do produto | **B-3PO Bot Platform** |
| Nome curto / aplicativo | **B-3PO** |
| Pacote técnico | `b3po-agent` |
| Comando | `b3po` (alias legado `hermes` mantido) |
| Proprietário | DZ23-LTDA |
| Licença | MIT, fork de Hermes Agent (Nous Research) — ver `NOTICE` |

O robô branco com `X` no peito e na orelha, anel azul, wordmark chrome. Não redesenhar, não recolorir,
não esticar, não cortar.

## Cores

Primitivas medidas na arte (`palette.json` → `primitives`): `night #000105`, `glow #73C4EA`,
`deep #0454A0`, `silver #D6DBE6`, entre outras. Os **tokens semânticos** (`semantic`) são o que a UI
consome. `#00BFFF` **não** é primitiva oficial: é aproximação visual e não deve entrar como token.

## Mascotes

20 PNGs 512×512 com transparência em `brand/avatars/`, catalogados em `avatars/manifest.json`
(id, nome, caminho). O mascote representa o bot; **nunca** executa trabalho nem define runtime.
Animações e estados ficam fora do primeiro marco (F5).

## Uso no produto

1. Desktop/web/CLI/TUI consomem os tokens gerados por `scripts/brand_sync.py`.
2. Em superfície escura, usar a arte como está; em superfície clara, usar o ícone quadrado.
3. Espaço livre mínimo em volta do wordmark: altura do `B`.
4. `B-3PO` acima do descritor; descritor `BOT PLATFORM` menor, espaçado, em `silverDim`.
5. Lançamento público bloqueado até pesquisa de anterioridade e aprovação do proprietário (`brand.json → publicLaunch`).
