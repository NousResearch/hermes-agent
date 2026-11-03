# AUTONOMOUS-EXECUTION-LOG

Registro cronológico da execução autônoma autorizada pelo proprietário (Adendo).

## Rodada 1 — Fase A + B

| Hora | Ação | Resultado |
|---|---|---|
| 20:43 | Auditoria de baseline (12 checagens) | A1/A2/A8/A10 passam; A5 falha (sem provider) |
| 21:05 | Descoberto Ollama local (3 modelos) e provado `/v1/chat/completions` | provider disponível sem chave |
| 21:06 | `setup-hermes.ps1 -RuntimeOnly` | exit 0 — Python 3.14.9, node, venv, chromium |
| 21:12 | `hermes -z` com `qwen2.5-coder:7b` | **falhou**: janela 32K < mínimo 64K exigido pelo Hermes |
| 21:13 | Troca para `qwen3.5:4b` (262K) | **`B-3PO online ✅`**, exit 0, 13,4 s |

## Rodada 2 — Fase C (marca)

| Hora | Ação | Resultado |
|---|---|---|
| 21:16 | `brand/` + 20 mascotes + NOTICE + rename-map | criado |
| 21:18 | `brand_check` 14/14 | ok |
| 21:19 | `python scripts/check` | **reprovou**: 7 footguns de encoding + 3 blobs > 500 KiB |
| 21:21 | Correções (utf-8-sig, webp, ícone 512, master fora do repo) | `scripts/check` → **14 checks, ok** |
| 21:22 | Commits `d6345f1d` + `21595add` | `main` intacta |

## Rodada 3 — Fase D/E/F (parcial, esta rodada)

| Hora | Ação | Resultado |
|---|---|---|
| 21:23 | `npm ci` (autorizado pelo Adendo) | 1222 pacotes, exit 0 |
| 21:23 | `npm run typecheck` | exit 0 |
| 21:23 | `npm run build` | exit 0, 0,5 min |
| 21:24 | `npm run pack` | exit 0 — `Hermes.exe` 235,4 MB, **não assinado** |
| 21:24 | Relatórios A–C + matriz de plataformas + ADRs | `docs/b3po/` |

## Decisões tomadas autonomamente (permitidas pelo Adendo)

1. Usar Ollama local como provider de teste — sem custo, sem secret, escopo de workspace isolado.
2. Trocar o modelo de teste quando o runtime recusou janela < 64K (correção reversível).
3. Remover o PNG master do repositório e apontá-lo por SHA-256 (limite de blob do repo).
4. Não criar projeto mobile: a decisão de arquitetura foi registrada e a implementação depende de
   contrato congelado + toolchain; criar agora violaria a ordem aprovada.

## Bloqueios registrados

| # | Bloqueio | Dependência exata |
|---|---|---|
| B1 | macOS/iOS | hardware Apple + Xcode + certificado de assinatura/notarização |
| B2 | Android | projeto cliente inexistente + keystore (SDK e JDK presentes) |
| B3 | Linux | container com electron-builder (Docker/WSL presentes; não executado) |
| B4 | Windows assinado | certificado Code Signing (AZURE_SIGN_* ou PFX) |
| B5 | F5 na UI | execução do fluxo com display (app aberto) — não provado |
