# ADR-006 — empacotamento por plataforma

**Status:** aceito · **Data:** 2026-11-02

## Contexto

O desktop usa `electron-builder` com targets declarados (`msix`, `dmg`, `zip`, `AppImage`, `deb`, `rpm`).
Cada plataforma exige runner e ferramentas próprias.

## Decisão

| Plataforma | Alvo | Onde é produzido |
|---|---|---|
| Windows | NSIS/MSIX + app `--dir` | runner Windows (esta máquina) |
| macOS | `.app` + `.dmg`/`.zip` | runner macOS **obrigatório** |
| Linux | AppImage + `.deb`/`.rpm` | container Linux (docker) |
| iOS | archive/IPA | runner macOS + Xcode + certificado |
| Android | APK/AAB | runner com SDK + Gradle (SDK presente, projeto não) |

Regra de nomenclatura: **nunca** chamar `.app`/`.dmg` de `.exe`; cada artefato declara formato correto.

## Consequências

- Nenhum artefato é anunciado como "assinado" sem assinatura verificada.
- Builds cruzados só por container/runner dedicado; nunca "simular" build de outra plataforma.
