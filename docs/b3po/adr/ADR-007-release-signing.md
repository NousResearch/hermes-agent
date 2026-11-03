# ADR-007 — assinatura e release

**Status:** aceito · **Data:** 2026-11-02

## Contexto

O empacotamento Windows avisou explicitamente: `AZURE_SIGN_* not set — payload binaries will be UNSIGNED`.
Não há certificado de assinatura nem credenciais Apple/Android no ambiente.

## Decisão

1. Artefatos saem **não assinados** e **assim são rotulados** — nunca "assinado", "notarizado" ou "pronto
   para loja" sem verificação.
2. A pipeline de release exige as variáveis de assinatura (`AZURE_SIGN_*`, certificado Apple, keystore
   Android); ausência = artefato marcado como *development-only*.
3. Publicação em loja é **proibida** sem credencial e aprovação do proprietário (Ata §7 e GOAL §17).
4. Checksums SHA-256 acompanham todo artefato no `ARTIFACT-MANIFEST.md`.

## Consequências

- Usuário verá alerta de SmartScreen no `.exe` — comportamento esperado e documentado, não "bug".
- Nenhum release comercial pode ser gerado nesta máquina.
- O relatório de prontidão declara o status de assinatura por artefato.
