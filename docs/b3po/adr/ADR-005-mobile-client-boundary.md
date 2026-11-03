# ADR-005 — fronteira do cliente mobile

**Status:** aceito · **Data:** 2026-11-02

## Decisão

Mobile é **cliente fino** de controle/observação/aprovação. Não executa ferramenta, não mantém sessão
própria, não duplica domínio nem execução. Tecnologia escolhida: **React Native (TypeScript)**, para
compartilhar tipos do contrato com desktop/web.

Detalhamento e comparação em `../MOBILE-ARCHITECTURE-DECISION.md`.

## Consequências

- Implementação bloqueada até: contrato `execution.*` congelado, transporte autenticado definido, keystore
  Android e (para iOS) macOS + Xcode + certificado.
- iOS permanece **BLOCKED_BY_EXTERNAL_DEPENDENCY** nesta máquina.
- Notificação recebida nunca é autorização implícita.
