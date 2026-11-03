# MOBILE-ARCHITECTURE-DECISION (ADR-005 resumido)

**Status:** decidido, implementação adiada (depende do contrato de sessão congelado).
**Decisão:** **React Native (TypeScript)** para Android e iOS, como **cliente fino** de controle,
observação e aprovação — nunca executando ferramentas nem mantendo sessão própria.

## Comparação

| Opção | Prós | Contras | Veredito |
|---|---|---|---|
| **React Native/TS** | compartilha tipos do contrato com desktop/web (TS); um código para as duas plataformas; ecossistema de keychain/keystore e push | toolchain nova no repo; performance inferior em telas pesadas | **escolhida** |
| Flutter | UI consistente, bom desempenho | linguagem nova (Dart), tipos duplicados do contrato, ecossistema de plugins divergente | rejeitada |
| Kotlin + Swift nativos | melhor integração de plataforma | **duas** bases para manter; custo dobra para um time pequeno | rejeitada |

## Limites obrigatórios (da ata)

- mobile **não** executa ferramenta, **não** cria segundo loop, **não** mantém sessão concorrente;
- aprovação recebida por push **não** é autorização implícita — sempre confirma;
- tokens em Keychain/Keystore, nunca em texto claro;
- reconexão sem duplicar execução (`execution_id` idempotente);
- deep link assinado/validado.

## Pré-condições para começar

1. contrato `execution.*` congelado no v1 (Fase E);
2. transporte definido (HTTP + SSE/WebSocket) e autenticado;
3. SDK/Gradle e keystore Android disponíveis (SDK já existe; keystore não);
4. macOS + Xcode + certificado para iOS (ausentes → BLOCKED).

## Consequência

Nada de mobile é implementado nesta rodada. O documento existe para que a decisão não seja improvisada
depois, e para registrar por que **não** se escolheu Flutter nem nativo duplo.
