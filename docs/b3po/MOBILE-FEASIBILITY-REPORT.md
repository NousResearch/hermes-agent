# MOBILE-FEASIBILITY-REPORT

## 1. Realidade do repositório

O B-3PO herdado do Hermes **não tem cliente mobile**. Não existe `apps/android`, `apps/ios` nem projeto
React Native/Flutter. Verificado por caminho:

```
apps\android  False
apps\ios      False
android       False
ios           False
apps\mobile   False
```

O desktop (Electron) **não é mobile**. O dashboard web e a TUI também não.

## 2. Ambiente desta máquina

| Requisito | Estado |
|---|---|
| JDK 21 | presente |
| Android SDK | presente (`D:\IA-Workspace\Deepseek\tools\android\sdk`) |
| `adb` | presente |
| Gradle | **ausente** (usar Gradle wrapper no projeto) |
| Xcode / macOS | **ausente** |
| Certificados Apple / Android keystore | **ausentes** |

## 3. O que isso significa

- **Android**: tecnicamente viável (SDK + JDK), mas exige **criar o cliente** — não é build de código
  existente. É trabalho novo, estimado em semanas, não em horas.
- **iOS**: **BLOCKED_BY_EXTERNAL_DEPENDENCY** — sem macOS/Xcode não há build, archive ou IPA. Nenhuma
  alegação de iOS pronto pode ser feita.

## 4. Recomendação de escopo

Mobile **não entra no primeiro marco** (F5 já exclui). A ordem correta é:

1. fechar o núcleo (schema + orquestração) e o desktop;
2. congelar o contrato de sessão/eventos;
3. então decidir o cliente mobile (ver `MOBILE-ARCHITECTURE-DECISION.md`).

Tentar criar mobile antes do contrato congelado produz dois contratos divergentes — exatamente o que a ata
proíbe.

## 5. Veredito

`BLOCKED_BY_EXTERNAL_DEPENDENCY` para **iOS** (macOS/Xcode) e **sem projeto** para **Android**. Nenhum
artefato mobile pode ser entregue nesta rodada sem violar a regra de não fabricar evidência.
