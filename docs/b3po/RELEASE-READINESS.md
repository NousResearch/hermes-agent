# RELEASE-READINESS

**Commit:** `b3po-brand` (docs desta rodada sobre `21595add`) · **Data:** 2026-11-02

## Por plataforma

| Critério | Windows | macOS | Linux | Android | iOS |
|---|---|---|---|---|---|
| Build | **OK** (`build` exit 0) | — | — | — | — |
| Empacotamento | **OK** (`pack`, desempacotado) | — | — | — | — |
| Instalador | não gerado | — | — | — | — |
| Instalação limpa | não testada | — | — | — | — |
| Assinatura | **não assinada** (declarado) | — | — | — | — |
| Execução do app | não provada na UI | — | — | — | — |
| Turno Hermes | **OK** (CLI, provider local) | — | — | — | — |
| Bot + mascote | código existe; **não provado na UI** | — | — | — | — |
| Cancelamento | código existe; **não provado** | — | — | — | — |
| Persistência/reinício | não testada | — | — | — | — |
| Status | **PARCIAL** | **BLOQUEADO** | **NÃO TENTADO** | **SEM PROJETO** | **BLOQUEADO** |

## Critérios do GOAL §16

| Critério | Evidência | Status |
|---|---|---|
| Windows com artefato e instalação comprovados | artefato sim; instalação **não** | ✗ |
| macOS com artefato ou bloqueio documentado | bloqueio documentado | ✔ (bloqueio) |
| Linux com artefato | não tentado | ✗ |
| Android com APK/AAB ou bloqueio | sem projeto + SDK presente | ✔ (bloqueio) |
| iOS com archive ou bloqueio | sem Xcode/macOS | ✔ (bloqueio) |
| Núcleo Hermes validado | turno real executado | ✔ |
| Bot persiste | não provado | ✗ |
| Autorização efetiva | política mapeada; não provada E2E | ✗ |
| Eventos reais | eventos existem; adaptador não implementado | ✗ |
| Sem execução duplicada | não testado | ✗ |
| Documentação atualizada | relatórios + ADRs | ✔ parcial |
| Segurança sem bloqueio crítico | lacunas S1–S6 registradas | ✗ |
| Licenças/proveniência documentadas | `NOTICE` + relatório | ✔ |

## Veredito

O projeto **não** atinge `READY_FOR_OWNER_REVIEW`: faltam prova de UI do F5, instalação limpa,
adaptador de eventos e as fases D/E. O que existe com evidência: núcleo executando, marca como dado com
gates, desktop compilando e empacotando, limites de plataforma medidos.

**Status:** bloqueio externo real (Apple/Xcode, Android sem cliente) + trabalho interno pendente (D/E/F).
