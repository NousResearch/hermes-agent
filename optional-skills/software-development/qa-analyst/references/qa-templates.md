# Templates de QA

## Caso de Teste

```md
### TC-<área>-<número> — <título curto>

- **Prioridade**: P0 (crítico) | P1 (alto) | P2 (médio) | P3 (baixo)
- **Tipo**: funcional | erro | inesperado | segurança | regressão
- **Pré-condições**: estado necessário antes do teste (usuário logado, dado X existente…)
- **Passos**:
  1. …
  2. …
- **Resultado esperado**: comportamento observável e verificável
- **Resultado obtido**: (preencher na execução) ✅ passou | ❌ falhou → BUG-<id>
```

## Relato de Bug

```md
### BUG-<número> — <título: sintoma observável, não causa presumida>

- **Severidade**: S1 (bloqueia uso) | S2 (função quebrada, com workaround) | S3 (degradação) | S4 (cosmético)
- **Prioridade**: P0–P3 (urgência de correção — pode divergir da severidade)
- **Ambiente**: SO, browser/versão, branch/commit, dados usados
- **Passos de reprodução** (mínimos):
  1. …
  2. …
- **Resultado esperado**: …
- **Resultado observado**: … (colar mensagem de erro / status HTTP literal)
- **Evidência**: screenshot, log, response body
- **Frequência**: sempre | intermitente (X de Y tentativas)
- **Caso de teste relacionado**: TC-…
```

## Plano de Testes

```md
# Plano de Testes — <feature>

## Escopo
- **Dentro**: …
- **Fora** (com justificativa): …

## Riscos priorizados
| # | Risco | Impacto | Cobertura planejada |
|---|-------|---------|---------------------|

## Estratégia por camada
| Camada | Ferramenta | O que cobre |
|--------|-----------|-------------|
| Unitário | … | … |
| Integração/API | … | … |
| E2E | Playwright | … |
| Manual exploratório | — | … |

## Critérios de saída
- Todos os casos P0/P1 executados e passando
- Nenhum bug S1/S2 aberto
```

## Análise de Causa Raiz (pós-ciclo)

```md
### RCA — BUG-<número>

- **Causa raiz** (5 porquês, resumido): …
- **Por que não foi detectado antes**: lacuna em <camada/processo>
- **Ação preventiva concreta**: <lint rule | teste de contrato | item de checklist | gate de CI>
- **Teste de regressão criado**: <path do teste> | pendente
```
