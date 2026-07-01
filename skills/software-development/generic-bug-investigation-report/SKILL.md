---
name: generic-bug-investigation-report
title: Prompt de Investigação de Bugs (Genérico + Relatório)
description: Prompt investigativo genérico curto para descobrir bugs em qualquer stack, mapeando fluxo, APIs e arquitetura. Inclui Differential Testing obrigatório e relatório quando o bug é confirmado.
triggers:
  - investigar bug geral
  - bug investigation
  - relatorio bug confirmado
---

# Prompt de Investigação de Bugs (Genérico + Relatório)

Copie, preencha `{{...}}` e use.

---

```
BUG INVESTIGATION — {{PROJECT_NAME}}
Stack: {{STACK}}
Bug: {{DESCRICAO}}
Ambiente: {{DEV/STAGING/PROD}}
Erro:
{{LOGS}}

FLUXO:
{{ENTRADA}} → [A] → [B] → ... → {{SAIDA}}
APIs externas: ...
APIs internas: ...
Dados: ...
Cache/filas/storage: ...
Infra: ...

HIPÓTESES:
[ ] A1: payload/headers errados
[ ] A2: cors/auth/rate limit bloqueando
[ ] A3: proxy/cdn/gateway desviando
[ ] B1: regra de negócio quebrada
[ ] B2: estado/cache/sessão corrompido
[ ] B3: transação rollback
[ ] B4: catch vazio
[ ] B5: race condition
[ ] C1: schema/migration ausente
[ ] C2: constraints
[ ] C3: encoding diferente
[ ] C4: query N+1
[ ] C5: índice quebrado
[ ] D1: perfil/config errada
[ ] D2: .env ausente/vazia
[ ] D3: caminho hardcoded
[ ] D4: falta de recurso
[ ] D5: rede
[ ] E1: API externa mudou
[ ] E2: timeout/retry errado
[ ] E3: webhook não entregue
[ ] E4: rate limit externo
[ ] F1: build antigo
[ ] F2: log insuficiente
[ ] G1: token/secret rotacionado
[ ] G2: CSP/HSTS bloqueando
[ ] G3: path traversal/injection

METODOLOGIA:
1. Reproduza (branch/Docker, 100%)
2. Differential Testing → baseline: .hermes/baseline_{{BUG_ID}}.json
3. Isole causa: logs + traces + comparação com staging
4. Invariantes: [ ] schema [ ] p95 {{X}}ms [ ] 5xx < {{Y}}% [ ] queries {{N}}

REPRODUZIU? {{SIM/NAO}}

CAUSA RAIZ: ...

FIX: ...

VALIDAÇÃO: ...

STATUS: {{CONFIRMADO / FALSO_POSITIVO / NAO_REPRODUZ}}
```

---

Use: `skill_view(name='generic-bug-investigation-report')` para carregar.
