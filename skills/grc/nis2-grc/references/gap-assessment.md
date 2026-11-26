# NIS2 — Gap Assessment (Modelo de Maturidade 0–5)

## Escala de maturidade
| Nível | Significado | Evidência esperada |
|------|-------------|--------------------|
| 0 | Inexistente | Sem política nem prática |
| 1 | Inicial/ad-hoc | Práticas esporádicas, não documentadas |
| 2 | Documentado | Política escrita, aplicação inconsistente |
| 3 | Definido e aplicado | Política + aplicação consistente + owner |
| 4 | Gerido e medido | Métricas/KPIs, melhoria contínua |
| 5 | Otimizado | Automatizado, auditado, benchmark externo |

**Critério de conformidade por medida:** nível ≥ 3 **E** evidência de aplicação **E** teste recente (ex.: DR testado, MFA em produção, pen-test realizado). Documentação isolada = nível 2 no máximo.

## Matriz de cálculo
- % cobertura = (soma das maturidades) / (10 × 5) × 100.
- Lacuna crítica = medida com nível < 3 em (a), (b), (c), (d), (f) — estas têm impacto direto em reporte e continuidade.
- Prioridade = Criticidade (1–5) × Exposição (1–5); ordenar descendente.

## Exemplo de preenchimento
| # | Medida (Art. 21.º) | Maturidade | Evidência | Lacuna | Prioridade |
|---|--------------------|-----------|-----------|--------|-----------|
| a | Gestão de risco | 3 | Risk register aprovado pelo board | — | P2 |
| b | Tratamento de incidentes | 2 | Runbook, sem SIEM | Sem deteção | P1 |
| c | Continuidade | 1 | Backup manual | Sem DR testado | P1 |
| d | Cadeia de fornecimento | 0 | — | Sem due diligence | P1 |
| e | Redes/infra | 3 | Segmentação + VM | — | P3 |
| f | Ativos/acessos | 2 | IAM parcial, MFA não geral | MFA incompleto | P1 |
| g | Aquisição/desenvolvimento | 3 | SDLC em projeto chave | Não em todos | P3 |
| h | Vulns/divulgação | 1 | — | Sem CVD | P2 |
| i | Higiene/formação | 4 | Formação trimestral | — | P4 |
| j | Criptografia | 3 | TLS + discos | Sem KMS formal | P3 |

% cobertura = 22/50 = 44%. Lacunas críticas: b, c, d, f.
