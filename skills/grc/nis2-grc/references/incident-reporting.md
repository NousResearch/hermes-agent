# NIS2 — Reporte de Incidentes

## O que é "incidente significativo"
Incidente que: (i) causou ou pode causar perturbação grave de serviço essencial, ou (ii) afetou gravemente a segurança de serviços (confidencialidade, integridade, disponibilidade). A entidade decide com base em critérios de impacto (grau + duração + número de utilizadores + dimensão geográfica + impacto económico).

## Janelas obrigatórias (a partir de "tomar conhecimento")
| Marco | Prazo | Conteúdo mínimo |
|-------|-------|-----------------|
| Aviso prévio (early warning) | **≤ 24 horas** | Confirmação de incidente significativo; se possível, indicação de cibersegurança ou acesso não autorizado; contacto |
| Notificação intermédia | **≤ 72 horas** | Estado, indicadores de comprometimento, gravidade, impacto, mitigação, pedido de assistência |
| Relatório final | **≤ 1 mês** | Causa raiz, cronologia, impacto, medidas tomadas, lições aprendidas |

## Passos (checklist)
1. Detetar / confirmar incidente significativo.
2. Acionar equipa de crise (CSIRT interno + contacto com autoridade).
3. Submeter **early warning ≤ 24h** à autoridade competente (e ponto de contacto CSIRT de nível da UE se aplicável).
4. Preparar e submeter **notificação intermédia ≤ 72h** (atualizar se houver evolução).
5. Se o incidente afetar parceiros da cadeia de valor, alertá-los (obrigação de reporte na cadeia).
6. Submeter **relatório final ≤ 1 mês**.
7. Registar todas as comunicações com timestamp.

## Canais
- Autoridade competente nacional (ver `references/scoping.md`).
- Computer Security Incident Response Team (CSIRT) de nível da UE quando relevante.
- Não confundir com notificação RGPD (72h ao regulador de proteção de dados) — pode correr em paralelo para o mesmo incidente.

## Notas
- Falha de reporte é infração direta e agrava penalizações.
- Documentar a decisão de **não** reportar (quando o incidente for considerado não-significativo) para efeitos de auditoria.
