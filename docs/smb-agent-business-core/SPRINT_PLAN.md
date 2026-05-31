# Sprint Plan — SMB Agent Business Core

## Metodología

Híbrida: disciplina documental BMAD + orquestación/gates SitioUno Factory. No hay deploy sandbox en este ciclo porque el resultado inicial son módulos directos del repo, migraciones y tools sobre Agent Core.

## Sprint 0 — Foundation / Planning

Entregables:
- PRD
- ADRs
- Task graph
- QA gates
- Repo organization
- Factory project/lane/tasks/gates

Criterio de salida:
- Documentación versionada en repo.
- Factory DB registra proyecto, lane y tareas.

## Sprint 1 — Commercial/Sales Core

Entregables:
- Schema/migrations para productos, catálogo extendido, inventario simple, pedidos, invoice/payment request.
- Tools: product/catalog/order/invoice/payment request.
- Smoke flow: lead -> quote -> order -> invoice -> payment link placeholder.

Criterio de salida:
- Tests pasan.
- Sin adapter externo requerido.
- Link de pago queda abstracto con adapter contract.

## Sprint 2 — Accounting Lite Core

Entregables:
- Schema/migrations para expenses, income, accounts, payable/receivable, monthly report snapshots.
- Tools para registrar gasto/ingreso y generar reporte mensual.
- Documentación del límite operativo vs fiscal.

Criterio de salida:
- Reporte mensual exportable.
- Tests de categorización y totales.

## Sprint 3 — Marketing Core

Entregables:
- Schema/migrations para brand profile, campaigns, content calendar, assets, publishing queue.
- Tools para crear plan, drafts, assets y publicar vía adapter placeholder.
- Integración conceptual con skills de imagen/video existentes.

Criterio de salida:
- Draft campaign end-to-end.
- Approval gate antes de publish.

## Sprint 4 — Adapter Contracts

Entregables:
- Contracts para payment, social publishing, email marketing, ERP/fiscal, CRM UI.
- Primer adapter candidato por categoría documentado, no necesariamente implementado.

Criterio de salida:
- Cada adapter tiene env vars, auth flow, failure modes y smoke plan.

## Sprint 5 — Agent Inheritance Package

Entregables:
- Skills operativas para cada módulo.
- Template de configuración para nuevos agentes cliente.
- SOUL.md/profile guidance con agent-prompt-architect.
- Demo scripts conversacionales.

Criterio de salida:
- Un nuevo agente puede heredar el paquete base y operar el 80% sin UI externa.
