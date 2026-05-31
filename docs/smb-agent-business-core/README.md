# SitioUno SMB Agent Business Core

Proyecto Factory Híbrido para convertir a Zeus en el prototipo heredable de agentes single-tenant para freelancers, emprendedores y pequeñas empresas.

## Objetivo

Diseñar y planificar módulos internos invisibles, basados en Agent Core SQL + herramientas Hermes, que cubran el 80% de las necesidades operativas de una pequeña empresa sin exigirle al cliente aprender CRM, ERP, marketing suites o dashboards.

El cliente opera por conversación: WhatsApp, Telegram, voz o email. El agente usa los módulos por debajo.

## Alcance inicial

1. Gestión comercial y ventas
   - CRM invisible
   - catálogo/productos
   - inventario simple
   - cotizaciones
   - pedidos
   - invoices operativas
   - link de pago mediante adapter

2. Marketing digital
   - perfil de marca
   - calendario de contenido
   - campañas
   - generación de texto, imagen y video
   - publishing queue
   - adapters a redes/email/blog

3. Contabilidad liviana
   - ingresos
   - gastos
   - salidas
   - cuentas por cobrar/pagar
   - soportes/recibos
   - reporte mensual para contador

## Principio arquitectónico

Core SQL para operación. Wiki para conocimiento. Skills para procedimientos. Adapters para backends externos. Agente como experiencia principal.

Los módulos internos no reemplazan Odoo, Twenty, Lago, Medusa, Meta, Stripe o plataformas similares cuando el cliente necesita escala, UI, fiscalidad o integraciones profundas. Extraen el 80% operativo y dejan adapters opcionales para el 20% avanzado.

## Documentos

- `PRD-001-smb-agent-business-core.md`
- `ADR-001-agent-first-local-cores.md`
- `ADR-002-module-boundaries-and-adapters.md`
- `ADR-003-operational-vs-fiscal-accounting.md`
- `ADR-004-marketing-core-and-publishing-adapters.md`
- `SPRINT_PLAN.md`
- `KANBAN_TASK_GRAPH.md`
- `QA_GATES.md`
- `IMPLEMENTATION_PLAN.md`
- `REPO_ORGANIZATION.md`
- `DELIVERY_REPORT.md`
