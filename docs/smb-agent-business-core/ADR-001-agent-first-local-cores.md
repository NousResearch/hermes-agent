# ADR-001 — Agent-first Local SQL Cores

## Estado

Aceptado para planificación.

## Contexto

El producto SitioUno apunta a clientes que no quieren aprender sistemas. Herramientas como Odoo, Twenty o ERPNext tienen mucho valor, pero son UI/backoffice-first. Para el segmento inicial, el agente debe operar lo básico en conversación y guardar estado estructurado local.

## Decisión

Crear cores funcionales invisibles en Agent Core SQL, operados por herramientas Hermes:

- CRM Core
- Sales/Commercial Core
- Inventory Core
- Marketing Core
- Accounting Lite Core

Cada core tendrá schema, migraciones, tools y tests propios.

## Consecuencias

Positivas:
- Menor fricción para clientes pequeños.
- Menor dependencia de backends externos.
- Mejor UX conversacional.
- Módulos heredables por nuevos agentes.

Negativas:
- Requiere diseñar boundaries claros para no recrear un ERP gigante.
- Hay que definir cuándo escalar a adapters.
- Hay riesgo de prometer fiscalidad o contabilidad formal si no se documenta el límite.

## Regla

Core local para operación simple. Adapter externo para UI, escala, compliance, fiscalidad o integraciones profundas.
