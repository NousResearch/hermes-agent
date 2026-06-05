# ADR-002 — Module Boundaries and Adapter Layer

## Estado

Aceptado para planificación.

## Contexto

Los módulos internos deben tomar lo mejor de repos/sistemas analizados sin acoplarse a ellos. Twenty, Odoo, ERPNext, Dolibarr, Medusa, Lago y otros son fuentes de patrones funcionales y posibles adapters.

## Decisión

Separar cada dominio en tres capas:

1. Core local: schema + tools + eventos internos.
2. Document/communication layer: render PDF, enviar mensajes, pedir aprobación.
3. Adapter layer: sistemas externos opcionales.

## Mapeo de inspiración

- Twenty: companies, people, opportunities, pipeline visual.
- Odoo: quote -> order -> invoice -> payment; productos; inventario; contabilidad formal.
- ERPNext: doctypes, stock ledger, accounting ledger.
- Dolibarr: simplicidad SMB para cotizaciones/facturas/contactos.
- Medusa: catálogo, variantes, precios, orders.
- Lago: invoices, billing status, suscripciones y customer billing profile.

## Consecuencias

El core debe ser simple, no una copia completa. Los adapters se activan sólo cuando el cliente los necesita.

## Regla de oro

No escribir directo en DBs externas. Toda integración se hace por API/adapters.
