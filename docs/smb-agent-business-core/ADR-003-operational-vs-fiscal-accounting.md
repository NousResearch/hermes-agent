# ADR-003 — Operational vs Fiscal Accounting Boundary

## Estado

Aceptado para planificación.

## Contexto

El agente puede registrar ventas, gastos, cobros y reportes mensuales, pero facturación fiscal, contabilidad oficial e impuestos dependen de país, regulaciones, proveedores y credenciales.

## Decisión

Accounting Lite Core e invoices locales son registros operativos, no documentos fiscales oficiales salvo que un adapter fiscal/ERP esté configurado.

## Incluido en core local

- Ingresos y gastos.
- Cuentas por cobrar/pagar.
- Recibos y soportes.
- Reporte mensual para contador.
- Invoices operativas o recibos simples.

## Requiere adapter externo

- Factura fiscal oficial.
- Libros contables oficiales.
- Declaraciones de impuestos.
- Nómina formal.
- Conciliación bancaria avanzada.
- Cumplimiento país-específico.

## Consecuencia

El agente debe comunicar claramente: “esto es un registro operativo” vs “esto fue emitido en el sistema fiscal/contable oficial”.
