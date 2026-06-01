# Accounting Lite Core — PRD

## Objetivo

Crear un módulo canónico de contabilidad básica para agentes Zeus/SitioUno que permita registrar recibos de pago, ingresos, egresos, cuentas bancarias/caja, asientos contables simples y exportaciones para el contador.

## Problema

Los agentes ya pueden manejar CRM, cotizaciones, invoices y firmas, pero falta una capa local de control contable ligero para saber qué dinero entró/salió, por qué concepto, desde qué cuenta y con qué evidencia. El usuario no debe aprender un ERP para operaciones básicas; el agente debe registrar y preparar la información para el contador.

## Alcance v1

- Schema `accounting` dentro del Agent Core DB compartido.
- Cuentas contables/bancarias básicas: assets, liabilities, equity, income, expense.
- Recibos de pago entrantes/salientes con workspace público estilo cotización.
- Eventos de recibo: sent, opened, commented, approved/signed, rejected.
- Asientos contables dobles balanceados con líneas débito/crédito.
- Export CSV para contador.
- Tools `accounting_*` heredables por otros agentes.

## Fuera de alcance v1

- Contabilidad fiscal/legal completa.
- Conciliación bancaria automática.
- Impuestos complejos, nómina, e-invoicing o reportes regulatorios.
- Sustituir Odoo/QuickBooks/Xero; esos serán adaptadores cuando el cliente lo requiera.

## Criterios de aceptación

- `accounting_status` devuelve conteos del schema.
- `accounting_account_upsert` crea/actualiza cuentas.
- `accounting_receipt_create` crea recibos con metadata y links públicos.
- `accounting_journal_entry_create` rechaza asientos no balanceados.
- `accounting_export_create` genera CSV para contador.
- Un recibo público puede comentar, aprobar con firma o rechazar con motivo.
