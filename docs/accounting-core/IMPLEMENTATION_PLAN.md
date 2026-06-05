# Implementation Plan — Accounting Lite Core

## Sprint 1 — Core DB + Tools

1. Crear migración `db/modules/accounting/000001_accounting_schema.sql`.
2. Registrar module metadata en `agent_core.modules` y `module_databases`.
3. Crear tools:
   - `accounting_status`
   - `accounting_account_upsert`
   - `accounting_receipt_create`
   - `accounting_receipt_get`
   - `accounting_journal_entry_create`
   - `accounting_export_create`
4. Registrar toolset `accounting` en `toolsets.py`.
5. Crear tests unitarios de toolset y validación de asientos balanceados.

## Sprint 2 — Receipt workspace pattern

1. Generar PDF de recibo con template similar a cotización.
2. Publicar workspace `/w/<token>/` con comentario, aprobar con firma y rechazar con motivo.
3. Reusar `/api/events` del delivery sandbox.
4. Ingestar eventos en `accounting.receipt_events`.
5. Al aprobar/firmar, marcar recibo `approved/signed` y generar PDF firmado visible.
6. Al rechazar, exigir `rejection_reason`, marcar recibo `rejected` y deshabilitar acciones.
7. Registrar bitácora visible en `comments.json`.

## Sprint 3 — Accountant export

1. Exportar CSV/XLSX de journal lines por negocio/fechas.
2. Incluir recibos, links públicos/PDF y referencias de pago.
3. Enviar al contador por email/adaptador cuando se configure destinatario.

## QA gates

- Migración aplica limpia en `agent-postgres`.
- Tools y toolset resuelven.
- Asientos no balanceados fallan.
- Workspace público renderiza estado/acciones/bitácora.
- PDF de recibo parsea y descarga.
- Export CSV contiene débitos/créditos balanceados.
