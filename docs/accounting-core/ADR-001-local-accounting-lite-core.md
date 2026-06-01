# ADR-001 — Local Agent Accounting Lite Core

## Estado

Aceptado.

## Decisión

Accounting Lite será un módulo local del Agent Core DB bajo el schema `accounting`, al mismo nivel conceptual que `crm`, `sales`, `signature` y `schedule`.

El módulo registra:

- cuentas contables y bancarias simples
- recibos de pago entrantes/salientes
- eventos públicos del recibo
- asientos contables dobles balanceados
- exportaciones para contador

## Razones

1. Jean quiere agentes single-tenant con cores locales invisibles antes de adapters pesados.
2. Un recibo de pago no debe vivir solo como PDF o email; debe tener control operacional y contable.
3. El patrón de cotizaciones ya probó el flujo correcto: workspace público, comentario, aprobación/rechazo, firma y bitácora.
4. El contador necesita exportes limpios, no logs de chat.

## Alternativas consideradas

- Usar Odoo/QuickBooks/Xero directamente: excesivo para v1 y agrega dependencia de UI/adapters.
- Guardar solo en CRM/Sales: mezcla eventos comerciales con contabilidad y dificulta exportar.
- Generar solo PDF: no deja trazabilidad estructurada ni asientos.

## Consecuencias

- Los agentes pueden registrar ingresos/egresos básicos sin ERP.
- Los recibos pueden heredar el mismo patrón de firma/aprobación que cotizaciones.
- Para necesidades fiscales, multiusuario, impuestos o reportes regulatorios, se conectará un adapter externo.
