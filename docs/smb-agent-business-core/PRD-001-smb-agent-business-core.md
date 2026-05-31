# PRD-001 — SitioUno SMB Agent Business Core

## 1. Resumen

SitioUno SMB Agent Business Core es el paquete base de módulos invisibles para agentes single-tenant. Permite que un agente como Zeus opere ventas, marketing y contabilidad liviana para pequeñas empresas sin interfaces obligatorias.

## 2. Problema

Freelancers, emprendedores y pequeñas empresas necesitan vender, cobrar, organizar clientes, publicar contenido, registrar gastos y preparar reportes, pero no quieren aprender varios sistemas separados. CRM, ERP, email marketing, redes sociales y contabilidad suelen ser demasiado pesados para el primer nivel de operación.

## 3. Visión de producto

El agente es el principal empleado digital del cliente. El dueño habla con el agente y el agente convierte instrucciones naturales en datos estructurados, documentos, seguimientos, publicaciones y reportes.

## 4. Usuarios objetivo

- Freelancers y consultores.
- Negocios pequeños de servicios.
- Repostería, restaurantes pequeños, boutiques, academias, entrenadores, clínicas pequeñas.
- Emprendedores que venden por WhatsApp/Instagram.
- Dueños que quieren menos interfaces y más ejecución.

## 5. Jobs to be Done

- Cuando recibo un lead, quiero que el agente lo registre y le dé seguimiento.
- Cuando vendo un producto/servicio, quiero que el agente cree cotización, pedido, invoice y link de pago.
- Cuando tengo inventario simple, quiero saber qué queda y qué debo comprar.
- Cuando necesito contenido, quiero que el agente planifique, genere y publique con aprobación.
- Cuando llega fin de mes, quiero un reporte limpio para mi contador.

## 6. Módulos MVP

### 6.1 Commercial/Sales Core

Incluye CRM Core existente más:
- Product catalog
- Inventory simple
- Quotes
- Orders
- Operational invoices
- Payment requests
- Document rendering hooks

### 6.2 Marketing Core

- Brand profile
- Audience/personas
- Content calendar
- Campaigns
- Content assets
- Publishing queue
- Review/approval workflow
- Adapter hooks para redes, email, blog y video

### 6.3 Accounting Lite Core

- Income records
- Expense records
- Accounts/cashboxes
- Receipts/supporting files metadata
- Accounts receivable/payable
- Monthly accountant report export

## 7. No objetivos del MVP

- No construir un ERP completo.
- No reemplazar facturación fiscal/legal.
- No prometer conciliación bancaria avanzada.
- No publicar en redes sin adapters/autenticación real.
- No crear UI propia obligatoria.
- No desplegar sandbox para este proyecto de planificación; son módulos directos del repo y Agent Core.

## 8. Requisitos funcionales

### Gestión comercial

- Crear/actualizar clientes, contactos y oportunidades.
- Crear productos/servicios reutilizables.
- Mantener stock simple y movimientos.
- Crear cotizaciones con items, descuentos e impuestos simples.
- Convertir cotizaciones aceptadas en pedidos.
- Crear invoice operativa desde pedido/cotización.
- Crear link de pago vía adapter configurado.
- Enviar documento por WhatsApp/email si el canal está disponible.

### Marketing

- Guardar perfil de marca por cliente.
- Crear calendario semanal/mensual de contenido.
- Generar copy, imágenes, videos y artículos como drafts.
- Requerir aprobación antes de publicar, salvo política explícita del cliente.
- Publicar por adapter cuando la cuenta esté autenticada.
- Mantener historial de campañas y piezas publicadas.

### Contabilidad liviana

- Registrar gastos e ingresos por conversación, foto, PDF o email.
- Categorizar transacciones con categorías simples.
- Asociar gastos a proveedores, productos, campañas o pedidos cuando aplique.
- Generar reporte mensual para contador.
- Marcar gaps: recibos faltantes, pagos pendientes, facturas vencidas.

## 9. Requisitos no funcionales

- Single-tenant: cada cliente mantiene Agent Core DB y secrets propios.
- Module-owned migrations: cada módulo migra su schema en Agent Core DB.
- API/tool-first: toda funcionalidad debe exponerse como herramientas Hermes.
- Adapter-neutral: no acoplar el core a Twenty/Odoo/Stripe/Meta.
- Auditabilidad: registrar interacciones y eventos relevantes.
- Seguridad: secrets sólo en Infisical/runtime env.
- Verificación: tests unitarios + smoke flows por módulo.

## 10. Métricas de éxito

- Un cliente pequeño puede operar ventas básicas sin Twenty/Odoo.
- El agente puede completar flows end-to-end desde WhatsApp.
- El 80% de casos simples no requieren UI externa.
- Adapters pueden activarse sin migrar la experiencia del usuario.
- Otros agentes pueden heredar módulos y skills desde el repo.
