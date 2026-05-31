# ADR-004 — Marketing Core and Publishing Adapters

## Estado

Aceptado para planificación.

## Contexto

El agente puede actuar como community manager: planificar contenido, crear copies, generar imágenes/videos y preparar campañas. Publicar en redes requiere autenticación y APIs de terceros.

## Decisión

Marketing Core guardará estrategia, calendario, piezas, assets y cola de publicación. La publicación real se realizará por adapters cuando el cliente autentique sus cuentas.

## Incluido en core local

- Brand profile.
- Content calendar.
- Campaigns.
- Draft posts/articles/emails.
- Asset metadata.
- Approval workflow.
- Publishing queue.

## Adapters opcionales

- Instagram/Facebook vía Meta Graph.
- X/Twitter.
- LinkedIn.
- TikTok.
- YouTube.
- WordPress/Webflow/website.
- Brevo/Mailchimp/SendGrid/Resend.
- Video generation pipeline propio o plataforma externa.

## Política de intervención humana

El agente guía al cliente cuando se requiera autenticación, aprobación de contenido, autorización de publicación, o acceso a cuentas externas.

## Decisión diferida

El conector de video/redes más eficiente se decidirá después de evaluar si conviene generación propia, proveedor externo o híbrido.
