# PRD — vapi-post-call-sales-supervisor

## 1. Producto

`vapi-post-call-sales-supervisor` es el servicio canónico de SitioUno para cerrar el ciclo comercial después de llamadas telefónicas de Sophie/Vapi.

Su misión es convertir cada llamada terminada en acciones comerciales reales, de alta calidad y trazables: análisis de contexto, CRM, generación de material, cotización formal cuando aplique, envío omnicanal, nota de voz, seguimiento y aprendizaje operativo.

## 2. Principio central

No existen dos caminos semánticos “demo” vs “real”. Toda llamada comercial se trata como una oportunidad real de venta y todo material enviado debe tener calidad de cierre.

Los materiales pueden ser:

- demostraciones comerciales para mostrar capacidad,
- cotizaciones formales,
- propuestas,
- documentos de aceptación,
- follow-ups,
- notas de voz,
- links de workspace,
- o combinaciones de ellos.

Pero el estándar de calidad es uno solo: material profesional, específico al cliente, QA antes de envío y registro CRM.

## 3. Problema

El worker anterior ejecutaba lógica determinística con clasificación rígida. Eso permitió un error crítico: una llamada administrativa-contable recibió inicialmente material de taller mecánico. Aunque el sistema técnicamente envió un PDF y SendGrid respondió 202, comercialmente era incorrecto.

Ese error demuestra que un cierre de ventas no puede depender solo de reglas fijas. Los clientes pueden pedir combinaciones variables y la respuesta debe ser razonada, contextual y creativa.

## 4. Objetivos

1. Analizar automáticamente cada llamada terminada con contexto completo.
2. Entender intención comercial, vertical, urgencia, material prometido y próximo paso.
3. Decidir acciones post-call con razonamiento supervisado.
4. Generar material customer-facing de calidad de cierre.
5. Enviar cotización formal cuando el cliente pida precios.
6. Enviar nota de voz Sophie cuando haya WhatsApp válido y sea apropiado.
7. Registrar todo en CRM Core.
8. Crear y ejecutar un funnel de follow-up para oportunidades abiertas.
9. Aprender de errores e iteraciones sin hardcodear casos específicos.
10. Mantener idempotencia, auditabilidad y seguridad.

## 5. No objetivos

- No reemplazar a Vapi como runtime de voz en tiempo real.
- No permitir que Sophie diga que ejecutó acciones durante la llamada si solo las registró.
- No crear rutas separadas “demo” y “real”.
- No enviar material sin QA solo porque una plantilla coincide por keyword.
- No guardar secretos o estado comercial sensible en el public callback container.

## 6. Roles

### Sophie — Voice Sales Worker

- Atiende o inicia llamadas.
- Calienta y califica leads.
- Captura datos y próximos pasos.
- Usa `get_customer_context` cuando el cliente ya existe.
- Registra compromisos vía herramientas.
- Escala a Zeus/supervisor, no directamente a Jean.

### Ingestion Worker — Deterministic Collector

- Consume eventos Vapi.
- Agrupa por `call_id`.
- Recupera transcript/artifact.
- Crea `post_call_task` idempotente.
- No decide el material final.
- No envía documentos customer-facing salvo en modo explícitamente aprobado por el supervisor.

### Sales Supervisor — Intelligent Post-call Agent

Puede ser Zeus directamente o un agente efímero controlado por Zeus. Debe tener herramientas de CRM, documentos, email, WhatsApp, TTS, calendar/follow-up y lectura de artifacts.

Responsabilidades:

- leer transcript completo y tool calls,
- revisar CRM/timeline,
- determinar intención y etapa del funnel,
- decidir acciones,
- generar documentos,
- QA de contenido y visual,
- enviar por canales,
- registrar CRM,
- crear follow-up,
- producir aprendizaje operativo.

### Zeus — Orchestrator/Owner

- Supervisa el proceso.
- Decide umbrales de autonomía.
- Aprueba cambios de metodología.
- Interviene en riesgos altos, casos estratégicos o errores.

## 7. Estados del servicio

1. `call_ingested`
2. `context_loaded`
3. `conversation_analyzed`
4. `action_plan_created`
5. `requires_supervisor_review` o `approved_for_execution`
6. `artifact_generated`
7. `artifact_qa_passed` o `artifact_qa_failed`
8. `delivery_sent`
9. `crm_logged`
10. `followup_scheduled`
11. `learning_recorded`
12. `closed` o `blocked`

## 8. Tipos de acción post-call

### A. Material de capacidad / demostración comercial

Se usa cuando el cliente quiere ver cómo trabajaría el agente en su caso.

Requisitos:

- específico al negocio del cliente,
- no plantilla genérica desconectada,
- muestra capacidades relevantes,
- puede incluir PDF, presentación, mock workspace link, ejemplos de flujo, nota de voz o WhatsApp.

### B. Cotización formal con aceptación

Se usa cuando el cliente pide precios, propuesta comercial, paquete, mensualidad o condiciones.

Requisitos:

- usar la plantilla/formato formal de cotización ya existente en Sales/Signature Core,
- incluir alcance, precio, condiciones, validez y próximos pasos,
- generar link `/w/<token>/` en `zeus-sandbox.kidu.app` cuando aplique,
- incluir aceptación/aprobación con Signature Core,
- registrar quote/opportunity en CRM,
- enviar por email y/o WhatsApp,
- programar follow-up.

### C. Seguimiento comercial

Se usa para oportunidades abiertas sin cierre inmediato.

Requisitos:

- no abrumar al cliente,
- alternar canales con criterio,
- personalizar mensajes según última interacción,
- escalar si hay señal fuerte,
- cerrar/pausar si hay rechazo o silencio prolongado.

## 9. Funnel canónico de follow-up

Todo lead con oportunidad abierta debe entrar en un ciclo de seguimiento hasta cierre, pérdida, pausa o escalamiento.

Cadencia inicial sugerida:

- T+0: envío del material prometido + nota Sophie por WhatsApp si hay número válido.
- T+1 día hábil: WhatsApp corto confirmando recepción.
- T+3 días hábiles: llamada Sophie de seguimiento si no respondió o si el material era importante.
- T+5 días hábiles: email/WhatsApp con valor adicional o respuesta a objeciones.
- T+8-10 días hábiles: segundo intento de llamada o mensaje de cierre suave.
- T+14 días: marcar como nurture/pausado si no hay respuesta; crear recordatorio posterior.

Reglas:

- Si el cliente responde positivamente, avanzar a cotización formal o reunión.
- Si pide precio, generar cotización formal con aceptación.
- Si pide más información, generar material específico.
- Si rechaza, registrar pérdida/razón y no insistir.
- Si hay dudas estratégicas o caso grande, escalar a Zeus.

## 10. QA gates obligatorios

Antes de enviar material:

1. `Transcript Fit Gate`: el material coincide con lo pedido en la llamada.
2. `Customer Specificity Gate`: nombre, empresa, vertical y necesidad correctos.
3. `No Wrong Vertical Gate`: no contiene ejemplos de otro sector salvo que estén claramente marcados como comparativos.
4. `Professional Copy Gate`: texto final, no notas internas, no razonamiento del agente.
5. `Document QA Gate`: parseo del PDF/Doc, sin placeholders, visual QA si aplica.
6. `Commercial Quality Gate`: el material impresiona y conecta con el dolor/oportunidad del cliente.
7. `Channel Delivery Gate`: proveedor confirma envío con message id/status.
8. `CRM Evidence Gate`: contacto, oportunidad, interacción, documento y follow-up registrados.

## 11. Métricas

- llamadas procesadas,
- tiempo desde llamada terminada hasta primer material enviado,
- material enviado por tipo,
- cotizaciones formales generadas,
- tasa de respuesta post-material,
- tasa de reuniones agendadas,
- tasa de aceptación/cierre,
- errores de QA bloqueados antes de envío,
- correcciones post-envío,
- oportunidades pausadas/perdidas y razón.

## 12. Riesgos

- Envío de material incorrecto.
- Exceso de follow-up que abrume al cliente.
- Sophie prometiendo acciones no soportadas.
- Confusión entre material de capacidad y cotización formal.
- Falta de QA visual.
- Duplicación de envíos por idempotencia mal diseñada.

## 13. Criterios de aceptación

- El worker determinístico ya no decide entregables finales customer-facing.
- Cada llamada terminada crea una tarea post-call idempotente.
- El supervisor analiza transcript + CRM antes de actuar.
- No existe split demo/real; todo usa estándar único de calidad comercial.
- Si el cliente pide precios, se genera cotización formal con aceptación.
- Si se envía material y hay WhatsApp válido, se puede enviar nota de voz Sophie.
- Todo envío queda registrado en CRM con provider message id.
- Todo lead abierto queda en funnel de follow-up.
- Hay tests para evitar enviar material de vertical incorrecta.
