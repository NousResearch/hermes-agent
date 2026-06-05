---
name: creative-media-api-validation
description: Class-level workflow to validate creative media APIs (image/video/avatar/lipsync) end-to-end across providers, distinguishing documented/UI capability from executable API access.
version: 1.0.0
author: Paimon
license: MIT
tags: [media, api-validation, creative, avatars, lipsync, governance, integration]
---

# Creative Media API Validation

## Cuándo usar
- Cuando se necesite validar acceso real por API a capacidades creativas (image/video/avatar/lipsync), más allá de marketing/UI.
- Cuando haya que comparar proveedores/modelos para integración productiva.
- Cuando se deban separar problemas de credencial, permisos/gating de workspace, schema y calidad de output.

## Resultado esperado
1. Veredicto por provider/modelo: **usable** vs **blocked** (con evidencia HTTP/runtime).
2. Contrato técnico confirmado: endpoint/model ID, auth, payload mínimo válido.
3. Smoke test reproducible con artefactos (URL/archivo, duración, resolución, costo/latencia).
4. Recomendación de integración con riesgos y fallback.

## Flujo operativo común
1. **Preflight de credenciales**
   - Verificar variables de entorno necesarias (o ejecución con gestor de secretos).
   - Si faltan credenciales: reportar “no verificable por falta de credencial”, no “bloqueado”.

2. **Sanity de API base**
   - Probar endpoints públicos estables para confirmar key/base URL válidas.

3. **Discovery de surface real**
   - Contrastar docs públicas, OpenAPI y surfaces de app/web.
   - Enumerar endpoints candidatos y distinguir API pública vs endpoints internos web.

4. **Validación de schema/enum**
   - Usar errores 4xx útiles (ej. 422 con enums) para descubrir catálogo efectivo de modelos/campos.

5. **Smoke test end-to-end**
   - Ejecutar job real con payload mínimo (image/audio/video según caso).
   - Guardar evidencia: response, output URL/path, metadatos de archivo, tiempo y costo estimado.

6. **Clasificación y reporte**
   - Distinguir claramente: 
     - feature visible en UI/docs vs API ejecutable,
     - modelo listado por schema vs modelo realmente usable en workspace.

## Señales de diagnóstico
- `200`: surface accesible (continuar a pruebas funcionales).
- `401 needs_authorization`: auth/scope insuficiente para ese surface.
- `403 forbidden/missing_permissions`: gating por plan/workspace/feature-flag.
- `422`: payload/schema inválido; puede servir para discovery de contrato.
- `405`: ruta existente con método incorrecto (no implica habilitación).

## Subsección: ElevenLabs (creative/content/avatar/flows)
- Priorizar verificación de API pública (`api.elevenlabs.io`) y no asumir paridad con UI.
- Si `content/models` está bloqueado, usar probe de enum vía `content/generations/price` con `model_id` inválido para descubrir catálogo.
- Validar por modelo individual: algunos pueden responder distinto (200/401/403/422).
- Verificar explícitamente paridad con wrappers (Composio/MCP/SDK) antes de afirmar soporte.

## Subsección: fal.ai (avatar/lipsync)
- Confirmar request schema desde OpenAPI por `endpoint_id` antes del run.
- Ejecutar smoke test real `image_url + audio_url` por modelo/variant.
- Verificar output con `ffprobe` (duración/resolución/fps) y calidad visual.
- Si hay fallos de codec/audio, reintentar con WAV/MP3 antes de concluir bloqueo.

## Pitfalls transversales
- No declarar “habilitado” sin prueba de ejecución end-to-end.
- No confundir descubrimiento de schema con permiso de ejecución.
- No extrapolar resultados de un workspace a todos los workspaces.
- No asumir que wrappers exponen el mismo surface que la API directa.

## Entregable recomendado
- Estado global por proveedor.
- Tabla por endpoint/modelo: código, evidencia breve, interpretación.
- Costo/latencia observada.
- Recomendación concreta: integrar / solicitar habilitación / fallback proveedor.
