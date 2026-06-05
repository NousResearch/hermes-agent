# PRD — Agent Signature Core

## Objetivo
Crear una capacidad canónica de firmas para agentes Zeus/SitioUno que permita pedir firma digital de cotizaciones, contratos y PDFs desde el sandbox público, registrar auditoría en Agent Core SQL y producir un hash de aprobación verificable.

## Problema
Las cotizaciones web ya permiten comentar/aprobar/rechazar, pero la aprobación no captura una firma ni un hash canónico. Para contratos y documentos PDF se necesita un flujo reutilizable: solicitar firma, capturar evidencia, generar hash, registrar eventos y notificar al agente/owner.

## Alcance v1
- Módulo `signature` en Agent Core DB, dentro del mismo Postgres del agente.
- Tools Hermes `signature_*` para crear templates/requests, consultar estado, registrar eventos y crear hashes de aprobación.
- Patrón sandbox `/sign/<slug>` o integración embebida en `/w/<token>` para capturar firma.
- Aprobación de quote con firma: typed/drawn signature, timestamp, user agent, IP si el sandbox lo provee, document hash y approval hash.
- Auditoría append-only con hash chain por request.
- Documentación de patrón basado en DocuSeal/OpenSign.

## Fuera de alcance v1
- Firma criptográfica PAdES completa con certificado P12/TSA.
- KBA/OTP/SMS obligatorio.
- Backoffice visual tipo DocuSeal/OpenSign.
- Multitenancy SaaS.

## Usuarios
- Owner/agent: pide “envíale este contrato para firma”.
- Cliente/firmante: abre link público, revisa documento, firma y aprueba.
- Zeus: recibe evento, valida, registra, notifica y continúa flujo comercial.

## Requisitos funcionales
1. Crear request de firma desde un documento o quote existente.
2. Generar submitters con token opaco/slug público.
3. Capturar firma typed/drawn y evidencia mínima.
4. Al aprobar, generar `approval_hash = sha256(canonical approval context)`.
5. Registrar eventos append-only con `previous_event_hash` y `event_hash`.
6. Relacionar request con `source_type/source_id` como quote, invoice, contract o pdf.
7. Consultar estado completo con request, submitters, events y approvals.

## Criterios de aceptación
- `signature_status` devuelve conteos desde Postgres.
- `signature_request_create` crea request + submitter link opaco.
- `signature_approval_hash_create` crea approval y marca request completed.
- Tests unitarios pasan.
- La cotización sandbox pide firma antes de aprobar.
- El approval queda persistido y verificable por hash.
