# ADR-001 — Firma como Agent Core local, DocuSeal como patrón principal

## Decisión
Usar un módulo local `signature` en Agent Core SQL como fuente canónica. DocuSeal será la guía principal de modelo conceptual (`template -> submission/request -> submitter -> events -> completed_documents`) y OpenSign aportará el patrón de PDF visual + firma criptográfica/certificado como evolución posterior.

## Alternativas analizadas

### DocuSeal
Fortalezas:
- Modelo claro: templates, submissions, submitters, submission_events, completed_documents.
- Snapshot de template en cada submission.
- Values JSON por field UUID y attachments separados.
- Hash SHA-256 de documentos completados.
- Audit trail PDF generado desde eventos.
- API fácil de adaptar a tools.

Debilidades:
- Rails/ActiveStorage/Sidekiq es pesado para el sandbox Zeus.
- PDF signing/certificado requiere más infraestructura.

### OpenSign
Fortalezas:
- Buen flujo de widgets PDF, firma visual y firma P12 final.
- Certificado de completion con hash, IP, timestamps y firmantes.
- Soporta orden estricto y múltiples signers.

Debilidades:
- Parse/Mongo con arrays embebidos mutables para audit trail/placeholders.
- Links de firma basados en base64 de IDs/email no son el patrón canónico para Zeus.
- Varias reglas críticas viven en frontend.

## Patrón elegido
- Canonical Core: Postgres schema `signature` en el Agent Core DB.
- Requests con `source_type/source_id` para integrarse con quotes/invoices/contracts/PDFs.
- Submitters con slug/token opaco; solo se guarda `token_hash_sha256`.
- Eventos append-only con hash chain.
- Approval hash sobre JSON canónico del contexto de aprobación.
- Attachments separados para firma/imágenes/PDFs.

## Consecuencias
- El agente puede operar firmas sin instalar DocuSeal/OpenSign por cliente.
- DocuSeal/OpenSign pueden ser adapters opcionales si un cliente necesita UI/backoffice completo.
- V1 produce evidencia/hash; V2 puede agregar certificado P12/TSA y audit PDF firmado.
