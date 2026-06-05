# KANBAN Task Graph — Agent Signature Core

| ID | Estado | Tarea | Criterio de hecho |
|---|---|---|---|
| SIG-001 | Done | Analizar DocuSeal/OpenSign | Patrón documentado con decisión ADR |
| SIG-002 | Done | Diseñar schema Signature Core | Migración SQL creada |
| SIG-003 | Done | Crear tools signature_* | Tool Python registrado |
| SIG-004 | Done | Registrar toolset signature | `resolve_toolset('signature')` funciona |
| SIG-005 | Done | Tests unitarios | Tests específicos pasan |
| SIG-006 | In progress | Integrar quote approval con firma | Sandbox exige firma y genera hash |
| SIG-007 | Pending | Página genérica `/sign/<slug>` para PDFs | Flujo standalone de firma PDF |
| SIG-008 | Pending | Certificado/audit PDF | Audit visible descargable |
| SIG-009 | Pending | Adapter DocuSeal/OpenSign opcional | Diseño de adapter, no default |

## Engine choice
- Zeus directo para bootstrap por tocar su propio Agent Core/sandbox.
- Revisión independiente recomendada con Codex/Claude antes de merge a main si se expande V2.
