# API Discovery Report Template

Use this template when documenting API reverse engineering findings.

---

# API Investigation Report: [Target Site]

**Date:** [YYYY-MM-DD]
**Target:** [URL]
**Metodologia:** [Static analysis, brute force, subdomain enumeration, etc.]

---

## RESUMO EXECUTIVO

[2-3 sentences summarizing key findings]

- X public APIs found / None found
- Architecture: [Next.js RSC, SSR, SPA, etc.]
- External services: [List key services]

---

## DESCOBERTAS TÉCNICAS

### 1. Arquitetura Principal

**Framework:** [Next.js, React, Vue, etc.]
- Hosting: [Vercel, AWS, custom]
- Rendering: [SSR, CSR, RSC, Hybrid]
- API Layer: [REST, GraphQL, Server Actions, None]

### 2. Serviços Externos Identificados

| Service | Domain | Purpose | Notes |
|---------|--------|---------|-------|
| [Service name] | [domain] | [purpose] | [notes] |

### 3. Endpoints Públicos Encontrados

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| [path] | [GET/POST] | [200/307] | [what it does] |

### 4. Rotas de Aplicativo (não APIs)

Descobertas via sitemap/navegação:
```
/route/{slug}          -> Description
/route/{slug}/detail   -> Description
```

### 5. Testes de API Realizados

**Subdomínios Testados:**
```
api.target.com         -> [404/200]
graphql.target.com     -> [404/200]
platform.target.com    -> [404/200]
```

**Endpoints REST Testados:**
```
/api/v1/contents       -> [404/200]
/api/catalog           -> [404/200]
/graphql               -> [404/200]
```

### 6. Padrões de Dados Identificados

**Estrutura de [Content Type]:**
```json
{
  "id": "uuid",
  "slug": "example-slug",
  "title": "Example Title",
  "workload": "50h",
  "tags": [...],
  "video_provider": "nivo|bunny|cloudfront"
}
```

---

## POR QUE NÃO HÁ APIs PÚBLICAS? (ou Similar)

List architectural reasons if no public APIs found:

1. [Reason 1: e.g., SSR/RSC architecture]
2. [Reason 2: e.g., Server Actions instead of REST]
3. [Reason 3: e.g., Auth-only APIs]

---

## VIAS DE INVESTIGAÇÃO ADICIONAL

1. **Com Autenticação:**
   - [What to try with login]
   
2. **Análise Dinâmica:**
   - [DevTools/Proxy approach]
   
3. **Fuzzing Avançado:**
   - [Tools: ffuf, gobuster, dirsearch]

---

## ESTATÍSTICAS DA INVESTIGAÇÃO

| Métrica | Valor |
|---------|-------|
| Chunks JS analisados | [N] |
| Páginas HTML analisadas | [N] |
| Endpoints testados | [N] |
| Subdomínios testados | [N] |
| APIs públicas encontradas | [N] |
| Serviços externos identificados | [N] |

---

## CONCLUSÕES

[Key takeaways for the user]

1. [Finding 1]
2. [Finding 2]
3. [Recommendation/Next steps]

---

## AVISO LEGAL

[If needed: disclaimer about passive methods only]

This investigation was conducted using passive methods:
- Public HTML/JS analysis
- Non-invasive endpoint testing
- Legal/ethical techniques only

No attacks, bypass attempts, or malicious activity performed.

---

**End of Report**