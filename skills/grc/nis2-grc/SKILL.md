---
name: nis2-grc
description: "Use when ajudar com conformidade NIS2 (Diretiva (UE) 2022/2555): determinar escopo de entidades (essenciais vs importantes), mapear as 10 medidas mínimas de cibersegurança (Art. 21.º n.º 2), fazer gap/maturity assessment, planear remediação, tratar reporte de incidentes (24h/72h/1 mês) e apoiar a governação e responsabilidade do órgão de gestão."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [grc, nis2, compliance, cybersecurity, euler, governance]
    related_skills: [plan]
---

# NIS2 GRC — Governação, Risco e Conformidade

## Overview
Skill para conduzir trabalhos de GRC (Governance, Risk & Compliance) relativos à Diretiva (UE) 2022/2555 (NIS2). Fornece um fluxo repetível para determinar o escopo de aplicação, mapear as obrigações de cibersegurança, avaliar lacunas (gap assessment), planear remediação, tratar o reporte de incidentes e apoiar a governação do órgão de gestão. O agente entrega artefactos concretos (decisão de escopo, matriz de medidas, plano de remediação, minuta de reporte, pack de governação) e não apenas texto explicativo.

## When to Use
- Determinar se uma entidade está no escopo NIS2 (essencial vs importante).
- Mapear as 10 medidas mínimas de cibersegurança (Art. 21.º, n.º 2).
- Fazer gap assessment / maturity assessment face ao NIS2.
- Planear e priorizar ações de remediação.
- Apoiar o reporte de incidentes (janelas 24h / 72h / 1 mês).
- Preparar o órgão de gestão (board) para obrigações e responsabilidade.
- Não usar como substituto de: RGPD/GDPR (privacidade), DORA (setor financeiro), sector-specific regimes — cross-referenciar, não substituir.

## Factos-chave NIS2
- Base legal: Diretiva (UE) 2022/2555 (reformula a Diretiva 2016/1148 "NIS1").
- Transposição pelos Estados-Membros: até 17 out 2024; aplicável a partir de 18 out 2024.
- Dois níveis: **entidades essenciais** e **entidades importantes** (obrigações idênticas; penas e regime de reporte diferem).
- Tamanho: regra geral, PME com <50 trabalhadores E <€10M volume de negócios anula o escopo (salvo designação expressa pela autoridade nacional).
- Penas: essenciais até €10M ou 2% do volume de negócios mundial (o maior); importantes até €7M ou 1%.
- Reporte de incidentes: aviso prévio ≤24h após tomar conhecimento; atualização intermédia ≤72h; relatório final ≤1 mês.
- Responsabilidade: o órgão de gestão pode ser pessoalmente responsabilizado; deve aprovar e supervisionar a estratégia de cibersegurança.

## Fluxo GRC (5 fases)
Execute em ordem; cada fase tem critério de conclusão verificável.

### Fase 1 — Scoping (escopo)
1. Identificar setor de atividade e tipo de entidade.
2. Aplicar as tabelas de `references/scoping.md`: essencial vs importante; exclusões por tamanho.
3. Verificar designação expressa pela autoridade nacional (ainda que abaixo dos limiares).
4. **Conclusão:** entregar `decisão de escopo` com nível (essencial / importante / fora de escopo) + fundamentação + autoridade competente.

### Fase 2 — Gap Assessment
1. Para cada uma das 10 medidas (Art. 21.º, n.º 2), pontuar maturidade 0–5 segundo `references/gap-assessment.md`.
2. Recolher evidência (políticas, logs, contratos, testes, atas).
3. Calcular % de cobertura e marcar lacunas críticas.
4. **Conclusão:** entregar `matriz de gap` (medida × maturidade × evidência × lacuna × prioridade).

### Fase 3 — Plano de Remediação
1. Priorizar por criticidade (impacto × exposição) e prazos regulatórios.
2. Atribuir owner, prazo e KPI por ação (SMART).
3. **Conclusão:** entregar `plano de remediação` com ações acionáveis e monitorizáveis.

### Fase 4 — Incident Reporting
1. Ao detetar incidente significativo, seguir `references/incident-reporting.md`.
2. Garantir janelas 24h / 72h / 1 mês e notificação à cadeia de valor se aplicável.
3. **Conclusão:** minuta de reporte preenchida e canal de submissão identificado.

### Fase 5 — Governance & Board
1. Mapear deveres do órgão de gestão (aprovação de estratégia, supervisão, formação, responsabilidade).
2. Preparar agenda de board e registo de aprovações.
3. **Conclusão:** `pack de governação` (mandato, estratégia aprovada, atas, declaração de conformidade).

## As 10 medidas mínimas (Art. 21.º, n.º 2)
Detalhe e control families em `references/measures.md`. Resumo executivo:
1. (a) Gestão de risco de cibersegurança
2. (b) Tratamento de incidentes (deteção, análise, resposta)
3. (c) Continuidade de negócio (backup, DR, gestão de crise)
4. (d) Segurança da cadeia de fornecimento
5. (e) Segurança de redes e infraestruturas (incl. gestão de vulnerabilidades)
6. (f) Gestão de ativos e controlo de acessos (incl. autenticação forte)
7. (g) Aquisição, desenvolvimento e operação seguros (secure by design)
8. (h) Tratamento e divulgação de vulnerabilidades
9. (i) Higiene cibernética básica e formação de pessoal
10. (j) Políticas de criptografia, encriptação e gestão de chaves

## Modelo de maturidade
Use a escala 0–5 definida em `references/gap-assessment.md`. Critério de "conforme" por medida: maturidade ≥3 com evidência documentada E testada (não apenas política).

## Templates
- `templates/gap-assessment.md` — matriz pronta a preencher (10 medidas × maturidade).
- `templates/incident-report.md` — minuta de reporte de incidente NIS2.
- `templates/scoping-questionnaire.md` — questionário de determinação de escopo.

## Common Pitfalls
1. **Confundir NIS2 com NIS1** — NIS2 alarga o escopo e introduz responsabilidade do board; não reutilizar artefactos antigos sem revisão.
2. **Exclusão automática por PME** — o limite 50 trabalhadores / €10M é regra geral; entidades designadas expressamente continuam no escopo.
3. **Só documentar, não implementar** — maturidade ≥3 exige aplicação e teste, não apenas política no SharePoint.
4. **Falhar janelas de reporte** — 24h contam a partir de "tomar conhecimento", não da descoberta da causa raiz.
5. **Ignorar cadeia de fornecimento** — obrigatório avaliar e mitigar riscos de terceiros (medida d).
6. **Tratar NIS2 como "IT-only"** — é governação; o board é responsável e deve aprovar a estratégia.
7. **Misturar com RGPD** — o mesmo incidente pode acionar ambos os regimes; reportar nos canais corretos de cada um.
8. **Assume uniformidade entre Estados-Membros** — transposição nacional varia (autoridades, setores adicionais); validar a lei local.

## Verification Checklist
- [ ] Nível de escopo determinado com fundamentação e autoridade competente identificada.
- [ ] As 10 medidas pontuadas (0–5) com evidência para cada uma.
- [ ] Lacunas críticas identificadas e priorizadas (impacto × exposição).
- [ ] Plano de remediação com owners, prazos e KPIs.
- [ ] Fluxos de reporte de incidentes (24h / 72h / 1 mês) definidos e testados.
- [ ] Dever do órgão de gestão mapeado e documentado (atas, aprovações, formação).
- [ ] Entregáveis guardados no diretório de trabalho do projeto.
