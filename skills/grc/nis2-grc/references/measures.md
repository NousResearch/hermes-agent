# NIS2 — As 10 Medidas Mínimas (Art. 21.º, n.º 2)

Cada medida deve ser implementada com base no risco (risk-based) e na gestão do ciclo de vida. Lista as control families típicas por medida para apoiar o gap assessment.

## (a) Gestão de risco de cibersegurança
- Quadro de gestão de risco (risk framework) documentado.
- Avaliação de risco periódica (ativo × ameaça × impacto).
- Tratamento de risco e aceitação documentada pelo board.
- Controls: ISO/IEC 27001 A.5 / NIST CSF GV, ID.RA.

## (b) Tratamento de incidentes (deteção, análise, resposta)
- Processo de gestão de incidentes (CSIRT ou equivalente).
- Capacidade de deteção e triagem (SIEM/EDR).
- Runbooks de resposta e equipa de crise.
- Controls: NIST CSF RS, DE; ISO 27001 A.5.24–A.5.26.

## (c) Continuidade de negócio (backup, DR, gestão de crise)
- Planos de continuidade (BCP) e recuperação (DRP).
- Backups offline/imutáveis e testes de restauro.
- Gestão de crise e comunicação.
- Controls: ISO 27001 A.5.29–A.5.30; NIST CSF RC.

## (d) Segurança da cadeia de fornecimento
- Due diligence de fornecedores (cyber).
- Cláusulas contratuais de segurança.
- Monitorização de risco de terceiros.
- Controls: NIST CSF GV.SC; ISO 27001 A.5.19–A.5.22.

## (e) Segurança de redes e infraestruturas (incl. gestão de vulnerabilidades)
- Segmentação de rede, hardening, gestão de vulnerabilidades.
- Patch management e configuração segura.
- Controls: NIST CSF PR.IR, PR.PS; CIS Controls 1–6.

## (f) Gestão de ativos e controlo de acessos (incl. autenticação forte)
- Inventário de ativos (hardware/software).
- Gestão de identidade e acessos (IAM), MFA.
- Princípio do menor privilégio.
- Controls: NIST CSF PR.AA; ISO 27001 A.5.15–A.5.18; CIS 5–6.

## (g) Aquisição, desenvolvimento e operação seguros (secure by design)
- SDLC seguro (secure development lifecycle).
- Gestão de mudança e testes de segurança.
- Controls: ISO 27001 A.8 (desenvolvimento); OWASP ASVS; NIST SSDF.

## (h) Tratamento e divulgação de vulnerabilidades
- Processo de receção e tratamento de vulns (incl. programa de divulgação responsável / Coordinated Vulnerability Disclosure).
- Coordenação com CSIRT/autoridade.
- Controls: ISO 30111 / 29147; CVD.

## (i) Higiene cibernética básica e formação de pessoal
- Consciencialização e formação regular (phishing, engenharia social).
- Políticas de higiene (atualizações, senhas, dispositivos).
- Controls: NIST CSF PR.AT; ISO 27001 A.6.

## (j) Políticas de criptografia, encriptação e gestão de chaves
- Encriptação em repouso e em trânsito.
- Gestão de chaves (KMS/HSM).
- Controls: NIST CSF PR.DS; ISO 27001 A.8.24; FIPS 140.

## Notas
- As medidas são **mínimas**; setores críticos podem exigir control adicional via padrões técnicos europeus (ESOs).
- A proporcionalidade é mandatória: medidas adaptadas ao risco e à dimensão da entidade, mas o nível de base não pode ser inferior.
