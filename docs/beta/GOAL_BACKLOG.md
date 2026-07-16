# Beta — Goal Mode Backlog

Branch alvo: `beta/core-orchestrator`

## GOAL — Transformar Hermes Agent no Beta Orquestrador

Transformar o fork do Hermes Agent no Beta, um Chief of Staff de IA que conversa com o Chefe, interpreta intenção, seleciona especialistas, delega execução, acompanha resultados, valida evidências e responde com uma única voz.

Princípios obrigatórios:

- Preservar o Hermes original como modo padrão.
- O Beta não deve virar um executor técnico genérico.
- Toda mudança deve ser incremental, testável e reversível.
- Não espalhar condicionais `if mode == beta` pelo código.
- Reutilizar `delegate_task`, Kanban, MemoryManager, tool registry e approval guardrails existentes.
- Nenhuma execução de risco alto sem autorização explícita.
- Não inventar execução nem resultado.

Critério final:

Ao receber `Verifique por que o PostgreSQL está lento`, o Beta deve classificar a intenção, selecionar DBA e Infra/Monitoring, delegar coleta somente leitura, consolidar evidências, informar causa provável e pedir autorização antes de qualquer ação de impacto.

---

## BETA-001 — Integrar o modo Beta ao runtime real

### Objetivo

Substituir o bootstrap temporário por uma integração nativa e centralizada no runtime empacotado.

### Requisitos

- Suportar `agent.mode: hermes|beta`.
- `hermes` deve ser o padrão.
- Criar um provider/resolver central de identidade.
- `agent/system_prompt.py` deve consumir a identidade resolvida.
- Compatibilidade com `SOUL.md` deve ser definida e testada.
- Remover dependência funcional de `sitecustomize.py`.
- Funcionar em modo editável, wheel e instalador oficial.
- Não espalhar condicionais específicas do Beta.

### Critérios de aceite

- Sem configuração, o prompt permanece Hermes.
- Com `agent.mode: beta`, a identidade principal é Beta.
- Valor inválido volta para Hermes com warning.
- Testes unitários e de integração passam.
- Documentação atualizada em `docs/beta/BETA-001.md`.

---

## BETA-002 — Criar registro declarativo de especialistas

Depende de: BETA-001.

### Objetivo

Criar um catálogo independente do orquestrador para descoberta de agentes especialistas.

### Requisitos

Cada especialista deve declarar:

- id e nome;
- descrição;
- capacidades e palavras-chave;
- toolsets permitidos;
- ferramentas bloqueadas;
- modelo/provider opcional;
- nível de risco permitido;
- acesso de memória;
- limite de concorrência;
- status habilitado/desabilitado.

Criar especialistas iniciais:

- infra;
- dba;
- security;
- monitoring;
- dev;
- devops;
- memory;
- qa-auditor.

### Critérios de aceite

- Adicionar agente novo não exige alterar o roteador.
- Manifestos inválidos falham com mensagem clara.
- IDs duplicados são rejeitados.
- Registro possui testes.
- Documentação inclui exemplo de manifesto.

---

## BETA-003 — Implementar roteador de intenção e especialistas

Depende de: BETA-002.

### Objetivo

Transformar solicitações do Chefe em classificação de intenção e seleção fundamentada de especialistas.

### Classificações mínimas

- conversa simples;
- informação;
- diagnóstico;
- planejamento;
- execução técnica;
- mudança em produção;
- auditoria;
- memória.

### Requisitos

O resultado do roteador deve ser estruturado e conter:

- intenção;
- especialistas selecionados;
- justificativa curta;
- risco inicial;
- necessidade de delegação;
- possibilidade de paralelismo;
- confiança.

### Critérios de aceite

- `PostgreSQL lento` seleciona DBA e Infra/Monitoring.
- `Revise este contrato` não seleciona agente técnico de infraestrutura.
- Conversa simples não gera delegação desnecessária.
- Seleção usa capabilities do registro, não uma lista fixa no código.
- Testes cobrem ambiguidades e múltiplos especialistas.

---

## BETA-004 — Implementar contrato estruturado de delegação

Depende de: BETA-003.

### Objetivo

Padronizar a comunicação entre Beta e especialistas usando o `delegate_task` existente.

### Contrato da tarefa

- task_id;
- specialist_id;
- objetivo;
- contexto mínimo;
- restrições;
- risco;
- ferramentas permitidas;
- entregável esperado;
- timeout;
- correlation_id.

### Contrato da resposta

- status;
- resumo;
- evidências;
- fatos;
- hipóteses;
- confiança;
- ações recomendadas;
- riscos;
- necessidade de autorização;
- erros.

### Requisitos

- Contexto do filho deve permanecer isolado.
- Especialista não pode escrever na memória estratégica do Beta.
- Especialista não pode falar diretamente com o Chefe.
- Suportar execução paralela com limite configurável.
- Preservar observabilidade dos subagentes.

### Critérios de aceite

- Respostas inválidas são marcadas como falha de contrato.
- Timeout parcial não elimina resultados válidos dos demais agentes.
- Correlação permite rastrear toda a delegação.
- Testes cobrem uma tarefa simples e uma execução paralela.

---

## BETA-005 — Implementar política de risco e Approval Gate

Depende de: BETA-004.

### Objetivo

Impedir que o Beta ou especialistas executem ações de impacto sem autorização explícita.

### Níveis

- baixo: leitura, logs, métricas, relatórios;
- médio: preparação de scripts/configurações e simulações;
- alto: alteração de produção, restart, deploy, banco, firewall, permissões e exclusões.

### Requisitos

- Classificar ferramentas e operações por risco.
- Risco alto deve parar antes da execução.
- Pedido de autorização deve informar alvo, ação, impacto e rollback.
- Autorização deve ser associada à operação exata.
- Alteração da operação invalida autorização anterior.
- Eventos devem ser auditáveis.

### Critérios de aceite

- Consulta de logs segue sem aprovação.
- Reinício de serviço é bloqueado até aprovação.
- Aprovação de um servidor não autoriza outro.
- Especialista não consegue contornar o gate.
- Testes cobrem permitir, negar, expirar e alterar escopo.

---

## BETA-006 — Consolidar e validar respostas dos especialistas

Depende de: BETA-004 e BETA-005.

### Objetivo

Criar o componente que combina resultados, separa fatos de hipóteses e entrega uma única resposta confiável ao Chefe.

### Requisitos

- Deduplicar achados.
- Cruzar evidências de agentes diferentes.
- Detectar contradições.
- Calcular confiança consolidada.
- Acionar QA/Auditor quando houver conflito relevante ou risco alto.
- Nunca apresentar hipótese como fato.
- Produzir recomendação e próximo passo.

### Formato de saída

- entendimento;
- agentes acionados, quando relevante;
- resultado;
- evidências;
- causa provável;
- confiança;
- risco;
- recomendação;
- autorização necessária;
- próximo passo.

### Critérios de aceite

- Conflito entre DBA e Infra é exposto ou enviado ao QA.
- Falta de evidência impede conclusão categórica.
- Resultado técnico detalhado é resumido para o Chefe.
- Testes cobrem consenso, conflito e falha parcial.

---

## BETA-007 — Separar memória estratégica e memória técnica

Depende de: BETA-002.

### Objetivo

Impedir que a memória central do Beta seja contaminada por detalhes técnicos que pertencem aos especialistas.

### Requisitos

Memória do Beta deve conter:

- preferências do Chefe;
- objetivos;
- decisões;
- prioridades;
- regras operacionais;
- estrutura da equipe.

Memória especializada deve conter:

- fatos técnicos do domínio;
- ambiente;
- soluções recorrentes;
- limitações de ferramentas.

Integrar com `MemoryManager` e Hindsight sem criar um segundo sistema paralelo.

### Critérios de aceite

- Fato sobre preferência do Chefe vai para memória estratégica.
- Configuração de PostgreSQL vai para memória do DBA.
- Especialistas não escrevem diretamente na memória do Beta.
- Recuperação consulta o escopo correto.
- Testes validam roteamento e isolamento.

---

## BETA-008 — Criar teste end-to-end do MVP

Depende de: BETA-001 a BETA-007.

### Objetivo

Validar o primeiro fluxo completo do Beta sem depender de infraestrutura real.

### Cenário obrigatório

Entrada:

`Chefe, verifique por que o PostgreSQL está lento.`

Respostas simuladas:

- Infra: CPU normal e disco com alta escrita.
- DBA: query longa mantendo lock e gerando I/O.
- Monitoring: pico de latência e escrita correlacionado ao início da query.

Resultado esperado:

- intenção classificada como diagnóstico;
- DBA, Infra e Monitoring selecionados;
- delegações somente leitura;
- execução paralela;
- consolidação das evidências;
- causa provável apresentada com confiança;
- recomendação de encerrar sessão ou aguardar;
- autorização solicitada antes de encerrar sessão;
- nenhuma ação de impacto executada automaticamente.

### Critérios de aceite

- Teste reproduzível e determinístico.
- Modo Hermes continua passando nos testes existentes.
- Falha de um especialista é tratada.
- Contradição aciona validação.
- Documentação mostra como executar o cenário.

---

## Instruções para Codex em modo goal

1. Trabalhe apenas na branch `beta/core-orchestrator`.
2. Execute as tarefas na ordem das dependências.
3. Uma issue deve produzir um PR ou commit isolado e testável.
4. Não avance quando os testes da etapa atual falharem.
5. Preserve compatibilidade com o upstream Hermes.
6. Evite grandes refatorações não relacionadas ao objetivo.
7. Documente decisões arquiteturais relevantes.
8. Ao concluir cada etapa, informe arquivos alterados, testes executados e limitações restantes.
