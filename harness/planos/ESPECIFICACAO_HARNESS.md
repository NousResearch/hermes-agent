# Especificação do Harness — Hermes Agent

**Versão:** 0.1.0  
**Data:** 2026-08-10  
**Localização:** `hermes-agent/harness/` (decisão D-001)

---

## Objetivo

Operar e manter o checkout local Hermes Agent (com patches **Hermes One**) de forma **executiva, segura e auditável** — via skills modulares, fluxos verificáveis e checkpoints, sem violar as invariantes de `AGENTS.md`.

---

## Escopo incluído

- Documentação de mapeamento e processos
- Catálogo de skills operacionais (sync, test, gateway, patches)
- Fluxos encadeados com aprovações humanas nos pontos críticos
- Registro de decisões e status
- Integração com ferramentas Cursor já disponíveis (Task, MCP, skills pessoais)

---

## Escopo excluído (sem aprovação explícita)

- Deploy em produção (VPS/Swarm)
- Contribuição PR upstream NousResearch
- Refactor de patches para plugin
- Instalação de novas dependências
- Alteração do core agent (`run_agent.py`, toolsets) além de bugfixes aprovados
- Telemetria ou analytics novos

---

## Restrições

| Restrição | Fonte |
|---|---|
| Prompt caching sagrado | AGENTS.md |
| Core estreito — capability nas bordas | AGENTS.md |
| Segredos só em `.env` / auth.json — nunca em docs | CLAUDE.md + harness-architect |
| Testes via `scripts/run_tests.sh` | AGENTS.md |
| Sem `git add .` | cursor-home rules |
| Windows PowerShell-safe para comandos usuário | CLAUDE.md |
| Deploy EasyPanel só via UI; Swarm separado | CLAUDE.md |

---

## Defaults provisórios (D-003..D-006)

| Decisão | Default |
|---|---|
| Objetivo | Operação Hermes One local |
| Patches | Manter manuais até pluginização |
| Superfície | Desktop + gateway Telegram |
| Ambiente | Windows nativo `%LOCALAPPDATA%\hermes\` |

---

## Arquitetura do harness

```
harness/
├── pesquisa/     # Evidências, mapa, deps, processos, patches
├── planos/       # Spec, skills, fluxos
├── guias/        # (futuro) regras engenharia local
├── registros/    # DECISOES.md, STATUS.md, LOG_EXECUCAO.md
├── rascunho/     # Experimentos temporários
└── entrega/      # Artefatos validados
```

**Skills Cursor:** viverão em `~/.cursor/skills/` (prefixo `hermes-*`) quando implementadas — o harness no repo documenta contratos; implementação fica no cursor-home ou skills pessoais.

---

## Critérios de sucesso

1. Merge de `main` sem perder patches Hermes One — verificável via checklist
2. Gateway operacional — logs acessíveis, zero secrets em docs
3. Testes verdes no slice afetado antes de considerar sync completo
4. Decisões registradas em `registros/DECISOES.md`

---

## Riscos

| Risco | Mitigação |
|---|---|
| Merge upstream sobrescreve patches | `hermes-patch-guard` + checklist PATCHES_LOCAIS |
| Commit acidental de harness em PR upstream | Branch local ou .git/info/exclude |
| Credential leak em logs/docs | Nunca dump auth.json; referenciar paths |
| pytest direto diverge CI | Wrapper obrigatório |

---

## Próximo portão

Confirmar defaults D-003..D-006 → implementar primeira skill (`hermes-patch-guard`).
