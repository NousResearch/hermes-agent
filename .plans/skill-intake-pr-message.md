Título sugerido
fix: harden skill intake pipeline and preserve macOS gateway wrapper

Resumo
Este PR consolida duas linhas de correção que ficaram no mesmo commit local:

1. Skill intake
- adiciona o built-in tool `skill_intake`
- expõe `skill_intake` no toolset `skills`
- implementa os modos `audit`, `convert`, `re-audit` e `admit`
- corrige o contrato/schema para refletir os quatro modos realmente suportados
- adiciona testes unitários, integration-style e um teste hermético do wrapper/plugin local

2. Gateway macOS / higiene operacional
- preserva o wrapper `~/.hermes/scripts/hermes-with-keychain.sh` na geração do LaunchAgent quando aplicável
- força descoberta de plugin slash commands antes de autocomplete/menu/registro
- atualiza lockfiles relacionados
- remove resíduo legado do WhatsApp bridge

Motivação
- evitar perda silenciosa do wrapper Keychain em instâncias macOS com launchd
- tornar o pipeline de skill intake utilizável de ponta a ponta
- alinhar o schema do tool com a implementação real
- remover um teste dependente do ambiente local e torná-lo reproduzível em CI

Principais mudanças
- `tools/skill_intake_tool.py`
  - implementação do pipeline completo de intake
  - correção do texto do schema/documentação interna
- `toolsets.py`
  - inclui `skill_intake` no toolset `skills`
- `tests/tools/test_skill_intake_tool.py`
  - cobertura unitária do tool
- `tests/tools/test_skill_intake_tool_integration.py`
  - cobertura do fluxo convert -> re-audit -> admit
- `tests/tools/test_skill_intake_plugin_local.py`
  - wrapper/plugin testado de forma hermética, sem depender de `~/.hermes/plugins`
- `hermes_cli/gateway.py`
  - prioriza o wrapper Keychain ao gerar o LaunchAgent no macOS
- `hermes_cli/commands.py` + `tests/hermes_cli/test_commands.py`
  - descoberta de plugins antes da exposição de slash commands

Validação
- `python -m pytest tests/hermes_cli/test_commands.py -q`
- `python -m pytest tests/tools/test_skill_intake_tool.py tests/tools/test_skill_intake_tool_integration.py tests/tools/test_skill_intake_plugin_local.py -q`

Resultados observados
- `tests/hermes_cli/test_commands.py`: 111 passed
- skill intake suite: 26 passed

Riscos / observações
- o commit local atual mistura skill intake + hardening do gateway macOS + limpeza de legado WhatsApp
- para um PR mais limpo, pode valer separar em dois PRs:
  - PR A: skill intake
  - PR B: gateway macOS + plugin command discovery + cleanup operacional

Checklist
- [x] schema alinhado com a implementação
- [x] teste de plugin tornou-se hermético
- [x] toolset `skills` expõe `skill_intake`
- [x] regressão de comandos/plugin coberta por teste
- [x] wrapper Keychain preservado na geração do LaunchAgent macOS
