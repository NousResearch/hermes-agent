#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"
AGENTS_DIR="${TARGET_DIR}/.claude/agents"
SPECS_DIR="${TARGET_DIR}/.specs"
REQUIRED_AGENTS=("engineer" "plan" "review" "test" "architecture")
ERRORS=0

log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_ok() { echo -e "\033[32m[PASS]\033[0m $1"; }
log_err() { echo -e "\033[31m[FAIL]\033[0m $1"; ERRORS=$((ERRORS + 1)); }

log_info "Validando Agent Harness em: ${TARGET_DIR}"

# 1. Verificar diretório de agentes
if [[ ! -d "${AGENTS_DIR}" ]]; then
  log_err "Diretório ${AGENTS_DIR} não encontrado."
  exit 1
fi

# 2. Validar cada agente obrigatório
for agent in "${REQUIRED_AGENTS[@]}"; do
  file="${AGENTS_DIR}/${agent}.md"
  
  if [[ ! -f "${file}" ]]; then
    log_err "Agente ausente: ${file}"
    continue
  fi
  
  # Validar delimitadores YAML
  delim_count=$(grep -c "^---$" "${file}" || true)
  if [[ "${delim_count}" -lt 2 ]]; then
    log_err "${agent}.md: Frontmatter YAML inválido (requer delimitadores '---')."
    continue
  fi

  # Extrair frontmatter
  frontmatter=$(awk '/^---$/{p++;next}(p==1){print}' "${file}")

  # Checar campo name:
  name_val=$(echo "${frontmatter}" | grep -E "^name:[[:space:]]*${agent}$" || true)
  if [[ -z "${name_val}" ]]; then
    log_err "${agent}.md: Campo 'name:' ausente ou diferente de '${agent}'."
  fi

  # Checar campo description com PROACTIVELY
  if ! echo "${frontmatter}" | grep -qi "PROACTIVELY"; then
    log_err "${agent}.md: 'description:' não contém diretiva 'PROACTIVELY'."
  fi

  # Checar declaração de tools
  if ! echo "${frontmatter}" | grep -qE "^tools:"; then
    log_err "${agent}.md: Declaração de 'tools:' ausente no frontmatter."
  fi

  # Validações semânticas específicas
  if [[ "${agent}" == "plan" ]]; then
    if ! grep -q "\.specs/" "${file}"; then
      log_err "plan.md: Deve conter referência ao diretório de especificações (.specs/)."
    fi
  fi

  if [[ "${agent}" == "test" ]]; then
    if grep -q "{{TEST_CMD}}" "${file}"; then
      log_err "test.md: Placeholder {{TEST_CMD}} não foi resolvido para um comando real."
    fi
  fi

  if [[ "${agent}" == "architecture" ]]; then
    if ! grep -q "docs/architecture/" "${file}"; then
      log_err "architecture.md: Deve conter referência ao diretório docs/architecture/."
    fi
  fi

  log_ok "Agente ${agent}.md validado com sucesso."
done

# 3. Verificar CLAUDE.md e o protocolo de memória
CLAUDE_FILE="${TARGET_DIR}/CLAUDE.md"
MEMORY_DIR="${TARGET_DIR}/.claude/memory"

if [[ ! -f "${CLAUDE_FILE}" ]]; then
  log_err "CLAUDE.md não encontrado."
elif ! grep -q "## Memory Protocol" "${CLAUDE_FILE}"; then
  log_err "CLAUDE.md: seção '## Memory Protocol' ausente (obrigatória)."
elif ! grep -q '## Prompts' "${CLAUDE_FILE}"; then
  log_err "CLAUDE.md: regra de registro de prompts ('## Prompts') ausente na seção Memory Protocol."
elif ! grep -qi 'session summary' "${CLAUDE_FILE}"; then
  log_err "CLAUDE.md: regra de resumo de sessão ('Session summary') ausente na seção Memory Protocol."
else
  log_ok "CLAUDE.md contém '## Memory Protocol' com registro de prompts e resumo de sessão."
fi

if [[ ! -f "${MEMORY_DIR}/memory.md" ]]; then
  log_err "Memória de curto prazo ausente: ${MEMORY_DIR}/memory.md"
else
  log_ok "Memória de curto prazo presente."
fi

if ! compgen -G "${MEMORY_DIR}/[0-9]*-memory.md" >/dev/null; then
  log_err "Nenhum arquivo de memória de longo prazo (${MEMORY_DIR}/{YYYYMMDD}-memory.md) encontrado."
else
  log_ok "Memória de longo prazo presente."
fi

if [[ ! -f "${TARGET_DIR}/.claude/MEMORY.md" ]]; then
  log_err "Documentação do protocolo ausente: ${TARGET_DIR}/.claude/MEMORY.md"
else
  log_ok ".claude/MEMORY.md presente."
fi

# 4. Verificar diretório .specs
if [[ ! -d "${SPECS_DIR}" ]]; then
  log_err "Diretório ${SPECS_DIR} ausente."
else
  log_ok "Diretório de especificações ${SPECS_DIR} presente."
fi

# 5. Resultado final
echo "----------------------------------------"
if [[ ${ERRORS} -eq 0 ]]; then
  log_ok "Todos os critérios do Harness foram atendidos com sucesso!"
  exit 0
else
  log_err "Validação falhou com ${ERRORS} erro(s)."
  exit 1
fi
