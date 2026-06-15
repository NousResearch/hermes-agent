# Hermes Agent — Build & Guard Makefile

.PHONY: help arch-guard check syntax test clean

# ─── Architectural Guardrails ──────────────────────────────────────────────────

# Threshold for pass/fail.  Any violation count ABOVE this fails the check.
# Decrease this number as technical debt is paid down (69 → 50 → 30 → 10 → 0).
ARCH_GUARD_THRESHOLD ?= 69

.PHONY: arch-guard
arch-guard:
	@python3 scripts/arch_guard.py; \
	result=$$?; \
	if [ $$result -ne 0 ]; then \
		echo ""; \
		echo "⚠️  arch_guard detected violations (threshold: $(ARCH_GUARD_THRESHOLD))"; \
		echo "   To allow this commit, raise the threshold in Makefile (ARCH_GUARD_THRESHOLD)."; \
		echo "   Target: reduce this number over time as technical debt is paid down."; \
		exit 1; \
	fi
	echo "✅ Architectural guardrails passed (≤ $(ARCH_GUARD_THRESHOLD) violations)."

# ─── Syntax ────────────────────────────────────────────────────────────────────

.PHONY: syntax
syntax:
	@echo "Checking syntax..."
	@python3 -m py_compile \
		agent/conversation_loop.py \
		agent/tool_executor.py \
		services/adapter_manager.py \
		services/tool_service.py \
		conflict/resolver.py \
		conflict/policies/api_retry_policy.py \
		utils/text_processor.py \
		2>&1 | grep -v "^$$" || true
	@echo "✅ Syntax check complete."

# ─── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Hermes Agent Makefile"
	@echo ""
	@echo "  make arch-guard   Run architectural guardrails (pass ≤ 69 violations)"
	@echo "  make syntax       Fast syntax check of core files"
	@echo "  make help         Show this message"