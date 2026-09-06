cd "$HOME/AppData/Local/hermes"
echo "===== validate reasoning_effort valid values in installed source ====="
grep -rihoE "reasoning_effort[\"']?\s*[:=].{0,80}" hermes-agent/ 2>/dev/null | grep -iE "ultra|low|medium|high|max|enum|valid|Literal" | head -8
echo "--- search for the literal 'ultra' as an effort token ---"
search=$(grep -rl "reasoning_effort" hermes-agent/ 2>/dev/null | head -5)
for f in $search; do grep -nE "ultra|xhigh|Literal\[|EFFORT|effort.*=.*\[" "$f" 2>/dev/null | head -4 | sed "s|$f|...|"; done
echo
echo "===== confirm NO openrouter/anthropic/openai key active in .env (names only) ====="
grep -iE "^(OPENROUTER|ANTHROPIC|OPENAI|ZAI|KIMI|MINIMAX|GROK|XAI)_" .env | sed -E 's/=.*/=<present>/' || echo "  none active"
echo "active (uncommented) .env keys total:"; grep -cE "^[A-Za-z_][A-Za-z0-9_]*=" .env
