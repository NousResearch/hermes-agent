# Reasoning echo

Replayed assistant messages keep reasoning_content, so llama-server's
--reasoning-preserve flag reuses the prompt cache across turns.

No config keys. The profile sets echo_reasoning_content.
