import re
with open("run_agent.py", "r") as f:
    code = f.read()

old_norm = """                elif self.api_mode == "google_genai":
                    # For non-streaming fallback, the normalization is already handled in _interruptible_api_call
                    # because it returns the SimpleNamespace wrapper. So we just unwrap it identically to chat_completions.
                    assistant_message = response.choices[0].message
                    finish_reason = response.choices[0].finish_reason
                    self._update_token_usage(response.usage)"""

new_norm = """                elif self.api_mode == "google_genai":
                    assistant_message = response.choices[0].message
                    finish_reason = response.choices[0].finish_reason"""

if old_norm in code:
    code = code.replace(old_norm, new_norm)
    with open("run_agent.py", "w") as f:
        f.write(code)
    print("Normalizer patched.")
