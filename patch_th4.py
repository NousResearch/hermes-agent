import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

fixed = """                extra = tc.get("extra_content", {})
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]"""

code = re.sub(r'                extra = tc\.get\("extra_content", \{\}\).*?thought_sig = extra\["thought_signature"\]', fixed, code, flags=re.DOTALL)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Indentation fixed.")
