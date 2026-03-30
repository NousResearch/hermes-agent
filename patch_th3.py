import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# When we send the functionCall dict, Gemini expects the `thought_signature` or `thoughtSignature`?
# According to the error: "missing a thought_signature in functionCall parts."
# But maybe we need to fetch it correctly from the input dict? 
# Wait, look at `tc["function"]["name"]`... `tc` is a dict, not an object.
# Yes, `getattr(tc, "extra_content")` fails on a dict.
# Let's fix the extraction of extra_content.

new_extraction = """                extra = tc.get("extra_content", {})
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]"""

code = re.sub(r'extra = getattr.*?thought_sig = tc\.__dict__\.get\("thought_signature"\)', new_extraction, code, flags=re.DOTALL)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Patched extraction")
