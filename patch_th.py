import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old_thought = """                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_signature = extra["thought_signature"]"""
new_thought = """                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_signature = extra["thought_signature"]
                elif hasattr(tc, "__dict__"):
                    thought_signature = tc.__dict__.get("thought_signature")"""
code = code.replace(old_thought, new_thought)

old_tc = """                    extra_content={"thought_signature": getattr(part.function_call, "thought_signature", None)} if hasattr(part.function_call, "thought_signature") else None
                ))"""
new_tc = """                    extra_content={"thought_signature": getattr(part.function_call, "thought_signature", getattr(part, "thought_signature", None))}
                ))"""
code = code.replace(old_tc, new_tc)

old_dict = """                thought_signature = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_signature = extra["thought_signature"]
                elif hasattr(tc, "__dict__"):
                    thought_signature = tc.__dict__.get("thought_signature")"""
                    
new_dict = """                thought_signature = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_signature = extra["thought_signature"]
                elif hasattr(tc, "extra_content") and isinstance(tc.extra_content, dict):
                    thought_signature = tc.extra_content.get("thought_signature")
                elif hasattr(tc, "thought_signature"):
                    thought_signature = getattr(tc, "thought_signature")
                elif hasattr(tc, "__dict__"):
                    thought_signature = tc.__dict__.get("thought_signature")
"""
code = code.replace(old_dict, new_dict)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Patched google_adapter thought signature.")
