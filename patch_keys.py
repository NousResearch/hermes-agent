import re

with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# Try changing "thought_signature" to "thoughtSignature" inside the functionCall dictionary just in case it maps directly to the raw API REST payload casing!
old_fc = """                if thought_sig:
                    fc_dict["thought_signature"] = thought_sig"""
                    
new_fc = """                if thought_sig:
                    fc_dict["thought_signature"] = thought_sig
                    fc_dict["thoughtSignature"] = thought_sig"""

code = code.replace(old_fc, new_fc)

# Also let's fix extraction from response
old_ext = """                if hasattr(part.function_call, "__pydantic_extra__") and part.function_call.__pydantic_extra__:
                    thought_sig = part.function_call.__pydantic_extra__.get("thought_signature")
                elif hasattr(part.function_call, "thought_signature"):
                    thought_sig = part.function_call.thought_signature"""

new_ext = """                if hasattr(part.function_call, "__pydantic_extra__") and part.function_call.__pydantic_extra__:
                    thought_sig = part.function_call.__pydantic_extra__.get("thought_signature") or part.function_call.__pydantic_extra__.get("thoughtSignature")
                elif hasattr(part.function_call, "thought_signature"):
                    thought_sig = part.function_call.thought_signature
                elif hasattr(part.function_call, "thoughtSignature"):
                    thought_sig = part.function_call.thoughtSignature"""
                    
code = code.replace(old_ext, new_ext)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)

print("Patched keys.")
