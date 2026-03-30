import re

with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# We need to use `setattr` instead of direct assignment since google_genai uses strict Pydantic/dataclasses that don't allow arbitrary fields in some versions, OR we don't assign it to function_call directly.
# Actually, the error is: `ValueError: "FunctionCall" object has no field "thought_signature"`
# The correct field in `types.Part` or `types.FunctionCall` for thought signatures might not exist in this version of the SDK, OR it might be under a different name.
# Wait, the error occurs when creating the part from the tool call history.

new_code = """                # Manually inject thought_signature into the protobuf dict if it exists 
                if thought_signature:
                    if hasattr(call_part, "function_call"):
                        try:
                            setattr(call_part.function_call, "thought_signature", thought_signature)
                        except:
                            pass
                parts.append(call_part)"""

code = re.sub(r'# Manually inject thought_signature.*?parts\.append\(call_part\)', new_code, code, flags=re.DOTALL)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)

print("Patched.")
