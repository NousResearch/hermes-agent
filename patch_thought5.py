import re

with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "r") as f:
    code = f.read()

# We need to override the class behavior differently to guarantee the JSON gets sent exactly.
# Let's bypass `google.genai` model construction entirely for the `types.FunctionCall` part and just give it the dict.
# Wait, pydantic strictly validates children models. We patched earlier to use raw dictionaries but pydantic rejected them.
# The proper way to do this in `google.genai` is to just use the dictionary directly inside the payload when calling `generate_content`
# Ah, if we pass `contents=[{'role': 'user', 'parts': [...]}]` it fails.

# But wait, look at the error message: "Function call is missing a thought_signature in functionCall parts."
# If `fc.__pydantic_extra__` didn't work, we can just add `thought_signature` as an actual attribute and dump it manually.
# Let's try passing the raw request to the HTTP client using the `google.genai` `http_options`.
# No, `google.genai` handles all the JSON serialization.

# Let's inject a custom encoder for types.FunctionCall
new_hack = """                fc = types.FunctionCall(name=func_name, args=func_args)
                if thought_sig:
                    fc.thought_signature = thought_sig
                    # Hack for serialization: inject it into the Pydantic dictionary representation directly.
                    if hasattr(fc, "__dict__"):
                        fc.__dict__["thought_signature"] = thought_sig
                    if hasattr(fc, "model_extra"):
                        if fc.model_extra is None:
                            fc.model_extra = {}
                        fc.model_extra["thought_signature"] = thought_sig
                        fc.model_extra["thoughtSignature"] = thought_sig
                parts.append(types.Part(function_call=fc))"""

code = re.sub(r'                # Use model_construct.*?parts\.append\(types\.Part\.model_construct\(function_call=fc\)\)', new_hack, code, flags=re.DOTALL)

with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "w") as f:
    f.write(code)

