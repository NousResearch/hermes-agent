import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# Instead of using the strong SDK types, we can pass raw dictionaries for the history
# Since the google-genai SDK `generate_content` method accepts raw dicts for contents:
old_tc = """                # Manually inject thought_signature into the protobuf dict if it exists 
                if thought_signature:
                    if hasattr(call_part, "function_call"):
                        try:
                            setattr(call_part.function_call, "thought_signature", thought_signature)
                        except:
                            pass
                parts.append(call_part)"""
                
new_tc = """                # Convert to raw dictionary to bypass pydantic validation of missing fields
                part_dict = {"function_call": {"name": func_name, "args": func_args}}
                if thought_signature:
                    # In some API versions this is passed alongside name/args, in others under a different key
                    # We inject it directly into the dictionary.
                    part_dict["function_call"]["thought_signature"] = thought_signature
                parts.append(part_dict)"""

code = code.replace(old_tc, new_tc)

old_text = """        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part.from_text(text=part["text"]))"""

new_text = """        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append({"text": msg["content"]})
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append({"text": part["text"]})"""
code = code.replace(old_text, new_text)

old_resp = """        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))"""

new_resp = """        if role == "tool":
            parts.append({"function_response": {"name": msg.get("name", "unknown_tool"), "response": {"result": msg.get("content", "")}}})"""
code = code.replace(old_resp, new_resp)

old_content = """        if parts:
            contents.append(types.Content(role=g_role, parts=parts))"""

new_content = """        if parts:
            contents.append({"role": g_role, "parts": parts})"""
code = code.replace(old_content, new_content)

with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Rewritten to dicts.")
