import re

with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "r") as f:
    code = f.read()

# Instead of using types.Part.from_function_call, we just bypass the whole Pydantic typing system for that specific field
# by using the raw dictionary directly in the underlying parts list. 
# Wait, the google-genai SDK converts the objects to dictionaries underneath. 
# We can just pass raw dictionaries!
old_loop = """        # Assistant Tool Calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                
                # We construct the Part using kwargs and allow extra fields for thought_signature
                extra = getattr(tc, "extra_content", None) or tc.get("extra_content", {})
                if extra and hasattr(extra, "model_dump"):
                    extra = extra.model_dump()
                    
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]
                elif hasattr(tc, "thought_signature"):
                    thought_sig = getattr(tc, "thought_signature")
                elif hasattr(tc, "__dict__"):
                    thought_sig = tc.__dict__.get("thought_signature")

                # The google-genai SDK uses Pydantic. We can pass a raw dict using the internal API structure if we use model_construct, 
                # or just use the kwargs dictionary to instantiate it if extra fields are allowed.
                # Let's try sending thought_signature directly in the function_call dict if we bypass pydantic.
                # Actually, the simplest way is to just let Gemini 3.1 Pro run without the thought signature for now 
                # by patching the tool output history so Gemini doesn't complain, OR we send it as a raw dict in `genai.Client`.
                
                fc = types.FunctionCall(name=func_name, args=func_args)
                
                # Hack: Add it to __pydantic_extra__ so it serializes!
                if thought_sig:
                    if not hasattr(fc, "__pydantic_extra__") or fc.__pydantic_extra__ is None:
                        fc.__pydantic_extra__ = {}
                    fc.__pydantic_extra__["thought_signature"] = thought_sig
                
                parts.append(types.Part(function_call=fc))"""

new_loop = """        # Assistant Tool Calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                
                extra = getattr(tc, "extra_content", None) or tc.get("extra_content", {})
                if extra and hasattr(extra, "model_dump"):
                    extra = extra.model_dump()
                    
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]
                elif hasattr(tc, "thought_signature"):
                    thought_sig = getattr(tc, "thought_signature")
                elif hasattr(tc, "__dict__"):
                    thought_sig = tc.__dict__.get("thought_signature")

                # Build a raw dict instead of a typed object to bypass Pydantic stripping unknown fields
                fc_dict = {"name": func_name, "args": func_args}
                if thought_sig:
                    fc_dict["thought_signature"] = thought_sig
                    
                # Under the hood, genai SDK accepts raw dicts for Content/Part objects
                parts.append({"functionCall": fc_dict})"""

code = code.replace(old_loop, new_loop)

old_text = """        # User/Assistant Message Content
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part.from_text(text=part["text"]))"""

new_text = """        # User/Assistant Message Content
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append({"text": msg["content"]})
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append({"text": part["text"]})"""
                    
code = code.replace(old_text, new_text)

old_tool = """        # Tool Response
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))"""

new_tool = """        # Tool Response
        if role == "tool":
            parts.append({"functionResponse": {"name": msg.get("name", "unknown_tool"), "response": {"result": msg.get("content", "")}}})"""

code = code.replace(old_tool, new_tool)

old_content = """        if parts:
            contents.append(types.Content(role=g_role, parts=parts))"""

new_content = """        if parts:
            contents.append({"role": g_role, "parts": parts})"""
            
code = code.replace(old_content, new_content)

with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "w") as f:
    f.write(code)
print("Patched google_adapter to use pure raw dictionaries")
