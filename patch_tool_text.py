import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old_tool = """        # Assistant Tool Calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                
                extra = tc.get('extra_content', {})
                thought_sig = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_sig = extra["thought_signature"]

                # Use model_construct to bypass init validation, passing thought_signature
                fc = types.FunctionCall.model_construct(name=func_name, args=func_args)
                if thought_sig:
                    fc.thought_signature = thought_sig
                    if not hasattr(fc, "__pydantic_extra__") or fc.__pydantic_extra__ is None:
                        fc.__pydantic_extra__ = {}
                    fc.__pydantic_extra__["thought_signature"] = thought_sig
                    fc.__pydantic_extra__["thoughtSignature"] = thought_sig
                    
                parts.append(types.Part.model_construct(function_call=fc))
                
        # Tool Response
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))"""

new_tool = """        # Assistant Tool Calls (Flattened to avoid thought_signature schema errors on Gemini 3.x)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                parts.append(types.Part.from_text(text=f"Action: I called {func_name} with arguments {func_args}"))
                
        # Tool Response
        if role == "tool":
            role = "user"  # Override to user
            parts.append(types.Part.from_text(text=f"Observation from {msg.get('name', 'tool')}: {msg.get('content', '')}"))"""

if old_tool in code:
    code = code.replace(old_tool, new_tool)
    with open("agent/google_adapter.py", "w") as f:
        f.write(code)
    print("Patched.")
else:
    print("Failed")
