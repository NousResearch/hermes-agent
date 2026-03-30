import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

old_code = """        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                parts.append(types.Part.from_function_call(name=func_name, args=func_args))
                
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))"""

new_code = """        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                parts.append(types.Part.from_text(text=f"Action: I called {func_name} with arguments {func_args}"))
                
        if role == "tool":
            role = "user"
            parts.append(types.Part.from_text(text=f"Observation from {msg.get('name', 'tool')}: {msg.get('content', '')}"))"""

code = code.replace(old_code, new_code)
with open("agent/google_adapter.py", "w") as f:
    f.write(code)
print("Done")
