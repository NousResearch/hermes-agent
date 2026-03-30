import re
with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "r") as f:
    code = f.read()

old_call = """                func_name = tc["function"]["name"]
                # Parse arguments properly if string
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                parts.append(types.Part.from_function_call(name=func_name, args=func_args))"""

new_call = """                func_name = tc["function"]["name"]
                # Parse arguments properly if string
                raw_args = tc["function"]["arguments"]
                func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                
                extra = getattr(tc, "extra_content", None) or tc.get("extra_content")
                if extra and hasattr(extra, "model_dump"):
                    extra = extra.model_dump()
                thought_signature = None
                if isinstance(extra, dict) and "thought_signature" in extra:
                    thought_signature = extra["thought_signature"]
                    
                call_part = types.Part.from_function_call(name=func_name, args=func_args)
                
                # Manually inject thought_signature into the protobuf dict if it exists 
                # (since google-genai 0.2.x doesn't fully expose thought_signature in the factory method)
                if thought_signature:
                    if hasattr(call_part, "function_call"):
                        call_part.function_call.thought_signature = thought_signature
                parts.append(call_part)"""

code = code.replace(old_call, new_call)

old_norm = """        for part in response.candidates[0].content.parts:
            if part.function_call:
                message.tool_calls.append(SimpleNamespace(
                    id="call_" + str(uuid.uuid4())[:8],
                    type="function",
                    function=SimpleNamespace(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    )
                ))
                finish_reason = "tool_calls\"\"\""""

# Let's fix the normalization logic
old_norm = """        for part in response.candidates[0].content.parts:
            if part.function_call:
                message.tool_calls.append(SimpleNamespace(
                    id="call_" + str(uuid.uuid4())[:8],
                    type="function",
                    function=SimpleNamespace(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    )
                ))
                finish_reason = "tool_calls\"\"\""""

# simpler regex replacement for norm
code = re.sub(
    r'message\.tool_calls\.append\(SimpleNamespace\([\s\S]*?arguments=json\.dumps\(dict\(part\.function_call\.args\)\)\n\s*\)\n\s*\)\)',
    """message.tool_calls.append(SimpleNamespace(
                    id="call_" + str(uuid.uuid4())[:8],
                    type="function",
                    function=SimpleNamespace(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    ),
                    extra_content={"thought_signature": getattr(part.function_call, "thought_signature", None)} if hasattr(part.function_call, "thought_signature") else None
                ))""",
    code
)

with open("/home/leo_dwelon_com/.openclaw/workspace/hermes-agent/agent/google_adapter.py", "w") as f:
    f.write(code)
