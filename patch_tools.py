import re

with open("run_agent.py", "r") as f:
    code = f.read()

# Since we cannot inject `thought_signature` back into the API call easily via the official SDK objects,
# the simplest and most reliable way to avoid the Gemini 3.x thought_signature bug on subsequent turns 
# is to NOT push the tool calls back as assistant/tool pairs into the context history. 
# Instead, we just append the output directly as a user observation!
# This breaks the strict OpenAI tool loop pattern but saves the context completely for Gemini.
# Actually, the error says it's missing in position 2. 
# Position 0: user "echo hello"
# Position 1: assistant tool_call "terminal"
# Position 2: tool "hello"
# If we just replace the assistant tool_call with text "I ran the terminal tool..."
# No, `run_agent.py` loops and appends `assistant_msg` and `tool_result_message`.
# Let's fix this in `google_adapter.py` by intercepting tool calls in history and rewriting them to standard text!

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

                fc = types.FunctionCall(name=func_name, args=func_args)
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
                parts.append(types.Part(function_call=fc))
                
        # Tool Response
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))"""

new_tool = """        # If Gemini complains about missing thought_signatures in history, we can just flatten past tool calls into text.
        # But wait, Gemini 1.5 doesn't complain, only 3.x with reasoning enabled.
        # Since we use the official SDK, let's just convert past tool calls into structured text to bypass the validation error!
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                parts.append(types.Part.from_text(text=f"Action: I called {func_name} with arguments {func_args}"))
                
        # Tool Response
        if role == "tool":
            parts.append(types.Part.from_text(text=f"Observation from {msg.get('name', 'tool')}: {msg.get('content', '')}"))
            role = "user"  # Override to user so Gemini accepts the observation"""

with open("agent/google_adapter.py", "w") as f:
    f.write(code.replace(old_tool, new_tool))

print("Tool calls flattened to text.")
