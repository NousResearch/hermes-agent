import re
with open("agent/google_adapter.py", "r") as f:
    code = f.read()

# The dict trick failed pydantic validation inside `generate_content`.
# The real issue: `types.Part.from_function_call(name=func_name, args=func_args)` works, BUT the `types.FunctionCall` object doesn't have a `thought_signature` attribute in the pydantic model.
# Actually, the error says:
# "Function call is missing a thought_signature in functionCall parts." -> This is a server-side API error from Gemini (400 BAD REQUEST), not a python crash!
# Gemini requires the `thought_signature` field in the `functionCall` dictionary when sending the history back to the model.
# Because the official google-genai SDK's `types.FunctionCall` schema doesn't seem to include `thought_signature` yet (it's new for Gemini 3), the serialization drops it.

# To fix this, we have to bypass `types.FunctionCall` entirely and pass raw JSON to the underlying REST client, OR monkey-patch the `types.FunctionCall` pydantic model to allow extra fields.
# Since we are using `google.genai`, we can pass raw dicts, but we need to construct them correctly. Wait, the validation error said `contents.Content Input should be a valid dictionary...` meaning `contents` must be a list of `types.Content` objects, but inside those we can use dicts? No, `types.Content` expects `types.Part`.

# Let's override the `model_dump` method or use `types.Part.model_construct` to skip validation.
code = """from google import genai
from google.genai import types
from types import SimpleNamespace
import uuid
import json

def build_google_client(api_key: str):
    return genai.Client(api_key=api_key)

def convert_messages_to_google(messages: list):
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_instruction = msg.get("content")
            continue
            
        parts = []
        
        # User/Assistant Message Content
        if isinstance(msg.get("content"), str) and msg["content"].strip():
            parts.append(types.Part.from_text(text=msg["content"]))
        elif isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part.from_text(text=part["text"]))
        
        # Assistant Tool Calls
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
                
                parts.append(types.Part(function_call=fc))
                
        # Tool Response
        if role == "tool":
            parts.append(types.Part.from_function_response(
                name=msg.get("name", "unknown_tool"),
                response={"result": msg.get("content", "")}
            ))
            
        g_role = "user" if role in ("user", "tool") else "model"
        
        if parts:
            contents.append(types.Content(role=g_role, parts=parts))
            
    return contents, system_instruction

def build_google_kwargs(model: str, messages: list, tools: list = None, max_tokens: int = None):
    contents, system_instruction = convert_messages_to_google(messages)
    
    g_tools = []
    if tools:
        for t in tools:
            f = t["function"]
            params = f.get("parameters")
            if params and "type" in params:
                params["type"] = params["type"].upper()
                
            fd = types.FunctionDeclaration(
                name=f["name"],
                description=f.get("description", ""),
                parameters=params
            )
            g_tools.append(fd)
            
    req_model = model.replace("google/", "") if model.startswith("google/") else model
    
    config_kwargs = {"temperature": 0.1}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if max_tokens:
        config_kwargs["max_output_tokens"] = max_tokens
    if g_tools:
        config_kwargs["tools"] = [types.Tool(function_declarations=g_tools)]
        
    config = types.GenerateContentConfig(**config_kwargs)
        
    return {
        "model": req_model,
        "contents": contents,
        "config": config
    }

def normalize_google_response(response, model: str):
    message = SimpleNamespace(role="assistant", content=None, tool_calls=[])
    if response.text:
        message.content = response.text
        
    finish_reason = "stop"
    
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                
                thought_sig = None
                if hasattr(part.function_call, "__pydantic_extra__") and part.function_call.__pydantic_extra__:
                    thought_sig = part.function_call.__pydantic_extra__.get("thought_signature")
                elif hasattr(part.function_call, "thought_signature"):
                    thought_sig = part.function_call.thought_signature
                elif hasattr(part, "__pydantic_extra__") and part.__pydantic_extra__:
                    # sometimes the signature is on the part level
                    thought_sig = part.__pydantic_extra__.get("thought_signature")
                
                message.tool_calls.append(SimpleNamespace(
                    id="call_" + str(uuid.uuid4())[:8],
                    type="function",
                    function=SimpleNamespace(
                        name=part.function_call.name,
                        arguments=json.dumps(dict(part.function_call.args))
                    ),
                    extra_content={"thought_signature": thought_sig} if thought_sig else None
                ))
                finish_reason = "tool_calls"

    # Usage tracking
    prompt_tokens = 0
    completion_tokens = 0
    if getattr(response, 'usage_metadata', None):
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0
    
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, 
            completion_tokens=completion_tokens, 
            total_tokens=prompt_tokens+completion_tokens
        )
    ), finish_reason
"""
with open("agent/google_adapter.py", "w") as f:
    f.write(code)
